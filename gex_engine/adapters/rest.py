"""REST Adapter: ThetaData v3 REST API 経由でオプションチェーンを取得する。

PROJECT_CONTEXT v12 セクション7 の Adapter パターンに従い、
DataFetcher Protocol を満たす実装。

設計方針（v13 段階4 で確定）:
    - 同期 httpx（並列化は段階6 以降の最適化テーマ）
    - フィルタは一切実装しない（5 段階フィルタは yfinance 固有問題への
      対症療法、ThetaData では構造的に不要）
    - エラーは 4 区分に分類:
        SUCCESS:    HTTP 200
        NO_DATA:    HTTP 472（休場日等）→ 警告ログ + 空 DataFrame
        RETRYABLE:  HTTP 429/470/474/570/571 → 指数バックオフで最大 3 回
        FATAL:      上記以外 → 即 raise

データ取得フロー:
    1. /v3/option/snapshot/open_interest?symbol=X&expiration=*
    2. /v3/option/snapshot/greeks/implied_volatility?symbol=X&expiration=*
    3. (symbol, expiration, strike, right) で outer join
    4. 片側欠落（open_interest または implied_volatility が NaN）を警告ログ
       に記録し、計算可能なレコードのみに絞る
    5. 統一スキーマに dtype 整形して返す

参考:
    - https://docs.thetadata.us/operations/option_snapshot_open_interest.html
    - https://docs.thetadata.us/operations/option_snapshot_greeks_implied_volatility.html
    - https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Error-Codes.html
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from enum import Enum
from typing import Any

import httpx
import pandas as pd

from ..schema import REQUIRED_DTYPES, coerce_to_schema, empty_dataframe

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# エラー分類（公式エラーコードを 4 区分にマッピング）
# ──────────────────────────────────────────────────────────

class ThetaErrorCategory(Enum):
    """Theta Data の独自エラーコードを実装上の振る舞いで分類。

    公式の 14 コードを「アクション単位」で 4 区分に集約。
    コード単位で分岐するより保守性が高い。
    """
    SUCCESS = "success"          # 200
    NO_DATA = "no_data"          # 472
    RETRYABLE = "retryable"      # 429, 470, 474, 570, 571
    FATAL = "fatal"              # 404, 471, 473, 475, 476, 477, 478, 572


# 公式ドキュメント: https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Error-Codes.html
_HTTP_TO_CATEGORY: dict[int, ThetaErrorCategory] = {
    200: ThetaErrorCategory.SUCCESS,
    404: ThetaErrorCategory.FATAL,      # NO_IMPL: 古い Terminal or 不正リクエスト
    429: ThetaErrorCategory.RETRYABLE,  # OS_LIMIT: OS スロットリング
    470: ThetaErrorCategory.RETRYABLE,  # GENERAL: 一般エラー
    471: ThetaErrorCategory.FATAL,      # PERMISSION: プラン権限なし（Free→Standard 切替時の事故）
    472: ThetaErrorCategory.NO_DATA,    # NO_DATA: データなし（休場日等）
    473: ThetaErrorCategory.FATAL,      # INVALID_PARAMS: パラメータ不正
    474: ThetaErrorCategory.RETRYABLE,  # DISCONNECTED: MDDS 接続切れ
    475: ThetaErrorCategory.FATAL,      # TERMINAL_PARSE: パース失敗
    476: ThetaErrorCategory.FATAL,      # WRONG_IP: 127.0.0.1/localhost 混在
    477: ThetaErrorCategory.FATAL,      # NO_PAGE_FOUND: ページなし
    478: ThetaErrorCategory.FATAL,      # INVALID_SESSION_ID: Terminal 多重起動
    570: ThetaErrorCategory.RETRYABLE,  # LARGE_REQUEST: 要求過大
    571: ThetaErrorCategory.RETRYABLE,  # SERVER_STARTING: 再起動中
    572: ThetaErrorCategory.FATAL,      # UNCAUGHT_ERROR: 内部エラー
}


def classify_status(status_code: int) -> ThetaErrorCategory:
    """HTTP ステータスを 4 区分に分類する。未知のコードは FATAL 扱い。"""
    return _HTTP_TO_CATEGORY.get(status_code, ThetaErrorCategory.FATAL)


# ──────────────────────────────────────────────────────────
# 例外
# ──────────────────────────────────────────────────────────

class ThetaDataError(Exception):
    """REST Adapter からの例外の基底クラス。"""

    def __init__(self, message: str, status_code: int, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ThetaPermissionError(ThetaDataError):
    """471 PERMISSION: プラン権限なし。Free でも Option 系を叩いた時等。"""


class ThetaFatalError(ThetaDataError):
    """FATAL カテゴリ全般。リトライ不可、即停止。"""


class ThetaRetryExhaustedError(ThetaDataError):
    """RETRYABLE を最大回数試みたが回復しなかった。"""


# ──────────────────────────────────────────────────────────
# REST Adapter 本体
# ──────────────────────────────────────────────────────────

class ThetaRestAdapter:
    """ThetaData v3 REST API を叩く DataFetcher 実装。

    Theta Terminal がローカルで起動している前提（認証は Terminal 側で完結）。

    Attributes:
        source_name: "rest" 固定
        base_url: 既定 http://127.0.0.1:25503/v3
        timeout: HTTP タイムアウト秒
        max_retries: RETRYABLE エラーの最大リトライ回数
        retry_backoff_base: 指数バックオフの初期待ち時間（秒）
    """

    source_name: str = "rest"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:25503/v3",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_base: float = 1.0,
        client: httpx.Client | None = None,
    ):
        """Args:
            base_url: REST API のベース URL。
            timeout: 1 リクエストのタイムアウト（秒）。
            max_retries: RETRYABLE エラー時の最大リトライ回数。
            retry_backoff_base: 1 回目のバックオフ秒（2 回目は ×2、3 回目は ×4）。
            client: テスト時に外部から注入する httpx.Client。
                    None なら内部で生成する。
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self._client = client
        self._owned_client = client is None

    # ── ライフサイクル ──

    def _ensure_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        """内部生成した Client を閉じる。注入された Client は閉じない。"""
        if self._owned_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "ThetaRestAdapter":
        self._ensure_client()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── 公開メソッド（DataFetcher Protocol） ──

    def get_option_chain(
        self, symbol: str, as_of: date
    ) -> pd.DataFrame:
        """指定銘柄のオプションチェーンを取得する。

        Snapshot エンドポイントは「最新値」を返す仕様のため、
        as_of は実装上は使用しない（ログとメタ情報用）。
        正しいタイミング（cron で ET 17:30〜24:00）に呼ぶ責任は呼び出し側。

        Args:
            symbol: シンボル（例: "SPY"）
            as_of: 基準日（記録用、Snapshot API は最新値しか返さない）

        Returns:
            統一スキーマに準拠した DataFrame。NO_DATA 時は空 DataFrame。

        Raises:
            ThetaPermissionError: 471 PERMISSION（Free プランで Option 叩いた等）
            ThetaFatalError: FATAL カテゴリ全般
            ThetaRetryExhaustedError: RETRYABLE が回復せずリトライ枯渇
        """
        logger.info("ThetaRestAdapter: fetching %s as_of=%s", symbol, as_of)

        # ── 2 エンドポイントを順次叩く ──
        oi_df = self._fetch_open_interest(symbol)
        if oi_df.empty:
            logger.warning("OI snapshot returned empty for %s. Returning empty chain.", symbol)
            return empty_dataframe()

        iv_df = self._fetch_implied_volatility(symbol)
        if iv_df.empty:
            logger.warning("IV snapshot returned empty for %s. Returning empty chain.", symbol)
            return empty_dataframe()

        # ── 4 列キーで outer join ──
        merged = self._merge_oi_iv(oi_df, iv_df)

        # ── 統一スキーマに整形 ──
        merged["symbol"] = symbol
        return coerce_to_schema(merged[list(REQUIRED_DTYPES.keys())])

    # ── 内部: HTTP リクエスト ──

    def _request_csv(self, path: str, params: dict[str, Any]) -> str:
        """エンドポイントを叩いて CSV テキストを返す。

        format=csv で要求し、4 区分のエラー分類に従って処理する。
        RETRYABLE は指数バックオフで最大 self.max_retries 回再試行。

        Returns:
            CSV テキスト（NO_DATA の時は空文字列）

        Raises:
            ThetaPermissionError / ThetaFatalError / ThetaRetryExhaustedError
        """
        params = {**params, "format": "csv"}
        url = f"{self.base_url}{path}"
        client = self._ensure_client()

        last_exc: Exception | None = None
        last_body: str = ""
        last_status: int = 0

        for attempt in range(self.max_retries + 1):
            try:
                response = client.get(url, params=params)
            except httpx.HTTPError as e:
                # 通信レベルのエラー（接続失敗、タイムアウト等）はリトライ可能扱い
                last_exc = e
                logger.warning(
                    "HTTP error on attempt %d/%d for %s: %s",
                    attempt + 1, self.max_retries + 1, path, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_base * (2 ** attempt))
                    continue
                raise ThetaRetryExhaustedError(
                    f"HTTP error after {self.max_retries + 1} attempts: {e}",
                    status_code=0,
                ) from e

            category = classify_status(response.status_code)
            last_status = response.status_code
            last_body = response.text

            if category == ThetaErrorCategory.SUCCESS:
                return response.text

            if category == ThetaErrorCategory.NO_DATA:
                logger.warning(
                    "NO_DATA (472) for %s params=%s. body=%r",
                    path, params, response.text[:200],
                )
                return ""  # 呼び出し側で空 DataFrame を返す

            if category == ThetaErrorCategory.RETRYABLE:
                logger.warning(
                    "Retryable error %d on attempt %d/%d for %s. body=%r",
                    response.status_code, attempt + 1, self.max_retries + 1,
                    path, response.text[:200],
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_base * (2 ** attempt))
                    continue
                raise ThetaRetryExhaustedError(
                    f"Retryable error {response.status_code} did not recover "
                    f"after {self.max_retries + 1} attempts",
                    status_code=response.status_code,
                    body=response.text,
                )

            # FATAL カテゴリ
            if response.status_code == 471:
                raise ThetaPermissionError(
                    f"PERMISSION (471): Plan does not allow this endpoint. "
                    f"path={path} body={response.text[:200]!r}",
                    status_code=471,
                    body=response.text,
                )
            raise ThetaFatalError(
                f"Fatal error {response.status_code} for path={path}. "
                f"body={response.text[:200]!r}",
                status_code=response.status_code,
                body=response.text,
            )

        # ループを抜けた場合（理論的には到達しないが念のため）
        raise ThetaRetryExhaustedError(
            f"Unexpected fall-through after {self.max_retries + 1} attempts",
            status_code=last_status,
            body=last_body,
        )

    # ── 内部: 個別エンドポイント ──

    def _fetch_open_interest(self, symbol: str) -> pd.DataFrame:
        """OI snapshot を取得し、必要な列だけ取り出した DataFrame を返す。

        Returns:
            列: symbol, expiration, strike, right, open_interest
            空レスポンス時は空 DataFrame（列なし）
        """
        csv_text = self._request_csv(
            "/option/snapshot/open_interest",
            {"symbol": symbol, "expiration": "*"},
        )
        if not csv_text.strip():
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(csv_text))

        # 公式仕様: timestamp, symbol, expiration, strike, right, open_interest
        # 必要列のみ抽出（schema に必要な分）
        required = ["symbol", "expiration", "strike", "right", "open_interest"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ThetaFatalError(
                f"OI response missing expected columns: {sorted(missing)}. "
                f"Got: {list(df.columns)}",
                status_code=200,
                body=csv_text[:500],
            )

        return df[required].copy()

    def _fetch_implied_volatility(self, symbol: str) -> pd.DataFrame:
        """IV snapshot を取得し、必要な列だけ取り出した DataFrame を返す。

        Returns:
            列: symbol, expiration, strike, right, bid, ask,
                implied_volatility, underlying_price
            空レスポンス時は空 DataFrame（列なし）
        """
        csv_text = self._request_csv(
            "/option/snapshot/greeks/implied_volatility",
            {"symbol": symbol, "expiration": "*"},
        )
        if not csv_text.strip():
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(csv_text))

        # 公式の IV レスポンス列名は "implied_vol"。
        # 統一スキーマでは "implied_volatility" なのでリネーム。
        if "implied_vol" in df.columns and "implied_volatility" not in df.columns:
            df = df.rename(columns={"implied_vol": "implied_volatility"})

        required = [
            "symbol", "expiration", "strike", "right",
            "bid", "ask", "implied_volatility", "underlying_price",
        ]
        missing = set(required) - set(df.columns)
        if missing:
            raise ThetaFatalError(
                f"IV response missing expected columns: {sorted(missing)}. "
                f"Got: {list(df.columns)}",
                status_code=200,
                body=csv_text[:500],
            )

        return df[required].copy()

    # ── 内部: 結合 ──

    @staticmethod
    def _merge_oi_iv(
        oi_df: pd.DataFrame, iv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """OI と IV を 4 列キーで outer join し、片側欠落を警告ログに記録。

        論点3a/3b の決定:
            - outer join（片側欠落も診断対象に残す）
            - 結合後に「OI と IV の両方が揃わないレコード」は除外
              （GEX 計算に必須なので）
            - 除外件数を警告ログに出す

        Args:
            oi_df: 列 symbol, expiration, strike, right, open_interest
            iv_df: 列 symbol, expiration, strike, right, bid, ask,
                       implied_volatility, underlying_price

        Returns:
            両側揃ったレコードのみ。dtype は文字列のまま、coerce は呼び出し側。
        """
        key = ["symbol", "expiration", "strike", "right"]
        merged = pd.merge(
            oi_df, iv_df, on=key, how="outer", indicator=True,
        )

        n_oi_only = (merged["_merge"] == "left_only").sum()
        n_iv_only = (merged["_merge"] == "right_only").sum()
        n_both = (merged["_merge"] == "both").sum()

        if n_oi_only or n_iv_only:
            logger.warning(
                "OI/IV merge: both=%d, oi_only=%d (IV missing), iv_only=%d (OI missing). "
                "Dropping single-side records (GEX cannot be computed without both).",
                n_both, n_oi_only, n_iv_only,
            )

        clean = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        return clean.reset_index(drop=True)
