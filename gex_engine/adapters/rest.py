"""REST Adapter: ThetaData v3 REST API 経由でオプションチェーンを取得する。

PROJECT_CONTEXT の Adapter パターンに従い、DataFetcher Protocol を
満たす実装。

設計方針:
    - 同期 httpx（並列化は段階6 以降の最適化テーマ）
    - フィルタは一切実装しない（5 段階フィルタは yfinance 固有問題への
      対症療法、ThetaData では構造的に不要）。ただし IV 算出失敗行の
      除外は「フィルタ」ではなく数学的前提（gamma 計算に σ>0 が必須）。
    - エラーは 4 区分に分類:
        SUCCESS:    HTTP 200
        NO_DATA:    HTTP 472（休場日等）→ 警告ログ + 空 DataFrame
        RETRYABLE:  HTTP 429/470/474/570/571 → 指数バックオフで最大 3 回
        FATAL:      上記以外 → 即 raise

データ取得フロー（history ベース、DESIGN_history_rest_adapter.md）:
    1. as_of から取引日 T と OI 用日付を算出（calendar/on_date）
    2. /v3/option/history/greeks/eod（取引日 T）
       → IV 健全性フィルタ（implied_vol<=0 / iv_error==100 を除外）
    3. /v3/option/history/open_interest（T の翌営業日を date に渡す）
    4. (symbol, expiration, strike, right) で outer join
    5. 片側欠落を診断ログに記録し、両側揃ったレコードのみに絞る
    6. 統一スキーマに dtype 整形して返す

snapshot → history への移行理由:
    snapshot は「叩いた瞬間の最新値」を返すため cron 遅延でデータが
    汚染される。history/greeks/eod は ET 17:15 生成の確定 EOD レポートを
    返し、いつ叩いても同じ値（実行時刻非依存）。

参考:
    - https://docs.thetadata.us/operations/option_history_open_interest.html
    - https://docs.thetadata.us/operations/option_history_greeks_eod.html
    - https://docs.thetadata.us/Articles/Errors-Exchanges-Conditions/Error-Codes.html
"""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import date, timedelta
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
        """指定銘柄のオプションチェーンを取得する（history ベース）。

        history/greeks/eod と history/open_interest を取得し、
        4 列キーで結合して統一スキーマに整形する。

        日付の解決（DESIGN 3 / 4.1）:
            as_of は処理基準日（通常は run_daily.py が渡す today）。
            この関数が内部で取引日 T と OI 用日付を算出する:
              T       = as_of の直近過去営業日（_resolve_trade_date）
              oi_date = T の翌営業日（_next_business_day）
            greeks/eod は T を、open_interest は oi_date を使う。
            この日付の非対称性は OPRA の OI 報告構造に由来し（DESIGN 3.5）、
            ここ 1 箇所に集約することで管理する。

        Args:
            symbol: シンボル（例: "SPY"）
            as_of: 処理基準日。Adapter が営業日調整を行う。

        Returns:
            統一スキーマ（schema.REQUIRED_DTYPES）に準拠した DataFrame。
            データなし（休場日等）の場合は schema.empty_dataframe()。

        Raises:
            ThetaPermissionError: 471 PERMISSION
            ThetaFatalError: FATAL カテゴリ全般・カレンダー異常
            ThetaRetryExhaustedError: RETRYABLE 枯渇
        """
        # ── 取引日 T と OI 用日付の解決 ──
        trade_date = self._resolve_trade_date(as_of)
        oi_date = self._next_business_day(trade_date)
        logger.info(
            "ThetaRestAdapter: %s as_of=%s -> trade_date=%s, oi_date=%s",
            symbol, as_of, trade_date, oi_date,
        )

        # ── greeks/eod（取引日 T）──
        iv_df = self._fetch_greeks_eod(symbol, trade_date)
        if iv_df.empty:
            logger.warning(
                "greeks/eod returned empty for %s (trade_date=%s). "
                "Returning empty chain.",
                symbol, trade_date,
            )
            return empty_dataframe()

        # ── open_interest（T の翌営業日を date に渡す）──
        oi_df = self._fetch_open_interest(symbol, oi_date)
        if oi_df.empty:
            logger.warning(
                "open_interest returned empty for %s (oi_date=%s). "
                "Returning empty chain.",
                symbol, oi_date,
            )
            return empty_dataframe()

        # ── 4 列キーで outer join（片側欠落・Wall 欠落の診断を含む）──
        merged = self._merge_oi_iv(oi_df, iv_df)

        # 監査14: マッチゼロは異常。IV/OI が両方とも非空（上の空チェックを
        # 通過済み）なのに both 行がゼロ = キーが 1 つも一致しなかった。
        # 休場日の空とは別物（expiration 表記の不一致や、片方が古い
        # データである可能性）。empty を返す挙動自体は休場日と同じだが、
        # 原因究明できるよう異常として WARNING に明示する。
        if merged.empty:
            logger.warning(
                "OI/IV merge produced 0 rows for %s (trade_date=%s). "
                "Both sources had data but no keys matched -- "
                "possible expiration-format mismatch or stale data.",
                symbol, trade_date,
            )
            return empty_dataframe()

        # 監査15: symbol は「上書き」せず「検証」する。
        # merged の symbol は 4 列 join キーなので、正常時は引数 symbol と
        # 一致する。上書き（merged["symbol"] = symbol）にすると、上流が
        # 別銘柄のデータを返した場合に取り違えを隠蔽してしまう。
        # 不一致を検出したら即 FATAL。
        actual_symbols = set(merged["symbol"].unique())
        if actual_symbols != {symbol}:
            raise ThetaFatalError(
                f"symbol mismatch: requested {symbol!r}, but merged data "
                f"contains {sorted(actual_symbols)}. "
                f"Possible data mix-up from upstream endpoints.",
                status_code=200,
            )

        # ── 統一スキーマに整形 ──
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

    # ── 内部: 営業日カレンダー ──
    #
    # history への移行に伴い新設。snapshot 時代には不要だった。
    # 取引日 T の解決（_resolve_trade_date / _next_business_day）が
    # この低レベル取得関数を使う。
    #
    # 実 API レスポンス形式（2026-05-16 に on_date を実機ダンプして確認）:
    #     type,open,close
    #     "open","09:30:00","16:00:00"      ← 平日
    #     "full_close",,                    ← 休場日（open/close 空）
    # ヘッダ行あり・データ1行・末尾空行あり・値はダブルクオート囲み。

    # on_date / today が返す type の全集合（公式ドキュメントで確定）。
    # 未知の値が来たら遡及ループが誤動作するため、明示的に検証する。
    _CALENDAR_TYPES: frozenset[str] = frozenset(
        {"open", "early_close", "full_close", "weekend"}
    )

    def _fetch_calendar_on_date(self, target: date) -> str:
        """calendar/on_date を叩き、その日の市場スケジュール type を返す。

        この関数は「1 日分の type を引く」だけの低レベル関数。
        営業日の遡及は呼び出し側（_resolve_trade_date）の責務。

        Args:
            target: スケジュールを問い合わせる日付。

        Returns:
            type 文字列。_CALENDAR_TYPES のいずれか:
                "open"        通常営業日
                "early_close" 短縮営業日（取引日として扱える）
                "full_close"  終日休場
                "weekend"     土日

        Raises:
            ThetaFatalError:
                - レスポンスが空（calendar が引けない = システム異常。
                  OI の 472=休場日とは意味が異なり、遡及を続けては
                  いけないため即停止する）
                - データ行が無い / type 列が読めない
                - type が _CALENDAR_TYPES のいずれでもない
            ThetaPermissionError / ThetaRetryExhaustedError:
                _request_csv から伝播。
        """
        date_str = target.strftime("%Y%m%d")
        csv_text = self._request_csv(
            "/calendar/on_date", {"date": date_str}
        )

        # 空レスポンス: _request_csv は NO_DATA(472) で "" を返す。
        # OI/IV では 472=休場日で正常だが、カレンダーが引けないのは
        # システム異常。遡及ループに「weekend」と誤認させないため、
        # ここで明確に停止させる。
        if not csv_text.strip():
            raise ThetaFatalError(
                f"calendar/on_date returned empty for date={date_str}. "
                f"Calendar must always resolve; cannot continue.",
                status_code=200,
            )

        # csv.reader で軽量パース。3 列 1 行のため pd.read_csv は使わない。
        # ヘッダ行・末尾空行をスキップして最初のデータ行を取る。
        rows = [
            row for row in csv.reader(io.StringIO(csv_text))
            if row and any(cell.strip() for cell in row)
        ]
        if not rows:
            raise ThetaFatalError(
                f"calendar/on_date: no rows for date={date_str}. "
                f"body={csv_text[:200]!r}",
                status_code=200,
                body=csv_text[:500],
            )

        header = rows[0]
        if "type" not in header:
            raise ThetaFatalError(
                f"calendar/on_date: 'type' column not found for "
                f"date={date_str}. header={header}",
                status_code=200,
                body=csv_text[:500],
            )
        if len(rows) < 2:
            raise ThetaFatalError(
                f"calendar/on_date: header present but no data row for "
                f"date={date_str}. body={csv_text[:200]!r}",
                status_code=200,
                body=csv_text[:500],
            )

        type_idx = header.index("type")
        type_value = rows[1][type_idx].strip().strip('"').lower()

        if type_value not in self._CALENDAR_TYPES:
            raise ThetaFatalError(
                f"calendar/on_date: unknown schedule type "
                f"{type_value!r} for date={date_str}. "
                f"Expected one of {sorted(self._CALENDAR_TYPES)}.",
                status_code=200,
                body=csv_text[:500],
            )

        return type_value

    # 営業日とみなす type。early_close（短縮営業日）も取引日であり、
    # EOD レポートは生成されるため取引日に含める（DESIGN 3.3 step4）。
    _TRADING_DAY_TYPES: frozenset[str] = frozenset({"open", "early_close"})

    # カレンダー走査の上限日数（無限ループ防止のサーキットブレーカー）。
    # _resolve_trade_date（過去向き）と _next_business_day（未来向き）の
    # 両方で共用する。米国市場の最長連続休場は年末年始の土日 + 元日でも
    # 3〜4 日程度。10 はそれを十分上回る安全マージン。この上限に達する
    # ことは「真の異常」（カレンダー API の不整合等）を意味するため、
    # 値の精度は問われない（7 でも 14 でも正常系の挙動は不変）。
    _MAX_CALENDAR_SCAN_DAYS: int = 10

    def _resolve_trade_date(self, as_of: date) -> date:
        """as_of から「直近の過去営業日 T」を解決する。

        DESIGN 3.2 の決着に従い、当日 EOD は当てにせず、必ず
        as_of の前日から過去へ遡る。cron 実行（ET 17:30）と
        greeks/eod 生成（ET 17:15）の差が 15 分しかなく、当日 EOD が
        生成済みと仮定するのは危険なため。

        Args:
            as_of: 処理基準日（通常は run_daily.py が渡す today）。

        Returns:
            as_of より前の直近の取引日（open または early_close）。

        Raises:
            ThetaFatalError:
                _MAX_CALENDAR_SCAN_DAYS 日遡っても取引日が
                見つからない場合（カレンダー異常）。
            ThetaPermissionError / ThetaRetryExhaustedError:
                _fetch_calendar_on_date から伝播。
        """
        candidate = as_of - timedelta(days=1)
        for _ in range(self._MAX_CALENDAR_SCAN_DAYS):
            # 監査7: on_date のレスポンスに date 列が無いため、
            # 「どの日を問い合わせた結果か」は candidate をログに併記して残す。
            schedule_type = self._fetch_calendar_on_date(candidate)
            logger.info(
                "resolve_trade_date: candidate=%s type=%s",
                candidate, schedule_type,
            )
            if schedule_type in self._TRADING_DAY_TYPES:
                return candidate
            candidate -= timedelta(days=1)

        raise ThetaFatalError(
            f"_resolve_trade_date: no trading day found within "
            f"{self._MAX_CALENDAR_SCAN_DAYS} days before {as_of}. "
            f"Calendar data may be inconsistent.",
            status_code=200,
        )

    def _next_business_day(self, target: date) -> date:
        """target の「翌営業日」を解決する。

        DESIGN 3.4 の日付非対称性に対応するための部品。
        open_interest は「date に渡した日の前営業日 EOD」を返す規約のため、
        取引日 T の OI が欲しい場合は date に「T の翌営業日」を渡す必要がある。

        _resolve_trade_date の鏡像（過去遡及ではなく未来探索）。

        Args:
            target: 起点の日付（通常は _resolve_trade_date が返した T）。

        Returns:
            target より後の直近の取引日（open または early_close）。

        Raises:
            ThetaFatalError:
                _MAX_CALENDAR_SCAN_DAYS 日進んでも取引日が
                見つからない場合（カレンダー異常）。
            ThetaPermissionError / ThetaRetryExhaustedError:
                _fetch_calendar_on_date から伝播。

        Note:
            未来方向の探索のため、on_date の対応範囲（2012-01-01 〜
            翌年末）の上端境界に近づく可能性がある。実運用では target は
            T（直近過去営業日）であり翌営業日は today 近傍なので
            通常は境界を踏まないが、範囲外を踏んだ際の on_date の挙動は
            実装後検証項目（DESIGN セクション6）として残る。
        """
        candidate = target + timedelta(days=1)
        for _ in range(self._MAX_CALENDAR_SCAN_DAYS):
            # 監査7: on_date レスポンスに date 列が無いため candidate を併記。
            schedule_type = self._fetch_calendar_on_date(candidate)
            logger.info(
                "next_business_day: candidate=%s type=%s",
                candidate, schedule_type,
            )
            if schedule_type in self._TRADING_DAY_TYPES:
                return candidate
            candidate += timedelta(days=1)

        raise ThetaFatalError(
            f"_next_business_day: no trading day found within "
            f"{self._MAX_CALENDAR_SCAN_DAYS} days after {target}. "
            f"Calendar data may be inconsistent.",
            status_code=200,
        )

    # ── 内部: 個別エンドポイント ──

    def _fetch_open_interest(
        self, symbol: str, oi_date: date
    ) -> pd.DataFrame:
        """history/open_interest を取得し、必要列の DataFrame を返す。

        DESIGN 5.1 で snapshot から history に改修。

        endpoint: /option/history/open_interest

        日付の非対称性（DESIGN 3.4 / 3.5）:
            OI は「date に渡した日の前営業日 EOD」を返す規約（OPRA が
            毎朝 06:30 ET に前営業日値を報告する構造に由来）。
            このため、取引日 T の OI が欲しい場合、呼び出し側は
            oi_date に「T の翌営業日」を渡す。この関数自身は日付を
            解釈せず、渡された oi_date をそのまま date パラメータにする。
            非対称性の管理は get_option_chain に集約する。

        Args:
            symbol: 取得対象シンボル。
            oi_date: date パラメータにそのまま渡す日付
                     （= 取得したい取引日 T の翌営業日）。

        Returns:
            列: symbol, expiration, strike, right, open_interest
            空レスポンス時は空 DataFrame（列なし）。

        Raises:
            ThetaFatalError: 必須列が欠落している場合。
        """
        csv_text = self._request_csv(
            "/option/history/open_interest",
            {
                "symbol": symbol,
                "expiration": "*",
                "date": oi_date.strftime("%Y%m%d"),
            },
        )
        if not csv_text.strip():
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(csv_text))

        # 公式仕様: symbol, expiration, strike, right, timestamp, open_interest
        # 必要列のみ抽出（snapshot 時代と列構造は一致、DESIGN 5.4）。
        required = ["symbol", "expiration", "strike", "right", "open_interest"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ThetaFatalError(
                f"OI response missing expected columns: {sorted(missing)}. "
                f"Got: {list(df.columns)}",
                status_code=200,
                body=csv_text[:500],
            )

        out = df[required].copy()
        # 統一スキーマでは right ∈ {"call", "put"} の小文字。
        # 実 ThetaData は CSV で "CALL" / "PUT" の大文字を返すため正規化。
        # （v15 段階 6C 初回実行で発覚、誤判断18 として記録）
        out["right"] = out["right"].astype(str).str.lower()
        return out

    # IV 算出失敗を示す iv_error の番兵値（sentinel）。
    # ThetaData の history/greeks/eod は IV 計算が発散・非収束だった行に
    # iv_error = 100.0 を立て、implied_vol にはシード値（0.1250 等）を残す。
    # 出典: 2026-05-12 SPY ダンプ（4_option_history_greeks_eod, 528行）の
    #       iv_error 分布解析 ── |iv_error| は [0, 0.1) に 523 行が集中し、
    #       [0.1, 100.0) が完全に空、ちょうど 100.0 に 5 行。連続誤差では
    #       説明できない 2 桁分の空白があり、100.0 は固定の番兵値と判断。
    # 注意: これは「閾値」ではなく「番兵値との完全一致判定」。
    #       iv_error == 0 を欠損マーカーと読むのと同じカテゴリ。
    #       RESEARCH_005 で公式仕様の明記は得られず（数理的推論）、
    #       backfill 時に別の番兵値が出れば、この定数の定義1箇所を直す。
    _IV_ERROR_SENTINEL: float = 100.0

    def _fetch_greeks_eod(
        self, symbol: str, trade_date: date
    ) -> pd.DataFrame:
        """greeks/eod を取得し、IV 健全性フィルタを通した DataFrame を返す。

        DESIGN 5.1 で _fetch_implied_volatility（snapshot 版）を置換。

        endpoint: /option/history/greeks/eod
            ★ /option/history/greeks/implied_volatility ではない。
              後者は interval 既定 1s で日中ティック列を返し、1 満期で
              約 1.6 GB / 1,235 万行になる（DESIGN 2.2 / 誤判断21）。
              エンドポイント名が 1 単語違いで結果が 1 万倍違うため厳重注意。

        Args:
            symbol: 取得対象シンボル。
            trade_date: 取引日 T。start_date = end_date = T で当日 EOD を取る。

        Returns:
            列: symbol, expiration, strike, right, bid, ask,
                implied_volatility, underlying_price（_merge_oi_iv の契約）
            IV 健全性フィルタ済み。dtype は文字列のまま（coerce は呼び出し側）。
            空レスポンス時は空 DataFrame（列なし）。

        Raises:
            ThetaFatalError: 必須列が欠落している場合。
        """
        date_str = trade_date.strftime("%Y%m%d")
        csv_text = self._request_csv(
            "/option/history/greeks/eod",
            {
                "symbol": symbol,
                "expiration": "*",
                "start_date": date_str,
                "end_date": date_str,
            },
        )
        if not csv_text.strip():
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(csv_text))

        # greeks/eod の IV 列名は "implied_vol"。統一スキーマは
        # "implied_volatility" なのでリネーム。
        if "implied_vol" in df.columns and "implied_volatility" not in df.columns:
            df = df.rename(columns={"implied_vol": "implied_volatility"})

        # IV 健全性フィルタには iv_error が必要。必須列に含める。
        required = [
            "symbol", "expiration", "strike", "right",
            "bid", "ask", "implied_volatility", "iv_error",
            "underlying_price",
        ]
        missing = set(required) - set(df.columns)
        if missing:
            raise ThetaFatalError(
                f"greeks/eod response missing expected columns: "
                f"{sorted(missing)}. Got: {list(df.columns)}",
                status_code=200,
                body=csv_text[:500],
            )

        out = df[required].copy()
        # 統一スキーマでは right ∈ {"call", "put"} の小文字。
        # 実 ThetaData は CSV で "CALL" / "PUT" の大文字を返すため正規化。
        out["right"] = out["right"].astype(str).str.lower()

        out = self._apply_iv_health_filter(out, symbol, date_str)

        # iv_error は健全性フィルタ専用の補助列。_merge_oi_iv の契約
        # （8 列）には含めないため、フィルタ適用後に落とす。
        return out.drop(columns=["iv_error"]).reset_index(drop=True)

    def _apply_iv_health_filter(
        self, df: pd.DataFrame, symbol: str, date_str: str
    ) -> pd.DataFrame:
        """IV 算出に失敗した行を除外し、除外内訳を INFO で集計出力する。

        除外条件（監査ポイント4 / RESEARCH_005 で確定）:
            (1) implied_volatility <= 0
                深 ITM で時間価値が消失し IV が数学的に算出不能。
                σ = 0 では gamma = N'(d1)/(S·σ·√T) が定義できない。
            (2) iv_error == _IV_ERROR_SENTINEL (100.0)
                求根アルゴリズムが発散・非収束。implied_volatility には
                シード値が残るため (1) では捕捉できない。

        いずれも「Black-Scholes が定義できない / IV が信用できない」行。
        マジックナンバーによる足切りではなく、数学的事実と番兵値判定。

        ログ方針（段階 6C の OI=0 大量警告の教訓）:
            1 行ごとの WARNING は出さない。除外総数と理由内訳を
            INFO で 1 行に集計する。

        Note:
            除外行に大きな OI が乗っていた場合の「サイレント Wall 欠落」
            WARNING は、この関数では出せない（OI は未結合）。
            OI 結合後の層で別途実装する（Step 3b）。

        Args:
            df: implied_volatility, iv_error を含む DataFrame。
            symbol, date_str: ログ用。

        Returns:
            健全な行のみの DataFrame。
        """
        iv = pd.to_numeric(df["implied_volatility"], errors="coerce")
        err = pd.to_numeric(df["iv_error"], errors="coerce")

        bad_iv = iv <= 0
        bad_err = err == self._IV_ERROR_SENTINEL
        # to_numeric で NaN になった行（数値化不能）も健全でないため除外。
        unparseable = iv.isna() | err.isna()
        exclude = bad_iv | bad_err | unparseable

        n_total = len(df)
        n_excluded = int(exclude.sum())
        if n_excluded:
            logger.info(
                "IV health filter [%s %s]: kept %d / %d rows. "
                "excluded %d (implied_vol<=0: %d, iv_error==%.1f: %d, "
                "unparseable: %d).",
                symbol, date_str,
                n_total - n_excluded, n_total, n_excluded,
                int(bad_iv.sum()), self._IV_ERROR_SENTINEL,
                int(bad_err.sum()), int(unparseable.sum()),
            )

        return df[~exclude]

    # ── 内部: 結合 ──

    @staticmethod
    def _merge_oi_iv(
        oi_df: pd.DataFrame, iv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """OI と IV を 4 列キーで outer join し、片側欠落を診断ログに記録。

        論点3a/3b の決定:
            - outer join（片側欠落も診断対象に残す）
            - 結合後に「OI と IV の両方が揃わないレコード」は除外
              （GEX 計算に必須なので）
            - 除外件数を警告ログに出す

        サイレント Wall 欠落の事実ログ（Step 3b）:
            left_only（OI あり・IV なし）の行は2種類の合成:
              (1) _fetch_greeks_eod の IV 健全性フィルタで落ちたストライク
              (2) そもそも greeks/eod に存在しなかったストライク
            どちらも「OI はあるが GEX 計算に使えない」点で同じ。
            これらに大きな OI が乗っていると GEX プロファイル（特に
            Call/Put Wall）が歪むため、left_only の OI 合計と全体比を
            事実として INFO ログに出す。
            ※ 閾値による WARNING 格上げは現時点では実装しない。
              「比率がどれだけで Wall が歪むか」は段階6E の実データ観察で
              根拠が出てから設計する（マジックナンバー回避）。

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

        # ── Step 3b: サイレント Wall 欠落の事実ログ ──
        # left_only（OI あり・IV なし）の OI 規模を全体比で報告する。
        # 閾値判定はせず、事実のみ。これだけで「サイレント」ではなくなる。
        if n_oi_only:
            oi_only_mask = merged["_merge"] == "left_only"
            lost_oi = pd.to_numeric(
                merged.loc[oi_only_mask, "open_interest"], errors="coerce"
            ).fillna(0)
            total_oi = pd.to_numeric(
                merged["open_interest"], errors="coerce"
            ).fillna(0)
            lost_sum = float(lost_oi.sum())
            total_sum = float(total_oi.sum())
            pct = (lost_sum / total_sum * 100.0) if total_sum > 0 else 0.0
            logger.info(
                "OI/IV merge: %d strike(s) have OI but no usable IV. "
                "Lost open_interest = %.0f (%.2f%% of total OI %.0f). "
                "If concentrated near a wall, the GEX profile may be distorted.",
                int(n_oi_only), lost_sum, pct, total_sum,
            )

        clean = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
        return clean.reset_index(drop=True)
