"""段階5 cron エントリポイント: 日次 GEX 計算と JSON 保存。

GitHub Actions の workflow_dispatch / cron から呼ばれる main スクリプト。

責務:
    1. GEX_DATA_SOURCE 環境変数を見て Adapter を選択（mock / rest）
    2. SPY のオプションチェーンを取得
    3. calculate_all() で GEX 計算
    4. save_gex_result() で gex_history.json に追記（atomic write）
    5. 例外発生時は sys.exit(1) で Job を確実に落とす

設計判断（v14 段階5 で確定）:
    - 銘柄は SPY のみハードコード（複数対応は段階6 以降）
    - 出力パスは gex_history.json（リポジトリ直下、EA 互換維持）
    - rest 選択時のパラメータはデフォルト値（max_retries=3, retry_backoff_base=1.0）

実行例:
    GEX_DATA_SOURCE=mock python -m gex_engine.scripts.run_daily
    GEX_DATA_SOURCE=rest python -m gex_engine.scripts.run_daily
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date

from ..adapters.base import DataFetcher
from ..adapters.mock import MockDataFetcher
from ..adapters.rest import ThetaRestAdapter
from ..core.gex import calculate_all
from ..io_layer import save_gex_result


# ──────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────

SYMBOL = "SPY"
OUTPUT_PATH = "gex_history.json"

# Mock の固定パラメータ（段階3.5 のスモークテストと整合）
MOCK_SPOT_PRICE = 450.0
MOCK_SEED = 42

# REST のデフォルトパラメータ（v14 セクション8-6 の設計値）
REST_MAX_RETRIES = 3
REST_RETRY_BACKOFF = 1.0


# ──────────────────────────────────────────────────────────
# ロガー設定（GitHub Actions のログで読みやすく）
# ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("run_daily")


# ──────────────────────────────────────────────────────────
# Adapter ファクトリ
# ──────────────────────────────────────────────────────────

def make_fetcher(source: str) -> DataFetcher:
    """環境変数の値から Adapter を生成。

    Args:
        source: "mock" または "rest"（小文字想定、呼び出し側で正規化済み）

    Returns:
        DataFetcher Protocol を満たす Adapter インスタンス

    Raises:
        ValueError: 未知の source が渡された
    """
    if source == "mock":
        logger.info(
            "Using MockDataFetcher (spot=%.2f, seed=%d)",
            MOCK_SPOT_PRICE, MOCK_SEED,
        )
        return MockDataFetcher(spot_price=MOCK_SPOT_PRICE, seed=MOCK_SEED)

    if source == "rest":
        logger.info(
            "Using ThetaRestAdapter (max_retries=%d, backoff_base=%.1f)",
            REST_MAX_RETRIES, REST_RETRY_BACKOFF,
        )
        return ThetaRestAdapter(
            max_retries=REST_MAX_RETRIES,
            retry_backoff_base=REST_RETRY_BACKOFF,
        )

    raise ValueError(
        f"Unknown GEX_DATA_SOURCE: {source!r}. Expected 'mock' or 'rest'."
    )


# ──────────────────────────────────────────────────────────
# メインフロー
# ──────────────────────────────────────────────────────────

def run() -> None:
    """1 日分の GEX 計算 → JSON 追記を実行する。

    Raises:
        例外は呼び出し側（main()）でキャッチされ、sys.exit(1) になる。
    """
    source = os.environ.get("GEX_DATA_SOURCE", "mock").strip().lower()
    today = date.today()

    logger.info("=" * 60)
    logger.info("GEX daily run start")
    logger.info("  source: %s", source)
    logger.info("  symbol: %s", SYMBOL)
    logger.info("  date:   %s", today.isoformat())
    logger.info("  output: %s", OUTPUT_PATH)
    logger.info("=" * 60)

    fetcher = make_fetcher(source)

    try:
        # ── データ取得 ──
        logger.info("Fetching option chain for %s...", SYMBOL)
        df = fetcher.get_option_chain(SYMBOL, today)
        logger.info("Fetched %d rows", len(df))

        if df.empty:
            # NO_DATA（休場日等）。ジョブとしては成功扱いで終わる。
            # 当日分のエントリは作成しない（既存履歴は壊さない）。
            logger.warning(
                "Empty option chain (likely market holiday or NO_DATA). "
                "Skipping write. gex_history.json is unchanged."
            )
            return

        # ── GEX 計算 ──
        logger.info("Calculating GEX...")
        result = calculate_all(df, as_of=today, data_source=fetcher.source_name)
        logger.info(
            "GEX result: spot=%.2f CW=%s PW=%s ZG=%.2f MP=%s total=%.2f",
            result.underlying_price,
            result.call_wall, result.put_wall,
            result.zero_gamma, result.max_pain,
            result.total_gex,
        )

        # ── JSON 追記 ──
        logger.info("Saving to %s...", OUTPUT_PATH)
        entry = save_gex_result(result, path=OUTPUT_PATH)
        logger.info("Saved entry: %s", entry)

    finally:
        # REST Adapter の場合は httpx.Client を閉じる
        # Mock は close() を持たないが、Protocol 違反ではない（ダックタイピング）
        close = getattr(fetcher, "close", None)
        if callable(close):
            close()

    logger.info("GEX daily run completed successfully")


def main() -> int:
    """exit code を返すエントリポイント。

    Returns:
        0: 正常終了（NO_DATA で書き込みスキップした場合も含む）
        1: 例外発生（GitHub Actions の Job を落とす）
    """
    try:
        run()
        return 0
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
