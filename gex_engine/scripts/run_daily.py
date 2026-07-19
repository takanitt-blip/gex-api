"""段階5 cron エントリポイント: 日次 GEX 計算と JSON 保存。

GitHub Actions の workflow_dispatch / cron から呼ばれる main スクリプト。

責務:
    1. GEX_DATA_SOURCE 環境変数を見て Adapter を選択（mock / rest）
    2. SPY のオプションチェーンを取得（誤判断36 対応: 空ならデッドラインまで
       リトライ。詳細は _wait_for_option_chain を参照）
    3. calculate_all() で GEX 計算
    4. save_gex_result() で gex_history.json に追記（atomic write）
    5. 例外発生時は sys.exit(1) で Job を確実に落とす

設計判断（v14 段階5 で確定）:
    - 銘柄は SPY のみハードコード（複数対応は段階6 以降）
    - 出力パスは gex_history.json（リポジトリ直下、EA 互換維持）
    - rest 選択時のパラメータはデフォルト値（max_retries=3, retry_backoff_base=1.0）

誤判断36 対応（2026-07 実装）:
    update_gex.yml の cron を「引けの翌未明（ET 01:30〜02:30 相当）」へ
    移行したことに伴い、大半の実行は初回の get_option_chain で成功する
    想定だが、PC_PIPELINE §5.10 で実測した稀なケース（金曜の生成遅延・
    OI の一時的な 472 リバウンド）に備え、ET 06:45 のデッドラインまで
    5 分間隔でリトライする。trade_date は calendar/on_date で検証済み
    （休場日ではあり得ない）ため、デッドラインまでに解消しなかった空応答は
    常に「要調査の異常」として扱い、意図的に Job を失敗させる
    （DeadlineExceededError → main() の except で exit 1）。
    これにより GitHub Actions 標準の失敗通知が発火する（Slack 等の別経路は
    v17 段階6 で YAGNI と判断済み、PC_PIPELINE §6.9）。デッドラインは
    寄り付き（09:30 ET）の 2 時間45分前に設定してあり、通知を見た人間が
    手動 workflow_dispatch でバックアップ実行する猶予を残す設計。

実行例:
    GEX_DATA_SOURCE=mock python -m gex_engine.scripts.run_daily
    GEX_DATA_SOURCE=rest python -m gex_engine.scripts.run_daily
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from ..adapters.base import DataFetcher
from ..adapters.mock import MockDataFetcher
from ..adapters.rest import ThetaRestAdapter
from ..core.gex import calculate_all
from ..market_calendar import next_business_day
from ..io_layer import save_gex_result
from ..io_layer.serializer import scale_total_gex


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

# ── 誤判断36 対応: OI/greeks 収束待ちリトライ（2026-07 実装）──
# PC_PIPELINE §5.10 実測（律速は greeks/eod、引けから 8h57m〜9h10m）に基づく。
# cron 発火時刻（update_gex.yml, UTC 06:30 = ET 01:30(EST)〜02:30(EDT)）は
# この生成境界の直後になるよう設計済みのため、大半の実行は初回成功する想定。
# 以下は稀なケース向けの安全網。
_ET = ZoneInfo("America/New_York")
_POLL_INTERVAL_SECONDS = 5 * 60

# デッドライン = ET 06:45。§5.10 提言の「目安 ET 07:00」から寄り付き
# （09:30 ET）側に 15 分寄せたバッファ。理由:
#   (1) GitHub Actions のジョブ既定タイムアウト(360分)に収まる余地を残す
#       （冬季 cron=ET01:30 起点で最大リトライ幅は約5h15m=315分、
#        セットアップ/後片付けを含めても timeout-minutes の設定値に収まる）
#   (2) デッドライン到達＝Job失敗の通知を見た人間が、寄り付きまでに
#       手動 workflow_dispatch でバックアップ実行する猶予（約2時間45分）を残す
_DEADLINE_ET_HOUR = 6
_DEADLINE_ET_MINUTE = 45


class DeadlineExceededError(Exception):
    """デッドラインまでに OI/greeks が揃わなかった（誤判断36 対応）。

    trade_date はカレンダー検証済みで休場日ではあり得ないため、これは
    常に要調査の状態（生成の異常な遅延、または真の障害）。main() の
    except Exception で捕捉され exit 1 になり、GitHub Actions 標準の
    失敗通知（メール等）が発火する設計。
    """


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
# 診断ログ（段階 6C 検証用、rest のときのみ呼ばれる）
# ──────────────────────────────────────────────────────────

def _log_oi_distribution(df, log) -> None:
    """OI トップ 10 ストライクをログに出力。

    段階 6C の合格基準 E 第 2 項「Walls が大量 OI ストライクと整合」を
    Actions ログだけで検証可能にするための診断出力。

    Mock のときは呼ばれないので、ログを汚さない。

    例外は握り潰す（メイン処理は既に完了しているため、
    診断ログの失敗で Job を落とすべきでない）。

    TODO(stage-6E-end): 観察期間が終わって安定運用に入ったら、
        この診断ログの恒久化 / 削除 / 別ツール化を判断する。
    """
    try:
        # (strike, right) 単位で OI 合計、トップ 10 を取得
        top = (
            df.groupby(["strike", "right"])["open_interest"]
              .sum()
              .sort_values(ascending=False)
              .head(10)
        )

        log.info("── OI top 10 strikes (for stage 6C verification) ──")
        for (strike, right), oi in top.items():
            log.info("  strike=%-8.2f right=%-4s oi=%d", strike, right, int(oi))
        log.info("──────────────────────────────────────────────────")
    except Exception as e:
        # 診断ログでメイン処理を巻き込まない
        log.warning("Failed to log OI distribution (non-fatal): %s", e)


# ──────────────────────────────────────────────────────────
# データ取得の収束待ち（誤判断36 対応）
# ──────────────────────────────────────────────────────────

def _compute_deadline() -> datetime:
    """本日の ET デッドライン時刻を返す（tz-aware）。

    datetime.now() への依存をこの関数1つに切り出すことで、テストが
    monkeypatch でこの関数ごと差し替えられるようにしている
    （run() 本体が実時刻に結合されたままだと、デッドライン超過の
    挙動を検証するテストが実際に数時間待つか、fragile な時刻依存に
    なってしまうため）。
    """
    return datetime.now(_ET).replace(
        hour=_DEADLINE_ET_HOUR, minute=_DEADLINE_ET_MINUTE,
        second=0, microsecond=0,
    )


def _wait_for_option_chain(
    fetcher: DataFetcher,
    symbol: str,
    as_of: date,
    deadline: datetime,
    poll_interval_seconds: int = _POLL_INTERVAL_SECONDS,
    sleep_fn=time.sleep,
):
    """get_option_chain が非空を返すまで、deadline までリトライする。

    trade_date は Adapter 内部で calendar/on_date により検証済み
    （休場日ではあり得ない）。したがって空応答は「まだ生成されて
    いない」か「真の異常」のいずれかであり、両者を区別する追加情報は
    現状 Adapter から取れない（EMPTY_BOTH/EMPTY_ASYMMETRIC/
    MERGE_MISMATCH の詳細は Adapter の ERROR ログにのみ出る）。
    ログのパースで区別するのは脆いため、意図的に「どちらであっても
    deadline まで待つ」設計にしている（誤判断36 実装時の簡略化、
    次に踏み込むならログではなく Adapter の返り値を構造化する必要が
    あるが、現時点では実測に基づく根拠が無いため見送り）。

    Args:
        fetcher: DataFetcher。
        symbol: 銘柄。
        as_of: 処理基準日。呼び出しのたびにそのまま渡す
            （Adapter 側が同じ trade_date を解決する前提）。
        deadline: これを過ぎたら諦める。tz-aware datetime。
        poll_interval_seconds: リトライ間隔（秒）。テスト時は 0 等に短縮可能。
        sleep_fn: 待機に使う関数。テスト時は fake に差し替え可能
            （real time.sleep を呼ばずにテストを高速化するため）。

    Returns:
        非空の DataFrame。

    Raises:
        DeadlineExceededError: deadline までに非空データが得られなかった。
    """
    attempt = 0
    while True:
        attempt += 1
        logger.info(
            "Fetching option chain for %s (attempt %d)...", symbol, attempt
        )
        df = fetcher.get_option_chain(symbol, as_of)

        if not df.empty:
            logger.info("Fetched %d rows (attempt %d)", len(df), attempt)
            return df

        now = datetime.now(deadline.tzinfo)
        if now >= deadline:
            logger.error(
                "Deadline reached (%s) without data after %d attempt(s) "
                "for %s as_of=%s. See adapter ERROR logs above for the "
                "specific empty-response category (EMPTY_BOTH / "
                "EMPTY_ASYMMETRIC / MERGE_MISMATCH). A manual "
                "workflow_dispatch backup run is recommended once the "
                "cause is understood (PC_PIPELINE §5.10).",
                deadline.isoformat(), attempt, symbol, as_of,
            )
            raise DeadlineExceededError(
                f"No data for {symbol} as_of={as_of} after {attempt} "
                f"attempt(s), deadline {deadline.isoformat()} reached"
            )

        logger.warning(
            "Empty option chain on attempt %d (now=%s, deadline=%s). "
            "Retrying in %ds...",
            attempt, now.isoformat(), deadline.isoformat(),
            poll_interval_seconds,
        )
        sleep_fn(poll_interval_seconds)


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
        # ── データ取得（誤判断36 対応: 空ならデッドラインまでリトライ）──
        # 旧実装は1回叩いて空なら「たぶん休場日ではない異常」と ERROR
        # ログを出しつつ exit 0 で静かに終了していた（監査20時点の設計、
        # PC_PIPELINE §6.2 決定#9 の「NO_DATA→書き込みスキップ+exit 0」は
        # 元々「休場日かもしれない」という前提だったが、trade_date が
        # calendar 検証済みになった時点でこの前提は既に崩れていた）。
        # cron を生成境界直後に寄せた今、空応答の大半は「まだ生成されて
        # いないだけ」であり得るため、静かに諦めず deadline まで待つ。
        # deadline を過ぎてなお空なら DeadlineExceededError を送出し、
        # main() 経由で exit 1 → GitHub Actions 標準の失敗通知を発火する
        # （意図的な仕様変更、モジュール docstring 参照）。
        deadline = _compute_deadline()
        logger.info("Data fetch deadline: %s", deadline.isoformat())
        df = _wait_for_option_chain(fetcher, SYMBOL, today, deadline)

        # ── Adapter が解釈した取引日 T を df から抽出（obs.F 修正、誤判断25）──
        # 旧コード: as_of=today (cron 起動日の JST today) を Core に渡していた。
        # これだと土曜 cron は土曜を Core に渡し、Adapter が解決した金曜の
        # データと食い違って JSON に非取引日キーが書き込まれる事故 (obs.F) が
        # 発生した。
        #
        # 新コード: schema.REQUIRED_DTYPES の trade_date 列 (γ-1, γ-2 で
        # Adapter が必ず出すようになった) から T を抽出して Core に渡す。
        # これで Adapter の T と Core の as_of が常に一致する。
        #
        # assert は「将来 Adapter が trade_date を忘れた / 複数 T を混在
        # させた」事故を即座に検出するための契約。
        assert "trade_date" in df.columns, (
            "Adapter must emit trade_date column (誤判断25)"
        )
        assert df["trade_date"].nunique() == 1, (
            "trade_date must be unique per get_option_chain call (誤判断25)"
        )
        trade_date = df["trade_date"].iloc[0].date()
        logger.info(
            "Adapter resolved trade_date: %s (vs cron today=%s)",
            trade_date, today,
        )

        # このエントリが支配する取引セッション = next_business_day(trade_date)。
        # obs.G 根治: JSON キーを now()(cron 発火日) ではなく trade_date から
        # 決定論的に決める。fetcher.schedule_type_on を calendar lookup に注入。
        session_date = next_business_day(trade_date, fetcher.schedule_type_on)
        logger.info(
            "session_date (JSON key) = next_business_day(%s) = %s",
            trade_date, session_date,
        )

        # ── GEX 計算 ──
        logger.info("Calculating GEX (trade_date=%s from Adapter)...", trade_date)
        result = calculate_all(df, as_of=trade_date, data_source=fetcher.source_name)

        # JSON 出力時のスケール変換と同じ計算をログでも行う
        # （obs.A 是正: ログと JSON の同名フィールドの単位差を解消）
        # - total_gex_scaled: JSON の "total_gex" と完全一致する単位（× S^2 × 0.01）
        # - total_gex_raw:    Core Logic の素の出力（γ × OI × 100 の合計）
        # serializer.py の _to_int_or_none と同じく int(round(...)) で整数化し、
        # JSON 値との末尾ズレ（0.5 丸め）を消す。
        scaled_total_gex = int(round(
            scale_total_gex(result.total_gex, result.underlying_price)
        ))

        logger.info(
            "GEX result: spot=%.2f CW=%s PW=%s ZG=%.2f MP=%s "
            "total_gex_scaled=%d total_gex_raw=%.2f",
            result.underlying_price,
            result.call_wall, result.put_wall,
            result.zero_gamma, result.max_pain,
            scaled_total_gex,
            result.total_gex,
        )

        # ── JSON 追記 ──
        logger.info("Saving to %s...", OUTPUT_PATH)
        entry = save_gex_result(result, path=OUTPUT_PATH, session_date=session_date)
        logger.info("Saved entry: %s", entry)

        # ── 診断ログ（段階 6C 検証用、rest のときのみ）──
        # 合格基準 E「Walls が大量 OI ストライクと整合」を Actions ログで
        # 検証可能にするための追加情報。Mock では出力しない。
        if source == "rest":
            _log_oi_distribution(df, logger)

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
        0: 正常終了（データ取得〜JSON 保存まで成功した場合のみ。
           誤判断36 対応により、NO_DATA での「書き込みスキップして
           静かに exit 0」は廃止。deadline までに解消しなければ
           DeadlineExceededError で exit 1 になる）
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
