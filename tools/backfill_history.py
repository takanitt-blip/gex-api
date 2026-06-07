"""tools/backfill_history.py ─ 過去 EOD GEX データの一括生成（Phase 3）。

検証フェーズ用に、ThetaData Standard から過去 N 営業日分の EOD を取得し、
本番と完全に同じ計算経路で GEX を計算して gex_history.json に追記する。

設計の核心（PC_VALIDATION §3.4/§3.5b, PC_GOVERNANCE 誤判断23課α/25/26/29）:

  1. 本番経路を再実装しない。get_option_chain → calculate_all →
     next_business_day → save_gex_result を run_daily.run() と同じ並びで呼ぶ。
     違いは as_of の決め方（cron は today、backfill は D+1暦日）と、冪等性/
     上書き/孤児掃除のラッパだけ。独立計算式を持たない（誤判断26 回避）。

  2. off-by-one の封じ込め（obs.F と同型の罠）:
     rest.py の _resolve_trade_date(as_of) は「as_of より厳密に前」の直近営業日を
     返す。取引日 D の EOD が欲しければ as_of = D + 1暦日 を渡す
     （_resolve は candidate=D を最初に当て、D は取引日なので即 D を返す）。
     取得後に df["trade_date"] == D を assert して、ズレを即検出する。

  3. 冪等性 × 上書きの統一:
     スキップ = 既存キーが「v17形式（data_quality を持つ）かつ data_source ∈
                {rest, rest_backfill}」= 現行の正しいパイプライン産。再取得しない。
     上書き   = それ以外（regime形式 = snapshot汚染/obs.F期、または不在）。
     → §3.4 の「5/12〜5/15 snapshot汚染を上書き」は regime形式なので自動成立。
     backfill が書くエントリの data_source は "rest_backfill"（provenance タグ）。

  4. キー意味論（obs.G 案2 = session-served）:
     JSON キー = next_business_day(trade_date)。営業日リストの successor で O(1) 算出
     （on_date 由来の同一カレンダーなので next_business_day と定義的に一致）。

  5. 孤児掃除（--clean-orphans, dry-run 既定）:
     範囲内の「非取引日キー」（例 2026.05.16 土）は now() 由来の遺物 → 削除。
     取引日キーは絶対に消さない（正当 cron を誤削除しない保守ルール）。
     gex_history.json からの恒久削除なので実行前に git commit を推奨。

実行例（リポジトリ直下から。env var は不要、常に rest Adapter を使う）:
    # 直近2年を取得（既定）
    python tools/backfill_history.py
    # 3年に伸ばす
    python tools/backfill_history.py --start 2023-06-01
    # まず計画だけ確認（fetch も write もしない）
    python tools/backfill_history.py --dry-run
    # 取得後に孤児掃除（まず dry-run で確認 → 外して実行）
    python tools/backfill_history.py --clean-orphans --dry-run
    python tools/backfill_history.py --clean-orphans

前提:
    Theta Terminal がローカル起動済み（rest.py と同じ）。これは時間単位ジョブ
    （2年で約2時間、Phase 2 実測）。1日ごとに atomic 保存するため、中断しても
    再実行で済んだ日はスキップして再開できる。
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import date, datetime, timedelta

# tools/ は gex_engine/ の兄弟。リポジトリ直下を import path に通す
# （python tools/backfill_history.py をどこから叩いても動くように）。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gex_engine.adapters.rest import (  # noqa: E402
    ThetaPermissionError,
    ThetaDataError,
    ThetaRestAdapter,
)
from gex_engine.core.gex import calculate_all  # noqa: E402
from gex_engine.io_layer import (  # noqa: E402
    load_history,
    save_gex_result,
    write_json_atomic,
)
from gex_engine.io_layer.serializer import make_date_key  # noqa: E402
from gex_engine.market_calendar import (  # noqa: E402
    TRADING_DAY_TYPES,
    next_business_day,
)


# ──────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────

SYMBOL_DEFAULT = "SPY"
OUTPUT_PATH = "gex_history.json"

# IV 必須（gamma 自前計算）のため backfill の床 = greeks/eod 境界。
# Phase 2 実 API probe で確定（PC_GOVERNANCE 誤判断29 / PC_VALIDATION §3.5b）。
HISTORY_FLOOR = date(2017, 1, 3)

# 既定の遡及年数（検証ニーズ 1〜2 年。3 年は --start で）。
DEFAULT_LOOKBACK_DAYS = 365 * 2

# 営業日リスト構築時、範囲末より先まで走査して successor（= session_date）を
# 確保するためのバッファ。米国市場の最長連続休場でも数日。
LIST_BUFFER_DAYS = 10

# 過去向きに「直近の完了済み取引日」を探す上限。
RESOLVE_BACK_SCAN_DAYS = 10

BACKFILL_DATA_SOURCE = "rest_backfill"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("backfill_history")


# ──────────────────────────────────────────────────────────
# 日付ユーティリティ
# ──────────────────────────────────────────────────────────

def _parse_date(s: str) -> date:
    """YYYY-MM-DD をパース。"""
    return datetime.strptime(s, "%Y-%m-%d").date()


def _key_to_date(key: str) -> date | None:
    """JSON キー "YYYY.MM.DD" を date に。形式不正なら None。"""
    try:
        return datetime.strptime(key, "%Y.%m.%d").date()
    except (ValueError, TypeError):
        return None


def resolve_default_end(fetcher: ThetaRestAdapter, today: date) -> date:
    """「直近の完了済み取引日」を返す（today より厳密に前の直近営業日）。

    rest.py の _resolve_trade_date(today) と同じ意味だが、private に依存せず
    公開 schedule_type_on で再現する（誤判断13: 隣接層の private を覗かない）。
    """
    candidate = today - timedelta(days=1)
    for _ in range(RESOLVE_BACK_SCAN_DAYS):
        if fetcher.schedule_type_on(candidate) in TRADING_DAY_TYPES:
            return candidate
        candidate -= timedelta(days=1)
    raise RuntimeError(
        f"resolve_default_end: {RESOLVE_BACK_SCAN_DAYS} 日遡っても取引日が "
        f"見つからない（{today} 起点）。カレンダー異常の疑い。"
    )


def build_trading_days(
    fetcher: ThetaRestAdapter, start: date, end_inclusive: date
) -> list[date]:
    """[start, end_inclusive + buffer] の取引日を on_date 走査で列挙する。

    検証済みの schedule_type_on のみ使う（year_holidays パーサは production に
    無いため使わない。誤判断16/17/18/24）。buffer は範囲末の successor 確保用。

    Returns:
        昇順の取引日リスト（open / early_close のみ）。
    """
    scan_end = end_inclusive + timedelta(days=LIST_BUFFER_DAYS)
    days: list[date] = []
    d = start
    n_calls = 0
    logger.info(
        "営業日リスト構築中: %s 〜 %s（buffer 込み 〜%s）を on_date で走査...",
        start, end_inclusive, scan_end,
    )
    while d <= scan_end:
        if fetcher.schedule_type_on(d) in TRADING_DAY_TYPES:
            days.append(d)
        n_calls += 1
        d += timedelta(days=1)
    logger.info(
        "営業日リスト構築完了: on_date %d 回 → 取引日 %d 件", n_calls, len(days)
    )
    return days


# ──────────────────────────────────────────────────────────
# スキップ判定
# ──────────────────────────────────────────────────────────

def is_current_pipeline_entry(entry: object) -> bool:
    """既存エントリが「現行の正しいパイプライン産」か判定する。

    True = v17形式（data_quality キーを持つ）かつ data_source ∈ {rest, rest_backfill}。
    これらは a7-A（obs.F 根治）以降に生成された正しい値なので再取得不要。
    False = regime形式（snapshot汚染 / obs.F期）や mock / 不明 → 上書き対象。
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("data_quality") is None:  # v17 で導入。pre-v17 は regime のみ
        return False
    return entry.get("data_source") in ("rest", BACKFILL_DATA_SOURCE)


# ──────────────────────────────────────────────────────────
# バックフィル本体
# ──────────────────────────────────────────────────────────

def run_backfill(
    fetcher: ThetaRestAdapter,
    *,
    symbol: str,
    start: date,
    end: date,
    force: bool,
    dry_run: bool,
    sleep: float,
) -> tuple[list[date], list[date], list[date]]:
    """start〜end の各取引日を逐次処理して gex_history.json に追記する。

    Returns:
        (saved, skipped, failed) の取引日リスト。
    """
    trading_days = build_trading_days(fetcher, start, end)
    index = {d: i for i, d in enumerate(trading_days)}
    process_days = [d for d in trading_days if start <= d <= end]

    if not process_days:
        logger.warning("対象取引日が 0 件（範囲: %s 〜 %s）。終了。", start, end)
        return [], [], []

    history = load_history(OUTPUT_PATH)
    total = len(process_days)
    logger.info(
        "バックフィル開始: symbol=%s 取引日 %d 件 (%s 〜 %s) force=%s dry_run=%s",
        symbol, total, process_days[0], process_days[-1], force, dry_run,
    )

    saved: list[date] = []
    skipped: list[date] = []
    failed: list[date] = []
    t0 = time.monotonic()

    for i, D in enumerate(process_days, start=1):
        # session_date = 営業日リストの successor（= next_business_day(D)）。
        # 末尾要素は buffer 内に successor があるはず。万一無ければ API で補う。
        idx = index[D]
        if idx + 1 < len(trading_days):
            session_date = trading_days[idx + 1]
        else:
            session_date = next_business_day(D, fetcher.schedule_type_on)
        key = make_date_key(session_date)

        # ── 進捗 ──
        elapsed = time.monotonic() - t0
        eta = (elapsed / i) * (total - i) if i else 0.0
        prefix = f"[{i}/{total}] D={D} key={key}"

        # ── スキップ判定（fetch 前。済みなら ~10秒の取得を省く）──
        existing = history.get(key)
        if existing is not None and is_current_pipeline_entry(existing) and not force:
            logger.info("%s SKIP（既に v17 形式・現行パイプライン産）", prefix)
            skipped.append(D)
            continue

        if dry_run:
            action = "OVERWRITE" if existing is not None else "FETCH"
            logger.info("%s DRY-RUN → %s（elapsed %.0fs, eta %.0fs）",
                        prefix, action, elapsed, eta)
            saved.append(D)  # dry-run では「やる予定」を saved として数える
            continue

        # ── 取得 → 計算 → 保存（本番と同一経路）──
        try:
            # off-by-one 封じ込め: 取引日 D の EOD = as_of に D+1暦日 を渡す。
            df = fetcher.get_option_chain(symbol, D + timedelta(days=1))

            if df.empty:
                # 472 / 休場相当。エントリを作らず次へ（既存履歴は壊さない）。
                logger.warning("%s NO_DATA（空チェーン）→ スキップ", prefix)
                failed.append(D)
                continue

            # Adapter が解釈した取引日 T を取り出し、D と一致を assert（ズレ検出器）。
            assert "trade_date" in df.columns, (
                f"{prefix}: Adapter が trade_date 列を出していない（誤判断25）"
            )
            assert df["trade_date"].nunique() == 1, (
                f"{prefix}: trade_date が複数混在（誤判断25）"
            )
            resolved = df["trade_date"].iloc[0].date()
            if resolved != D:
                # off-by-one が起きた = カレンダー不整合。続行すると誤キーで
                # 汚染するので即停止（意味のズレは fail loud）。
                raise RuntimeError(
                    f"{prefix}: as_of=D+1 のはずが Adapter 解決 trade_date="
                    f"{resolved} != D={D}。カレンダー不整合。中断。"
                )

            result = calculate_all(df, as_of=D, data_source=fetcher.source_name)

            # data_source を "rest_backfill" に上書きして保存（provenance タグ）。
            # zero_gamma=None は calculate_all 側で data_quality="data_error" に
            # なり、そのまま保存される（検証で除外、Phase1 仕様）。
            save_gex_result(
                result,
                path=OUTPUT_PATH,
                session_date=session_date,
                data_source=BACKFILL_DATA_SOURCE,
            )
            # 保存後の状態を手元 history にも反映（同一 run 内の重複判定用）
            history = load_history(OUTPUT_PATH)
            logger.info(
                "%s SAVED dq=%s CW=%s ZG=%s PW=%s（elapsed %.0fs, eta %.0fs）",
                prefix, result.data_quality, result.call_wall,
                result.zero_gamma, result.put_wall, elapsed, eta,
            )
            saved.append(D)

        except ThetaPermissionError:
            # 471: プラン権限なし。全日同じ結果なので続行無意味 → 即 abort。
            logger.exception("%s PERMISSION エラー → 全体中断", prefix)
            raise
        except (ThetaDataError, AssertionError) as e:
            # その他の取得失敗 / 契約違反は当日だけ記録して継続。
            # 再実行すれば成功済みはスキップされる（冪等性）。
            logger.error("%s FAILED（継続）: %s", prefix, e)
            failed.append(D)

        if sleep > 0:
            time.sleep(sleep)

    logger.info(
        "バックフィル完了: saved=%d skipped=%d failed=%d (経過 %.0fs)",
        len(saved), len(skipped), len(failed), time.monotonic() - t0,
    )
    if failed:
        logger.warning(
            "失敗/欠損のあった取引日 %d 件: %s", len(failed),
            ", ".join(d.isoformat() for d in failed),
        )
    return saved, skipped, failed


# ──────────────────────────────────────────────────────────
# 孤児掃除
# ──────────────────────────────────────────────────────────

def clean_orphans(
    fetcher: ThetaRestAdapter,
    *,
    start: date,
    end: date,
    dry_run: bool,
) -> list[str]:
    """範囲 [start, end] 内の「非取引日キー」を孤児として削除する。

    session-served キーは定義上必ず取引日になるので、範囲内で取引日でない
    キー（例 2026.05.16 土）= now() 由来の遺物。取引日キーは絶対に消さない
    （正当 cron を誤削除しない保守ルール）。範囲外のキーには一切触れない。

    Returns:
        削除した（dry-run なら削除予定の）キーのリスト。
    """
    logger.info("孤児掃除: 範囲 %s 〜 %s（dry_run=%s）", start, end, dry_run)
    trading_days = set(build_trading_days(fetcher, start, end))

    history = load_history(OUTPUT_PATH)
    orphans: list[str] = []
    for key in list(history.keys()):
        d = _key_to_date(key)
        if d is None:
            logger.warning("孤児掃除: キー %r は日付形式でない → 安全側で温存", key)
            continue
        if not (start <= d <= end):
            continue  # 範囲外は対象外
        if d not in trading_days:  # 範囲内・非取引日 = 孤児
            orphans.append(key)

    if not orphans:
        logger.info("孤児掃除: 範囲内に孤児キーなし。")
        return []

    for key in orphans:
        logger.info("孤児掃除: %s（%s）を削除%s",
                    key, _key_to_date(key),
                    "（DRY-RUN）" if dry_run else "")

    if dry_run:
        logger.info("孤児掃除: DRY-RUN のため未削除（%d 件）。"
                    "実削除は --clean-orphans を --dry-run なしで。", len(orphans))
        return orphans

    for key in orphans:
        del history[key]
    write_json_atomic(OUTPUT_PATH, history)
    logger.info("孤児掃除: %d 件削除し %s を更新。", len(orphans), OUTPUT_PATH)
    return orphans


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="過去 EOD GEX データを gex_history.json に一括生成する。"
    )
    p.add_argument("--symbol", default=SYMBOL_DEFAULT, help="対象シンボル（既定 SPY）")
    p.add_argument("--start", type=_parse_date, default=None,
                   help="開始日 YYYY-MM-DD（既定: end の約2年前、床 2017-01-03）")
    p.add_argument("--end", type=_parse_date, default=None,
                   help="終了日 YYYY-MM-DD（既定: 直近の完了済み取引日）")
    p.add_argument("--force", action="store_true",
                   help="v17形式の既存エントリも強制的に再計算・上書きする")
    p.add_argument("--clean-orphans", action="store_true",
                   help="範囲内の非取引日キー（now() 由来の孤児）を掃除する")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch も write もせず、実行計画／削除予定のみ表示する")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="各取引日の処理間に挟む待機秒（既定 0、Phase2 で throttle 不要）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    fetcher = ThetaRestAdapter(max_retries=3, retry_backoff_base=1.0)
    try:
        today = date.today()
        end = args.end or resolve_default_end(fetcher, today)
        start = args.start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))

        # 床でクランプ（greeks/eod の遡及境界、誤判断29）。
        if start < HISTORY_FLOOR:
            logger.warning("start=%s は床 %s より古いのでクランプ。",
                           start, HISTORY_FLOOR)
            start = HISTORY_FLOOR
        if end < start:
            logger.error("end=%s が start=%s より前。中断。", end, start)
            return 1

        logger.info("=" * 60)
        logger.info("backfill_history: %s  %s 〜 %s", args.symbol, start, end)
        logger.info("  force=%s clean_orphans=%s dry_run=%s",
                    args.force, args.clean_orphans, args.dry_run)
        if not args.dry_run and (args.force or args.clean_orphans):
            logger.info("  ※ 既存データを書き換えます。実行前の git commit を推奨。")
        logger.info("=" * 60)

        run_backfill(
            fetcher,
            symbol=args.symbol,
            start=start,
            end=end,
            force=args.force,
            dry_run=args.dry_run,
            sleep=args.sleep,
        )

        if args.clean_orphans:
            clean_orphans(fetcher, start=start, end=end, dry_run=args.dry_run)

        return 0

    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error: %s", e)
        return 1
    finally:
        close = getattr(fetcher, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
