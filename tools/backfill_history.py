"""tools/backfill_history.py ─ 過去 EOD GEX データの一括生成（Phase 3）。

検証フェーズ用に、ThetaData Standard から過去 N 営業日分の EOD を取得し、
本番と完全に同じ計算経路で GEX を計算して gex_history.json に追記する。

設計の核心:

  1. 本番経路を再実装しない。get_option_chain → calculate_all →
     next_business_day → save_gex_result を run_daily.run() と同じ並びで呼ぶ。
     違いは as_of の決め方（cron は today、backfill は D+1暦日）と、冪等性/
     上書き/孤児掃除/断耐性のラッパだけ。独立計算式を持たない（誤判断26 回避）。

  2. off-by-one の封じ込め（obs.F と同型の罠）:
     rest.py の _resolve_trade_date(as_of) は「as_of より厳密に前」の直近営業日を
     返す。取引日 D の EOD が欲しければ as_of = D + 1暦日 を渡す。取得後に
     df["trade_date"] == D を確認して、ズレを即検出する。

  3. 冪等性 × 上書きの統一:
     通常 skip = 既存キーが v17形式（data_quality を持つ）かつ data_source ∈
                {rest, rest_backfill, rest_backfill_v2} = 現行パイプライン産。再取得しない。
     --force skip = v2（rest_backfill_v2 ＝ 誤判断32 修正後の再計算済み）のみ。
                stale な rest_backfill / rest / 不在は再計算する（分割実行の自動レジューム）。
     上書き   = それ以外（regime形式 = snapshot汚染/obs.F期、または不在）。
     backfill が書く data_source は "rest_backfill_v2"（誤判断32 以降の provenance タグ）。

  4. キー意味論（obs.G 案2 = session-served）:
     JSON キー = next_business_day(trade_date)。営業日リストの successor で O(1) 算出。

  5. 結果の分類（旧版の反省: NO_DATA と取得失敗を区別できなかった）:
     saved / skipped / no_data（retention・休場の空チェーン）/ errors（取得失敗）
     を別カウント。サマリで原因が一目で分かる。

  6. ネットワーク断への耐性（誤判断: 22:15:59 の DNS 断で 413件が偽 failed 化した件）:
     - fatal が連続したらネットワーク断を疑い、待機して on_date で疎通プローブ。
       復帰すればその日を再試行して継続（瞬断を乗り切る）。
     - 一定時間プローブしても戻らなければ「上流断・resumable」と明示して即停止。
       死んだ上流に残り全日を叩いて偽 failed を量産しない。

  7. 孤児掃除（--clean-orphans, dry-run 既定）:
     範囲内の「非取引日キー」（now() 由来の遺物）を削除。取引日キーは消さない。

実行例（リポジトリ直下から。env var 不要、常に rest Adapter を使う）:
    python tools/backfill_history.py                     # 直近2年
    python tools/backfill_history.py --start 2023-06-07  # 3年に拡張
    python tools/backfill_history.py --dry-run           # 計画のみ（fetch/write なし）
    python tools/backfill_history.py --limit 60          # 新規取得を最大60日で停止（分割消化）
    python tools/backfill_history.py --clean-orphans --dry-run
    python tools/backfill_history.py --clean-orphans

前提:
    Theta Terminal (v3) がローカル起動済み。長時間ジョブなので PC のスリープと
    NIC の省電力は無効にしておくこと（断の自爆を防ぐ）。1日ごとに atomic 保存する
    ため、中断・断のどちらでも、同じコマンドの再実行で済んだ日は SKIP して再開する。
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import date, datetime, timedelta

# tools/ は gex_engine/ の兄弟。リポジトリ直下を import path に通す。
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

# IV 必須（gamma 自前計算）のため backfill の床 = greeks/eod 境界（誤判断29）。
HISTORY_FLOOR = date(2017, 1, 3)

DEFAULT_LOOKBACK_DAYS = 365 * 2          # 既定2年（3年は --start で）
LIST_BUFFER_DAYS = 10                    # successor 確保用に範囲末より先まで走査
RESOLVE_BACK_SCAN_DAYS = 10             # 直近完了取引日を探す過去向き上限

BACKFILL_DATA_SOURCE = "rest_backfill_v2"  # 誤判断32（当日満期除外）以降の再計算分

# ネットワーク断の検知＆復帰待ち（誤判断: DNS 断で偽 failed 量産の再発防止）
NETWORK_SUSPECT_CONSECUTIVE = 3          # fatal が連続 N 回 → 断を疑う
RECOVERY_WAIT_S = 30                      # 疎通プローブの間隔（秒）
RECOVERY_MAX_PROBES = 10                  # 30s × 10 = 最大5分まで復帰を待つ
MAX_DAY_RETRIES = 2                       # 復帰後、同じ日を再試行する上限
ERR_LOG_MAXLEN = 160                      # エラー本文ログの切り詰め（巨大 HTML body 対策）


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
    return datetime.strptime(s, "%Y-%m-%d").date()


def _key_to_date(key: str) -> date | None:
    try:
        return datetime.strptime(key, "%Y.%m.%d").date()
    except (ValueError, TypeError):
        return None


def _short(e: object) -> str:
    """例外メッセージを安全に短縮（500 の巨大 HTML body 等でログを汚さない）。"""
    s = str(e).replace("\n", " ")
    return s if len(s) <= ERR_LOG_MAXLEN else s[:ERR_LOG_MAXLEN] + "…"


def resolve_default_end(fetcher: ThetaRestAdapter, today: date) -> date:
    """直近の完了済み取引日（today より厳密に前の直近営業日）を返す。"""
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
# スキップ判定 / ネットワーク復帰プローブ
# ──────────────────────────────────────────────────────────

def is_current_pipeline_entry(entry: object) -> bool:
    """既存エントリが「現行パイプライン産（通常 run では再取得不要）」か判定する。

    True = v17形式（data_quality を持つ）かつ data_source ∈
           {rest, rest_backfill, rest_backfill_v2}。
    False = regime形式（snapshot汚染 / obs.F期）や mock / 不明 → 上書き対象。

    通常（非 force）run はこの集合を skip する。stale な rest_backfill（誤判断32 前の
    計算）も「通常は再取得しない」ため、修正再 backfill は明示的に --force で行う。
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("data_quality") is None:  # v17 で導入。pre-v17 は regime のみ
        return False
    return entry.get("data_source") in ("rest", "rest_backfill", "rest_backfill_v2")


def is_recomputed_entry(entry: object) -> bool:
    """誤判断32 修正後の再計算で書かれた v2 エントリか判定する。

    --force 時の skip 集合。force は「v2 でない日（stale rest_backfill / rest /
    不在）だけ再計算」とし、既に v2 の日は skip する。これにより --force --limit の
    分割実行が自動レジュームになり、正しい day を無駄に再構築しない。
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("data_quality") is None:
        return False
    return entry.get("data_source") == BACKFILL_DATA_SOURCE


def _probe_network(fetcher: ThetaRestAdapter, probe_date: date) -> bool:
    """on_date（最軽量）で上流疎通を確認。True=疎通あり。"""
    try:
        fetcher.schedule_type_on(probe_date)
        return True
    except Exception:  # noqa: BLE001  断の判定が目的。種別は問わない
        return False


def _wait_for_network(fetcher: ThetaRestAdapter, probe_date: date) -> bool:
    """fatal 連続を検知したら待機しつつ疎通プローブ。復帰したら True。"""
    for p in range(1, RECOVERY_MAX_PROBES + 1):
        logger.warning(
            "ネットワーク断の疑い。%ds 待機して疎通プローブ（%d/%d）...",
            RECOVERY_WAIT_S, p, RECOVERY_MAX_PROBES,
        )
        time.sleep(RECOVERY_WAIT_S)
        if _probe_network(fetcher, probe_date):
            logger.info("疎通回復を確認。処理を再開します。")
            return True
    return False


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
    limit: int | None,
) -> dict:
    """start〜end の各取引日を逐次処理して gex_history.json に追記する。

    Returns:
        {saved, skipped, no_data, errors, aborted} の集計。
        aborted は None / "network"（上流断）/ "limit"（--limit 到達）。
    """
    trading_days = build_trading_days(fetcher, start, end)
    index = {d: i for i, d in enumerate(trading_days)}
    process_days = [d for d in trading_days if start <= d <= end]

    if not process_days:
        logger.warning("対象取引日が 0 件（範囲: %s 〜 %s）。終了。", start, end)
        return {"saved": [], "skipped": [], "no_data": [], "errors": [],
                "aborted": None}

    probe_date = end                       # 疎通プローブ用の確実な取引日
    history = load_history(OUTPUT_PATH)
    total = len(process_days)
    logger.info(
        "バックフィル開始: symbol=%s 取引日 %d 件 (%s 〜 %s) force=%s dry_run=%s limit=%s",
        symbol, total, process_days[0], process_days[-1], force, dry_run, limit,
    )

    saved: list[date] = []
    skipped: list[date] = []
    no_data: list[date] = []
    errors: list[date] = []
    consecutive_fatals = 0
    fresh_fetches = 0
    aborted: str | None = None
    t0 = time.monotonic()

    for i, D in enumerate(process_days, start=1):
        idx = index[D]
        if idx + 1 < len(trading_days):
            session_date = trading_days[idx + 1]
        else:
            session_date = next_business_day(D, fetcher.schedule_type_on)
        key = make_date_key(session_date)

        elapsed = time.monotonic() - t0
        eta = (elapsed / i) * (total - i) if i else 0.0
        prefix = f"[{i}/{total}] D={D} key={key}"

        # ── スキップ判定（fetch 前。済みなら取得を省く）──
        # 通常: 現行パイプライン産（rest/backfill/v2）を skip。
        # force: v2（修正後再計算済み）のみ skip ＝ stale 日だけ再計算（自動レジューム）。
        existing = history.get(key)
        already_done = (
            is_recomputed_entry(existing) if force
            else is_current_pipeline_entry(existing)
        )
        if existing is not None and already_done:
            logger.info("%s SKIP（%s）", prefix, "v2・再計算済み" if force else "v17・現行産")
            skipped.append(D)
            continue

        if dry_run:
            action = "OVERWRITE" if existing is not None else "FETCH"
            logger.info("%s DRY-RUN → %s（elapsed %.0fs, eta %.0fs）",
                        prefix, action, elapsed, eta)
            saved.append(D)
            continue

        if limit is not None and fresh_fetches >= limit:
            logger.info("%s --limit %d 到達。ここで停止（resumable）。", prefix, limit)
            aborted = "limit"
            break

        # ── 1日分を取得（ネットワーク復帰リトライ付き）──
        day_attempt = 0
        while True:
            try:
                df = fetcher.get_option_chain(symbol, D + timedelta(days=1))
            except ThetaPermissionError:
                # 471: プラン権限なし。全日同じ結果なので即 abort。
                logger.exception("%s PERMISSION エラー → 全体中断", prefix)
                raise
            except ThetaDataError as e:
                consecutive_fatals += 1
                logger.error("%s FETCH FAILED（連続%d）: %s",
                             prefix, consecutive_fatals, _short(e))
                if consecutive_fatals >= NETWORK_SUSPECT_CONSECUTIVE:
                    if _wait_for_network(fetcher, probe_date):
                        consecutive_fatals = 0
                        if day_attempt < MAX_DAY_RETRIES:
                            day_attempt += 1
                            logger.info("%s 復帰につきこの日を再試行（%d回目）。",
                                        prefix, day_attempt)
                            continue            # 同じ D を再試行
                        errors.append(D)
                        break
                    # 規定時間プローブしても戻らない → 上流断。resumable に中断。
                    logger.error(
                        "上流が %d 分復帰せず。resumable 中断（saved=%d）。"
                        "ネットワーク確認後に同じコマンドで再開を。",
                        RECOVERY_WAIT_S * RECOVERY_MAX_PROBES // 60, len(saved),
                    )
                    aborted = "network"
                    break
                # まだ断と断定しない範囲 → この日は error として次へ
                errors.append(D)
                break
            else:
                # 取得成功（例外なし）
                consecutive_fatals = 0
                fresh_fetches += 1
                if df.empty:
                    logger.warning("%s NO_DATA（空チェーン・retention/休場相当）", prefix)
                    no_data.append(D)
                    break
                if "trade_date" not in df.columns or df["trade_date"].nunique() != 1:
                    logger.error("%s 契約違反: trade_date 列が不正（誤判断25）→ error",
                                 prefix)
                    errors.append(D)
                    break
                resolved = df["trade_date"].iloc[0].date()
                if resolved != D:
                    logger.error("%s off-by-one: 解決 trade_date=%s != D。要調査 → error",
                                 prefix, resolved)
                    errors.append(D)
                    break
                try:
                    result = calculate_all(df, as_of=D,
                                           data_source=fetcher.source_name)
                    save_gex_result(
                        result, path=OUTPUT_PATH, session_date=session_date,
                        data_source=BACKFILL_DATA_SOURCE,
                    )
                except Exception as e:  # noqa: BLE001  1日の計算/保存失敗で全体を殺さない
                    logger.error("%s 計算/保存で例外 → error: %s", prefix, _short(e))
                    errors.append(D)
                    break
                history = load_history(OUTPUT_PATH)
                logger.info(
                    "%s SAVED dq=%s CW=%s ZG=%s PW=%s（elapsed %.0fs, eta %.0fs）",
                    prefix, result.data_quality, result.call_wall,
                    result.zero_gamma, result.put_wall, elapsed, eta,
                )
                saved.append(D)
                break
        # end inner while

        if aborted:
            break
        if sleep > 0:
            time.sleep(sleep)

    # ── サマリ ──
    tail = f" [中断: {aborted}]" if aborted else ""
    logger.info(
        "バックフィル完了: saved=%d skipped=%d no_data=%d errors=%d%s (経過 %.0fs)",
        len(saved), len(skipped), len(no_data), len(errors), tail,
        time.monotonic() - t0,
    )
    if no_data:
        logger.warning("NO_DATA（空チェーン）%d 件: %s", len(no_data),
                       ", ".join(d.isoformat() for d in no_data))
    if errors:
        logger.warning("取得失敗（error）%d 件: %s", len(errors),
                       ", ".join(d.isoformat() for d in errors))
    if aborted == "network":
        logger.warning("→ ネットワーク復旧後、同じコマンドで再実行すれば保存済みは SKIP され続きから再開。")
    elif aborted == "limit":
        logger.info("→ 同じコマンドを再実行すれば続きを取得（保存済みは SKIP）。")

    return {"saved": saved, "skipped": skipped, "no_data": no_data,
            "errors": errors, "aborted": aborted}


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

    session-served キーは定義上必ず取引日。範囲内で取引日でないキー（例
    2026.05.16 土）= now() 由来の遺物。取引日キーは絶対に消さない。範囲外も不可侵。
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
            continue
        if d not in trading_days:
            orphans.append(key)

    if not orphans:
        logger.info("孤児掃除: 範囲内に孤児キーなし。")
        return []

    for key in orphans:
        logger.info("孤児掃除: %s（%s）を削除%s",
                    key, _key_to_date(key), "（DRY-RUN）" if dry_run else "")

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
    p.add_argument("--limit", type=int, default=None,
                   help="新規取得を最大 N 日で停止（resumable。長時間ジョブの分割消化用）")
    p.add_argument("--clean-orphans", action="store_true",
                   help="範囲内の非取引日キー（now() 由来の孤児）を掃除する")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch も write もせず、実行計画／削除予定のみ表示する")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="各取引日の処理間に挟む待機秒（既定 0）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    fetcher = ThetaRestAdapter(max_retries=3, retry_backoff_base=1.0)
    try:
        today = date.today()
        end = args.end or resolve_default_end(fetcher, today)
        start = args.start or (end - timedelta(days=DEFAULT_LOOKBACK_DAYS))

        if start < HISTORY_FLOOR:
            logger.warning("start=%s は床 %s より古いのでクランプ。", start, HISTORY_FLOOR)
            start = HISTORY_FLOOR
        if end < start:
            logger.error("end=%s が start=%s より前。中断。", end, start)
            return 1

        logger.info("=" * 60)
        logger.info("backfill_history: %s  %s 〜 %s", args.symbol, start, end)
        logger.info("  force=%s limit=%s clean_orphans=%s dry_run=%s",
                    args.force, args.limit, args.clean_orphans, args.dry_run)
        if not args.dry_run and (args.force or args.clean_orphans):
            logger.info("  ※ 既存データを書き換えます。実行前の git commit を推奨。")
        logger.info("=" * 60)

        result = run_backfill(
            fetcher,
            symbol=args.symbol,
            start=start,
            end=end,
            force=args.force,
            dry_run=args.dry_run,
            sleep=args.sleep,
            limit=args.limit,
        )

        if args.clean_orphans:
            if result.get("aborted"):
                logger.warning("backfill が中断(%s)したため孤児掃除をスキップ。"
                               "完了後に再実行を。", result["aborted"])
            else:
                clean_orphans(fetcher, start=start, end=end, dry_run=args.dry_run)

        # ネットワーク断で中断した場合は非ゼロ終了（気づけるように）
        return 2 if result.get("aborted") == "network" else 0

    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error: %s", e)
        return 1
    finally:
        close = getattr(fetcher, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
