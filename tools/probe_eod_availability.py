#!/usr/bin/env python3
"""
tools/probe_eod_availability.py

【目的】(PC_GOVERNANCE 誤判断36 対応・単発診断ツール、CI 非実行)
open_interest（OPRA 公式 EOD レポート）が、当日朝の何時ごろに
安定して取得可能になるかを実測する。

【設計上の決定（合意済み、詳細は該当チャットログ参照）】
- 律速は open_interest 側（OPRA が取引日 T の OI を T+1 日 06:30 ET に配信）。
  greeks/eod（ThetaData 自前生成、前日 17:15 ET 生成）は「cron を翌朝へ
  移す」前提のもとでは 12 時間以上の余裕があるため、本スクリプトの
  ポーリングループ対象からは外し、起動時に 1 回だけ健全性チェックする
  （境界測定ではなく、前夜の生成自体が失敗していないかの確認）。
- 成功判定は「行数が 2 回連続で同一（かつ > 0）」＝生成完了とみなす。
  固定の行数閾値や固定時刻を決め打たず、その日のチェーンサイズに対する
  相対判定にすることで恣意的な閾値を持ち込まない（カーブフィッティング回避）。
- ポーリング間隔は疎密可変:
    06:00-07:00 ET : 5分間隔（立ち上がりの形状を精密に捉える）
    07:00-09:00 ET : 30分間隔（安全圏、コスト削減）
    09:00-09:25 ET : 5分間隔（締切直前の遅延を必ず捕捉）
  09:30 ET（NY 現物寄り付き）が実務上の締切のため 09:25 で打ち切る。
- rest.py（本番 Adapter）には依存しないスタンドアロン実装。
  tools/probe_thetadata_history.py と同じ方針（PC_GOVERNANCE 誤判断29 系）。
  カレンダー遡及ロジック（前営業日 T の決定）は _resolve_trade_date と
  同じ「as_of の前日から遡る」方式を、本ファイル内に独立実装する。
- エラーコード分類は PC_PIPELINE §5.6 の表をそのまま踏襲する
  （200=成功／472=NO_DATA(過渡状態、生成待ち)／429,470,474,570,571=
  リトライ可能／404,471,473,475,476,477,478,572=致命的で即例外）。

【前提・要確認事項】
- 接続先は本番 rest.py と同じくローカルの Theta Terminal REST プロキシ
  （既定 http://127.0.0.1:25503/v3）。GitHub Actions 上で実行する場合は
  既存 update_gex.yml が持つ Theta Terminal 起動手順（JDK セットアップ→
  jar 起動→認証→ヘルスチェック待機）を workflow 側で用意すること。
  本スクリプト自体はその起動を行わない。
- calendar/on_date の応答フォーマットは PC_PIPELINE §5.2 記載の
  `type ∈ {open, early_close, full_close, weekend}` を前提にしている。
  実 API の実際のキー名が異なる場合はここで KeyError が出る
  （サイレントなフォールバックはしない＝誤判断26 の教訓）。

【出力】
<output-dir>/oi_boundary_<T:YYYY-MM-DD>.jsonl
  1 行 1 ポーリングの JSON レコード。個人購読データの中身（IV・OI の
  実数値そのもの）は一切含まない。含むのはメタデータ（時刻・行数・
  ステータスコード）のみなので、artifact として保存して問題ない。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ET = ZoneInfo("America/New_York")

# PC_PIPELINE §5.6 のエラーコード分類（無断で拡張・変更しない）
RETRYABLE_CODES = {429, 470, 474, 570, 571}
NO_DATA_CODE = 472
FATAL_CODES = {404, 471, 473, 475, 476, 477, 478, 572}

TRADING_DAY_TYPES = {"open", "early_close"}

_MAX_CALENDAR_SCAN_DAYS = 10  # rest.py の同名定数と同じ安全弁。無限ループ防止。

_WINDOW_0700 = datetime.strptime("07:00", "%H:%M").time()
_WINDOW_0900 = datetime.strptime("09:00", "%H:%M").time()
_CUTOFF_0925 = datetime.strptime("09:25", "%H:%M").time()


@dataclass
class PollRecord:
    poll_time_et: str
    poll_time_utc: str
    trade_date: str
    endpoint: str
    http_status: int
    row_count: int
    elapsed_ms: int
    content_hash: str = ""
    note: str = ""


def now_et() -> datetime:
    return datetime.now(ET)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_date(d: date) -> str:
    return d.strftime("%Y%m%d")


def _get_with_retry(
    client: httpx.Client, url: str, params: dict, max_retries: int = 3
) -> tuple[httpx.Response | None, int]:
    """
    低レベル GET + リトライ。戻り値: (response_or_None, elapsed_ms)
    ・200 / 472(NO_DATA) はそのまま応答を返す（呼び出し側が解釈する）。
    ・RETRYABLE はここで指数バックオフして再試行。
    ・FATAL は即座に例外を送出（安全側に倒す。隠蔽しない）。
    ・リトライ上限到達、または通信例外が続いた場合は (None, 0) を返す
      （呼び出し側はこの回のサンプルを「欠測」として扱う）。
    """
    backoff = 2.0
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            resp = client.get(url, params=params, timeout=30.0)
        except httpx.TransportError:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None, 0
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        code = resp.status_code
        if code == 200 or code == NO_DATA_CODE:
            return resp, elapsed_ms

        if code in FATAL_CODES:
            raise RuntimeError(
                f"FATAL error code {code} from {url} params={params}: {resp.text[:200]}"
            )

        if code in RETRYABLE_CODES and attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
            continue

        # 未分類コード: サイレントなフォールバック禁止、例外化する
        raise RuntimeError(
            f"未分類の応答コード {code} from {url} params={params}: {resp.text[:200]}"
        )

    return None, 0


def fetch_calendar_type(client: httpx.Client, base_url: str, d: date) -> str:
    """
    calendar/on_date の実際の応答仕様（公式APIリファレンス確認済み、2026-07-08）:
    - format パラメータ省略時のデフォルトは CSV（JSONではない）。
    - JSON指定時でも応答は単一オブジェクトではなく「配列」
      （array of {type, open, close}）。
    ここでは他エンドポイント（open_interest, greeks/eod）と同じくCSVで統一し、
    ヘッダ行を除いた最初のデータ行から type 列を取り出す。
    """
    resp, _ = _get_with_retry(
        client, f"{base_url}/calendar/on_date", {"date": fmt_date(d), "format": "csv"}
    )
    if resp is None:
        raise RuntimeError(f"calendar/on_date({d}) が通信エラーで取得できなかった")
    if resp.status_code == NO_DATA_CODE:
        raise RuntimeError(f"calendar/on_date({d}) で NO_DATA は想定外の応答")
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    if not rows:
        raise RuntimeError(
            f"calendar/on_date({d}) の応答が空で type を取得できない: {resp.text[:200]!r}"
        )
    return rows[0]["type"]


def resolve_previous_trading_day(client: httpx.Client, base_url: str, as_of: date) -> date:
    """as_of の直近過去営業日 T を返す（_resolve_trade_date と同型の遡及方式、独立実装）。"""
    candidate = as_of - timedelta(days=1)
    for _ in range(_MAX_CALENDAR_SCAN_DAYS):
        if fetch_calendar_type(client, base_url, candidate) in TRADING_DAY_TYPES:
            return candidate
        candidate -= timedelta(days=1)
    raise RuntimeError(
        f"直近営業日が {_MAX_CALENDAR_SCAN_DAYS} 日遡っても見つからない。"
        f"calendar/on_date の応答異常、または休場が異常に連続している可能性。"
    )


def count_csv_rows(text: str) -> int:
    """CSV 応答のデータ行数（ヘッダ除く）を数える。中身は一切ログに残さない。"""
    rows = list(csv.reader(io.StringIO(text)))
    return max(0, len(rows) - 1) if rows else 0


def compute_content_hash(text: str) -> str:
    """
    レスポンス本文全体のハッシュ（先頭16文字に短縮）。
    行数が同じでも中身(各行のOI値)が変わっていないかを検出するための
    独立したシグナル。個人購読データそのものはログに残さず、
    ハッシュ値だけを記録する。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fetch_row_count(
    client: httpx.Client, url: str, params: dict
) -> tuple[int, int, int, str]:
    """戻り値: (http_status, row_count, elapsed_ms, content_hash)。通信欠測時は (-1, -1, 0, "")。"""
    resp, elapsed_ms = _get_with_retry(client, url, params)
    if resp is None:
        return -1, -1, 0, ""
    if resp.status_code == NO_DATA_CODE:
        return resp.status_code, 0, elapsed_ms, ""
    return resp.status_code, count_csv_rows(resp.text), elapsed_ms, compute_content_hash(resp.text)


def interval_seconds_for(now: datetime) -> int:
    """疎密可変のポーリング間隔。"""
    t = now.timetz().replace(tzinfo=None)
    if t < _WINDOW_0700:
        return 5 * 60
    if t < _WINDOW_0900:
        return 30 * 60
    return 5 * 60


def _append_record(out_path: Path, rec: PollRecord) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def run_probe(
    symbol: str, base_url: str, output_dir: Path, trade_date_override: date | None = None
) -> int:
    with httpx.Client() as client:
        if trade_date_override is not None:
            # 明示指定モード: 「今日が取引日か」のゲートは適用しない。
            # 週末を挟む金曜分などを、土曜日(非取引日)に起動して狙う場合に使う。
            # 指定日自体が取引日だったかだけは確認する(誤指定の早期検出)。
            override_type = fetch_calendar_type(client, base_url, trade_date_override)
            if override_type not in TRADING_DAY_TYPES:
                print(
                    f"--trade-date で指定された {trade_date_override} は取引日ではない "
                    f"(type={override_type})。指定を確認すること。"
                )
                return 0
            trade_date = trade_date_override
            print(f"probe 対象 T = {trade_date}（--trade-date で明示指定）")
        else:
            today_et = now_et().date()

            # 前提チェック: 今日自体が取引日でなければ probe を実行せず即終了
            today_type = fetch_calendar_type(client, base_url, today_et)
            if today_type not in TRADING_DAY_TYPES:
                print(f"{today_et} は取引日ではない (type={today_type})。probe を実行せず終了。")
                return 0

            trade_date = resolve_previous_trading_day(client, base_url, today_et)
            print(f"probe 対象 T = {trade_date}（今日 {today_et} の前営業日）")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"oi_boundary_{trade_date.isoformat()}.jsonl"

        # --- 診断: greeks/eod の健全性チェック（1回のみ、ループしない） ---
        # 翌朝 cron 前提では前夜生成分に 12 時間以上の余裕があるはずで、
        # 常に成功する想定。ここで失敗する場合は「境界」ではなく前夜の
        # 生成自体が失敗した別障害として扱う。
        try:
            g_status, g_rows, g_ms, g_hash = fetch_row_count(
                client,
                f"{base_url}/option/history/greeks/eod",
                {
                    "symbol": symbol,
                    "expiration": "*",
                    "start_date": fmt_date(trade_date),
                    "end_date": fmt_date(trade_date),
                    "format": "csv",
                },
            )
            _append_record(
                out_path,
                PollRecord(
                    poll_time_et=now_et().isoformat(),
                    poll_time_utc=now_utc_iso(),
                    trade_date=trade_date.isoformat(),
                    endpoint="greeks/eod(diagnostic_once)",
                    http_status=g_status,
                    row_count=g_rows,
                    elapsed_ms=g_ms,
                    content_hash=g_hash,
                    note="前日夜間生成分の健全性チェック。境界計測の対象外。",
                ),
            )
            if g_rows <= 0:
                print(
                    "警告: greeks/eod(前日分) が 0 行以下。翌朝 cron の前提"
                    "（前夜生成が完了している）が崩れている可能性。要調査。"
                    " OI probe は続行する。"
                )
        except Exception as exc:  # noqa: BLE001 - 診断チェックは失敗しても本題を止めない
            print(f"警告: greeks/eod 健全性チェックで例外: {exc}。OI probe は続行する。")

        # --- 本題: open_interest の境界を実測するループ ---
        # cutoff は「T がいつか」に関わらず、常に「今実行している暦日の09:25 ET」。
        # --trade-date 明示指定モード(土曜起動など)でも、実行している暦日自体は
        # 土曜のままなので、その日の09:25が締切になる(通常モードと同じ理屈)。
        run_date = now_et().date()
        cutoff = datetime.combine(run_date, _CUTOFF_0925, tzinfo=ET)
        prev_row_count: int | None = None
        prev_content_hash: str | None = None
        stable_count = 0
        converged = False
        first_poll = True

        while now_et() < cutoff:
            t_now = now_et()
            status, row_count, elapsed_ms, content_hash = fetch_row_count(
                client,
                f"{base_url}/option/history/open_interest",
                {"symbol": symbol, "expiration": "*", "date": fmt_date(trade_date), "format": "csv"},
            )
            note = "" if status != -1 else "通信欠測（リトライ上限到達）。この回は欠測扱い。"

            if first_poll and row_count > 0:
                # cron の発火遅延（schedule トリガー時のジッター）等で、
                # このジョブ自体の開始が想定より遅く、真の境界が既に
                # 過ぎていた可能性がある。恣意的な閾値ではなく「観測の
                # 限界」を明示するフラグ（左側打ち切り＝真値は不明、この
                # 時刻より前としか言えない）。後段の分析で必ず参照すること。
                note = (
                    (note + " " if note else "")
                    + "警告(左側打ち切り): 初回ポーリングから既に非ゼロ行。"
                    "真の境界はこの実行開始時刻より前にある可能性があり、"
                    "この日のサンプルを境界の実測値として信用してはならない。"
                )
            first_poll = False

            _append_record(
                out_path,
                PollRecord(
                    poll_time_et=t_now.isoformat(),
                    poll_time_utc=now_utc_iso(),
                    trade_date=trade_date.isoformat(),
                    endpoint="open_interest",
                    http_status=status,
                    row_count=row_count,
                    elapsed_ms=elapsed_ms,
                    content_hash=content_hash,
                    note=note,
                ),
            )
            print(f"[{t_now.strftime('%H:%M:%S %Z')}] status={status} rows={row_count} hash={content_hash} {note}")

            # 収束判定: 行数だけでなく content_hash も2回連続で一致することを
            # 要求する。行数が同じでも中身(各行のOI値)が訂正されている
            # ケースを、行数だけの判定では見逃すため（2026-07-XX 追加）。
            is_stable_match = (
                row_count > 0
                and row_count == prev_row_count
                and content_hash
                and content_hash == prev_content_hash
            )
            if is_stable_match:
                stable_count += 1
                if stable_count >= 2:
                    converged = True
                    print(
                        f"収束: {t_now.strftime('%H:%M:%S %Z')} 時点で行数 {row_count} と"
                        f"content_hash が2回連続一致。T={trade_date} の OI 境界とみなす。"
                    )
                    break
            else:
                stable_count = 0

            if row_count >= 0:
                prev_row_count = row_count
            if content_hash:
                prev_content_hash = content_hash

            time.sleep(interval_seconds_for(now_et()))

        if not converged:
            print(
                f"未収束: 09:25 ET までに行数の安定を確認できなかった。"
                f"T={trade_date} はこの日、締切内に OI が揃わなかった可能性がある。"
                f"ログ({out_path})を必ず確認すること。"
            )
            return 1

        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--base-url", default="http://127.0.0.1:25503/v3")
    parser.add_argument("--output-dir", default="probe_results", type=Path)
    parser.add_argument(
        "--trade-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "対象取引日を明示指定する。省略時は実行日の前営業日を自動解決する"
            "(平日朝の通常運用)。週末を挟む金曜分を土曜日に観測する場合など、"
            "「今日」自体が取引日でない日に起動する際に使う。"
        ),
    )
    args = parser.parse_args()

    trade_date_override = None
    if args.trade_date:
        try:
            trade_date_override = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
        except ValueError:
            print(
                f"FATAL: --trade-date の形式が不正: {args.trade_date!r} "
                f"(YYYY-MM-DD 形式で指定すること)",
                file=sys.stderr,
            )
            return 2

    try:
        return run_probe(args.symbol, args.base_url, args.output_dir, trade_date_override)
    except Exception:
        # rc=1(未収束、想定内)と rc=2(予期しない例外、実装バグ)を区別する。
        # workflow 側は rc=2 だけを「本物の失敗」として扱う設計
        # （2026-07-08 のインシデント: continue-on-error が両者を区別
        #   できず、クラッシュを「成功」として見せてしまっていたため）。
        import traceback

        traceback.print_exc()
        print(
            "FATAL: probe が予期しない例外で異常終了した(rc=2)。"
            "これは「未収束」とは異なり、実装のバグの可能性が高い。",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
