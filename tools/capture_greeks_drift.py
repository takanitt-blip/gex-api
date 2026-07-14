#!/usr/bin/env python3
"""
tools/capture_greeks_drift.py

【目的】(単発診断・影響定量化ツール)
probe_eod_availability.py で判明した「greeks/eod が出現後も単一値に収束せず、
複数バージョン(hash A/B/C...)を循環し続ける」現象について、その循環が
最終成果物である GEX ウォール(Call Wall / Put Wall / Zero Gamma)に
どれだけの影響を与えるかを定量化するために、greeks と OI の
【実データ本体】を時系列で採取・保存する。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【!!! 最重要 !!! このスクリプトは絶対にローカル(Windows)機でのみ実行する】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 本スクリプトは greeks/eod と open_interest の【レスポンス本体(CSV)】を
  ディスクに保存する。これは ThetaData の購読データそのものであり、
  再配布不可・GitHub へのコミット禁止・GitHub Actions での実行禁止。
- probe_eod_availability.py が「hash のみ保存し購読データを残さない」設計
  だったのに対し、本スクリプトは影響測定のために【あえて本体を保存する】。
  この違いゆえに、実行環境をローカルに限定する必要がある。
- 保存先ディレクトリ(既定 greeks_drift_snapshots/)は必ず .gitignore に
  追加すること(PC_VALIDATION §2.4 / obs.I 巻き添え事故の教訓)。

【何を保存するか】
各ポーリング回ごとに:
  - greeks/eod のレスポンス本体 → snapshots/<T>/greeks_<HHMMSS>_<hash>.csv
  - open_interest のレスポンス本体 → snapshots/<T>/oi_<HHMMSS>_<hash>.csv
  - メタデータ(時刻・hash・行数) → snapshots/<T>/manifest.jsonl
OI もペアで保存するのは、GEX 計算に gamma(greeks側)と OI(oi側)の
両方が必要なため。greeks だけ採取しても後で GEX を計算できない。

【使い方(ローカル Windows、Theta Terminal 起動済みの状態で)】
  python tools/capture_greeks_drift.py --trade-date 2026-07-13 --count 10 --interval 300
    → T=2026-07-13 の greeks/OI を、5分(300秒)間隔で10回採取して保存
  同一 hash が連続しても保存し続ける(循環の周期を捉えるため)。
  --count 回取得したら終了。Ctrl-C でいつでも中断可(それまでの分は保存済み)。

【この後の分析(別途)】
  採取した greeks の各バージョン(A/B/C...)と OI をペアにして、本番と同じ
  GEX ロジックで Call/Put Wall・Zero Gamma を計算し、バージョン間で
  ウォールが何ポイントずれるかを比較する。ズレが誤差レベルなら
  「循環は無視してよい(掴んだ値でロック)」、大きいなら対策が必要、と判断する。
  ※本スクリプトは「素材採取」まで。GEX 計算・比較は分析フェーズで行う。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ET = ZoneInfo("America/New_York")

NO_DATA_CODE = 472
RETRYABLE_CODES = {429, 470, 474, 570, 571}
FATAL_CODES = {404, 471, 473, 475, 476, 477, 478, 572}


def now_et() -> datetime:
    return datetime.now(ET)


def fmt_date_compact(d: date) -> str:
    return d.strftime("%Y%m%d")


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _get(client: httpx.Client, url: str, params: dict, max_retries: int = 3):
    backoff = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params, timeout=60.0)
        except httpx.TransportError:
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            return None
        code = resp.status_code
        if code == 200 or code == NO_DATA_CODE:
            return resp
        if code in FATAL_CODES:
            raise RuntimeError(f"FATAL {code} from {url}: {resp.text[:200]}")
        if code in RETRYABLE_CODES and attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(f"未分類の応答コード {code} from {url}: {resp.text[:200]}")
    return None


def count_data_rows(text: str) -> int:
    lines = text.splitlines()
    return max(0, len(lines) - 1)


def capture(symbol: str, base_url: str, trade_date: date, out_dir: Path,
            count: int, interval: int) -> int:
    day_dir = out_dir / trade_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    manifest = day_dir / "manifest.jsonl"

    endpoints = {
        "greeks": {
            "url": f"{base_url}/option/history/greeks/eod",
            "params": {
                "symbol": symbol, "expiration": "*",
                "start_date": fmt_date_compact(trade_date),
                "end_date": fmt_date_compact(trade_date),
                "format": "csv",
            },
        },
        "oi": {
            "url": f"{base_url}/option/history/open_interest",
            "params": {
                "symbol": symbol, "expiration": "*",
                "date": fmt_date_compact(trade_date), "format": "csv",
            },
        },
    }

    seen_hashes = {"greeks": set(), "oi": set()}

    with httpx.Client() as client:
        for i in range(count):
            t = now_et()
            hhmmss = t.strftime("%H%M%S")
            rec = {
                "poll_time_et": t.isoformat(),
                "poll_time_utc": datetime.now(timezone.utc).isoformat(),
                "trade_date": trade_date.isoformat(),
                "iteration": i + 1,
            }

            for name, spec in endpoints.items():
                resp = _get(client, spec["url"], spec["params"])
                if resp is None:
                    rec[name] = {"status": -1, "note": "通信欠測"}
                    print(f"[{t.strftime('%H:%M:%S %Z')}] {name}: 通信欠測")
                    continue
                if resp.status_code == NO_DATA_CODE:
                    rec[name] = {"status": 472, "rows": 0, "hash": "", "saved": None}
                    print(f"[{t.strftime('%H:%M:%S %Z')}] {name}: 472 (未生成)")
                    continue

                text = resp.text
                h = short_hash(text)
                rows = count_data_rows(text)
                # 本体を保存(hash をファイル名に含め、同一 hash は1度だけ保存して容量節約。
                # ただし「いつ・どの hash が返ったか」は manifest に毎回記録するので
                # 循環の時系列は完全に追える)
                is_new = h not in seen_hashes[name]
                fpath = day_dir / f"{name}_{h}.csv"
                if is_new:
                    fpath.write_text(text, encoding="utf-8")
                    seen_hashes[name].add(h)
                rec[name] = {
                    "status": 200, "rows": rows, "hash": h,
                    "saved_file": fpath.name if is_new else f"(既出: {fpath.name})",
                }
                flag = " ★新バージョン保存" if is_new else " (既出hash)"
                print(f"[{t.strftime('%H:%M:%S %Z')}] {name}: rows={rows} hash={h}{flag}")

            with manifest.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i < count - 1:
                time.sleep(interval)

    # サマリ
    print("\n=== 採取完了 ===")
    for name in endpoints:
        print(f"  {name}: {len(seen_hashes[name])} 種類のユニークな hash を採取 "
              f"→ {sorted(seen_hashes[name])}")
    print(f"  保存先: {day_dir}")
    print(f"  manifest: {manifest}")
    print("\n次の分析: 各 greeks バージョンと OI をペアにして GEX ウォールを計算し、"
          "バージョン間の Call/Put Wall・Zero Gamma のズレを比較する。")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--base-url", default="http://127.0.0.1:25503/v3")
    p.add_argument("--trade-date", required=True, metavar="YYYY-MM-DD",
                   help="採取対象の取引日 T")
    p.add_argument("--count", type=int, default=10, help="採取回数(既定10)")
    p.add_argument("--interval", type=int, default=300,
                   help="採取間隔(秒、既定300=5分)")
    p.add_argument("--output-dir", type=Path, default=Path("greeks_drift_snapshots"))
    args = p.parse_args()

    try:
        td = datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    except ValueError:
        print(f"FATAL: --trade-date の形式が不正: {args.trade_date!r}", file=sys.stderr)
        return 2

    print("━" * 50)
    print("capture_greeks_drift.py — ローカル専用・購読データ本体を保存します")
    print("GitHub Actions での実行・リポジトリへのコミットは禁止です")
    print(f"対象 T={td}  採取={args.count}回  間隔={args.interval}秒")
    print("━" * 50)

    try:
        return capture(args.symbol, args.base_url, td, args.output_dir,
                       args.count, args.interval)
    except KeyboardInterrupt:
        print("\n中断されました(それまでの採取分は保存済み)。")
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
