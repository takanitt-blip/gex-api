"""Step 1A (第3回・最終): ThetaData /option/history/greeks/eod の生レスポンス調査。

これが手探りダンプの最終回。これ以降は rest.py の再設計に入る。

ここまでの経緯（Step 1A の全履歴）:
    第1回:
      [1] /option/history/open_interest        → 200。13,228行/805KB。
          symbol,expiration,strike,right,timestamp,open_interest
          right は "CALL"/"PUT" 大文字。expiration=* OK。
      [2] /option/history/greeks/implied_volatility (expiration=*)
          → 400 "Cannot specify '*' for the date"
      [3] /v3/hist/stock/eod (root / symbol)   → 410 / 404。v2 系の死んだパス。
    第2回:
      [2] /option/history/greeks/implied_volatility (expiration=1満期, strike=*)
          → 200 だが 12,355,730行 / 1.6GB。
          後にドキュメントで原因判明: このエンドポイントは interval 必須
          パラメータの Default が "1s"。指定しなかったため 1 秒足の
          日中時系列（09:30-16:00 = 23,400点/contract）を引いた。
          → エンドポイントが重いのではなく interval 見落としが真因。

公式ドキュメント精読の結論（snapshot→history という大きな方針転換に伴い、
全 history 系エンドポイントのドキュメントを読み直した結果）:
    - OI:      /option/history/open_interest を使う。
               date=(T+1) で取引日 T の EOD。OPRA 報告構造由来の
               1営業日オフセットは start_date 指定でも消えない。
    - IV+spot: /option/history/greeks/eod を使う。★今回ダンプする対象
               17:15 ET 生成の確定 EOD レポート。implied_vol と
               underlying_price を同梱。interval パラメータは存在しない
               （= greeks/implied_volatility のような 1s 既定の地雷はない、はず）。
               start_date=end_date=(T) の「当日付」規約。
               expiration=* は「1日ずつ」制約あり（日次 cron なら問題なし）。
    - gamma:   eod レスポンスに gamma は同梱されるが使わない。
               Zero Gamma 計算が任意 S* での gamma 関数を要求するため、
               black_scholes.py による自前計算を維持する。
               eod からは implied_vol と underlying_price のみ採用。

今回の唯一の目的:
    /option/history/greeks/eod を 1 満期だけ叩き、実レスポンスを実証する。
    特に「1 満期で何行返るか」。
        数百行       → 「1 contract 1 行」。expiration=* 本番が安全と確定。
        数百万行     → 隠れた既定パラメータがある（ドキュメントの読み落とし）。
                       greeks/implied_volatility の interval=1s と同じ轍。

probe パラメータ設計（ドキュメント index 15 準拠）:
    symbol      = SPY
    expiration  = 2026-06-18   1 満期。expiration=* の 1/36 サイズで
                               地雷があっても軽傷に留める。第1回 OI ダンプで
                               実在確認済み、実行日(5/16)時点でも未失効。
    start_date  = 20260512     greeks/eod は「当日付」規約。5/12(月) EOD を狙う。
    end_date    = 20260512     start=end で 1 日のみ。
    strike      指定しない     Default: *。1 満期なら全 strike でも数百行のはず。
    format      = csv

このスクリプトの設計方針（第1回から不変）:
    - gex_engine を import しない（依存ゼロ、httpx と標準ライブラリのみ）。
    - 一切パースしない。生テキストを保存し、ヘッダと先頭数行のみ表示。
    - エラーでも落とさない。4xx も「収穫」として記録する。

使い方:
    GitHub Actions の workflow_dispatch から呼ぶ前提。
        python tools/dump_history_raw.py
    出力は ./history_dump/ 配下に保存され artifact としてアップロードされる。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ──────────────────────────────────────────────────────────
# 調査対象の定義
# ──────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:25503/v3"

# greeks/eod は「当日付」規約（17:15 ET 生成レポート、OPRA 翌朝報告ではない）。
# 取引日 5/12(月) の EOD を狙うので start_date=end_date=20260512。
TARGET_TRADE_DATE = "20260512"

# 1 満期に絞る。expiration=* の 1/36 サイズで地雷を軽傷化する。
# 第1回 OI ダンプに実在を確認済み。実行日(5/16)時点でも未失効。
TARGET_EXPIRATION = "2026-06-18"

SYMBOL = "SPY"

OUTPUT_DIR = Path("./history_dump")

# 今回は greeks/eod の 1 満期 probe のみ。
PROBES: list[dict] = [
    {
        "name": "4_option_history_greeks_eod",
        "path": "/option/history/greeks/eod",
        "params": {
            "symbol": SYMBOL,
            "expiration": TARGET_EXPIRATION,
            "start_date": TARGET_TRADE_DATE,
            "end_date": TARGET_TRADE_DATE,
            # strike は指定しない（Default: *）。
            # interval パラメータはこのエンドポイントには存在しない
            # （ドキュメント index 15 で確認済み）。
        },
    },
]

# 標準出力に表示する生テキストの先頭行数（保存ファイルには全文を残す）。
PREVIEW_LINES = 8


# ──────────────────────────────────────────────────────────
# 1 probe の実行（第1回から不変のロジック）
# ──────────────────────────────────────────────────────────

def run_probe(client: httpx.Client, probe: dict) -> dict:
    """1 エンドポイントを叩き、結果を保存して要約 dict を返す。

    エラーでも例外を投げない。HTTP ステータスと生ボディを記録し続ける。
    通信レベルのエラー（接続不可等）のみ例外情報を記録する。
    """
    path = probe["path"]
    params = {**probe["params"], "format": "csv"}
    url = f"{BASE_URL}{path}"

    print(f"\n{'=' * 70}")
    print(f"PROBE: {probe['name']}")
    print(f"  GET {url}")
    print(f"  params: {params}")
    print(f"{'-' * 70}")

    summary: dict = {
        "name": probe["name"],
        "url": url,
        "params": params,
    }

    try:
        response = client.get(url, params=params)
    except httpx.HTTPError as e:
        print(f"  [NETWORK ERROR] {type(e).__name__}: {e}")
        summary["network_error"] = f"{type(e).__name__}: {e}"
        return summary

    body = response.text
    status = response.status_code
    summary["status_code"] = status
    summary["body_bytes"] = len(body)

    # 生ボディを「全文」ファイルに保存する（パースしない）。
    out_path = OUTPUT_DIR / f"{probe['name']}__status{status}.csv"
    out_path.write_text(body, encoding="utf-8")
    summary["saved_to"] = str(out_path)

    print(f"  HTTP {status}  ({len(body)} bytes)  -> saved {out_path}")

    lines = body.splitlines()
    summary["total_lines"] = len(lines)
    print(f"  total lines: {len(lines)}")
    if lines:
        print(f"  --- header (line 1) ---")
        print(f"  {lines[0]}")
        if len(lines) > 1:
            print(f"  --- first {min(PREVIEW_LINES, len(lines) - 1)} data rows ---")
            for ln in lines[1 : 1 + PREVIEW_LINES]:
                print(f"  {ln}")
        summary["header_line"] = lines[0]
    else:
        print("  (empty body)")
        summary["header_line"] = ""

    return summary


# ──────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1A (3rd / final run): /option/history/greeks/eod raw dump")
    print(f"  base_url:    {BASE_URL}")
    print(f"  symbol:      {SYMBOL}")
    print(f"  expiration:  {TARGET_EXPIRATION} (single expiry, 1/36 of expiration=*)")
    print(f"  date range:  {TARGET_TRADE_DATE}..{TARGET_TRADE_DATE} "
          f"(greeks/eod uses the trade-date convention)")
    print(f"  run_utc:     {datetime.now(timezone.utc).isoformat()}")

    summaries: list[dict] = []
    with httpx.Client(timeout=60.0) as client:
        for probe in PROBES:
            summaries.append(run_probe(client, probe))

    summary_path = OUTPUT_DIR / "_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    print(f"DONE. summary -> {summary_path}")
    print(f"{'=' * 70}")
    print("\n観察してほしいポイント:")
    print("  [A] total_lines は何行か ★最重要")
    print("      数百行   → 1 contract 1 行。expiration=* 本番が安全と確定")
    print("      数百万行 → 隠れた既定パラメータあり。ドキュメント読み落とし")
    print("  [B] ヘッダに implied_vol と underlying_price があるか")
    print("      （gamma 等も同梱されるはずだが、使わないので有無は問わない）")
    print("  [C] right は CALL/PUT 大文字か（OI history と揃っているか）")
    print("  [D] timestamp 列が 1 種類か（日中時系列でないことの確認）")

    all_network_failed = all("network_error" in s for s in summaries)
    if all_network_failed:
        print("\n[FATAL] 全 probe が通信エラー。Theta Terminal 未起動の疑い。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
