"""Step 1A: ThetaData history エンドポイントの生レスポンス調査ツール。

目的:
    snapshot → history への rest.py 作り直しに先立ち、history 系
    エンドポイントの「実レスポンス」を 1 営業日分ダンプして目視確認する。

    PC_GOVERNANCE.md 誤判断 16/17/18 の共通根本原因
    （公式ドキュメント/既存事例を信頼しすぎて実 API 生レスポンスでの
    確認を怠った）への恒久対策。本番 rest.py を書く前に、必ずこれで
    実構造を「契約」として確定する。

このスクリプトの設計方針:
    - gex_engine を import しない（依存ゼロ、httpx と標準ライブラリのみ）。
      → 本番コードのバグに巻き込まれず、純粋に API 観察に徹する。
    - 一切パースしない。pandas.read_csv すら呼ばない。
      → パースは「解釈」。解釈が入った時点で観察ではなくなる。
        生テキストをそのまま保存し、ヘッダと先頭数行だけ標準出力に出す。
    - エラーでも落とさない。404 や 472 も「収穫」として記録する。
      → 404 (NO_IMPL) が返ればパスの仮定が外れていると分かる。
        それ自体が Step 1A の目的の一部。

確認したい 6 項目（README 代わり）:
    [1] /v3/option/history/open_interest
        - 列名一覧 / right は CALL/PUT 大文字か call/put 小文字か
        - expiration の形式 / date 列が応答に含まれるか / 行数
    [2] /v3/option/history/greeks/implied_volatility
        - IV 列名は implied_vol か / underlying_price が同梱されているか★
    [3] /v3/hist/stock/eod
        - root か symbol か / 返る行数 / date 列の実値★★
        - option history の date ズレと規約が一致するか不一致か

★★ : start_date と end_date を「わざと 2 日レンジ」で投げる。
      1 日だけ投げると規約を逆算できないが、2 日レンジなら返ってきた
      date 列の実値を見てズレの有無を判定できる。

使い方:
    GitHub Actions の workflow_dispatch から呼ぶ前提。
    Theta Terminal が localhost:25503 で起動している環境で実行する。

        python tools/dump_history_raw.py

    出力は ./history_dump/ 配下に生 CSV（または生ボディ）として保存され、
    Actions の artifact としてアップロードされる。
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

# 申し送りマッピング:
#   gex_history["2026.05.12"] ↔ option history date=20260513
# を再現する。OI/IV は「翌日付 = 前営業日 EOD」の仮説で 20260513 を投げる。
TARGET_OPTION_DATE = "20260513"   # → 5/12(月) EOD の OI/IV を期待

# stock EOD は「翌朝報告」の事情がない（取引所が引け後に即確定する）ため、
# date 規約が option history と一致しない可能性がある。
# わざと 2 日レンジで投げ、返ってきた date 列の実値で規約を逆算する。
STOCK_START_DATE = "20260512"
STOCK_END_DATE = "20260513"

SYMBOL = "SPY"

OUTPUT_DIR = Path("./history_dump")

# 調査対象エンドポイント。
# 各 dict: name(保存ファイル名), path, params
# パス文字列は申し送り/リサーチ結果のものをそのまま使う。
# これらが 404 を返したら「パスの仮定が外れている」という収穫。
PROBES: list[dict] = [
    {
        "name": "1_option_history_open_interest",
        "path": "/option/history/open_interest",
        "params": {
            "symbol": SYMBOL,
            "expiration": "*",
            "strike": "*",
            "date": TARGET_OPTION_DATE,
        },
    },
    {
        "name": "2_option_history_iv",
        "path": "/option/history/greeks/implied_volatility",
        "params": {
            "symbol": SYMBOL,
            "expiration": "*",
            "strike": "*",
            "date": TARGET_OPTION_DATE,
        },
    },
    {
        "name": "3_stock_history_eod_root",
        "path": "/hist/stock/eod",
        "params": {
            # リサーチ結果は「Stock History では伝統的に root」と言うが
            # 確証はない。まず root で投げ、別 probe で symbol も試す。
            "root": SYMBOL,
            "start_date": STOCK_START_DATE,
            "end_date": STOCK_END_DATE,
        },
    },
    {
        "name": "3b_stock_history_eod_symbol",
        "path": "/hist/stock/eod",
        "params": {
            # root が 4xx を返した場合の対照群。
            # root / symbol どちらが正しいかを 1 回の実行で確定させる。
            "symbol": SYMBOL,
            "start_date": STOCK_START_DATE,
            "end_date": STOCK_END_DATE,
        },
    },
]

# 標準出力に表示する生テキストの先頭行数（保存ファイルには全文を残す）。
PREVIEW_LINES = 8


# ──────────────────────────────────────────────────────────
# 1 probe の実行
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
        # 接続不可・タイムアウト等。Terminal 未起動ならここに来る。
        print(f"  [NETWORK ERROR] {type(e).__name__}: {e}")
        summary["network_error"] = f"{type(e).__name__}: {e}"
        return summary

    body = response.text
    status = response.status_code
    summary["status_code"] = status
    summary["body_bytes"] = len(body)

    # 生ボディを「全文」ファイルに保存する（パースしない）。
    # 200 でも 4xx でも保存する。4xx のボディには Theta の
    # エラーコード文字列が入っており、それ自体が観察対象。
    out_path = OUTPUT_DIR / f"{probe['name']}__status{status}.csv"
    out_path.write_text(body, encoding="utf-8")
    summary["saved_to"] = str(out_path)

    print(f"  HTTP {status}  ({len(body)} bytes)  -> saved {out_path}")

    # 標準出力には先頭数行だけプレビュー（Actions ログで概観できるように）。
    lines = body.splitlines()
    summary["total_lines"] = len(lines)
    print(f"  total lines: {len(lines)}")
    if lines:
        # CSV の 1 行目はヘッダの可能性が高い。最重要の観察対象。
        print(f"  --- header (line 1) ---")
        print(f"  {lines[0]}")
        if len(lines) > 1:
            print(f"  --- first {min(PREVIEW_LINES, len(lines) - 1)} data rows ---")
            for ln in lines[1 : 1 + PREVIEW_LINES]:
                print(f"  {ln}")
        # ヘッダ文字列を summary にも残す（後で一覧化しやすい）。
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

    print("Step 1A: ThetaData history endpoint raw dump")
    print(f"  base_url: {BASE_URL}")
    print(f"  symbol:   {SYMBOL}")
    print(f"  option history date: {TARGET_OPTION_DATE} "
          f"(expecting 5/12 EOD per the carry-over mapping)")
    print(f"  stock EOD range: {STOCK_START_DATE}..{STOCK_END_DATE} "
          f"(2-day range, on purpose, to read back the date convention)")
    print(f"  run_utc: {datetime.now(timezone.utc).isoformat()}")

    summaries: list[dict] = []
    with httpx.Client(timeout=60.0) as client:
        for probe in PROBES:
            summaries.append(run_probe(client, probe))

    # 全 probe の要約を JSON で 1 ファイルに残す。
    # Actions ログを追わずとも、この 1 ファイルで全体を把握できるように。
    summary_path = OUTPUT_DIR / "_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    print(f"DONE. summary -> {summary_path}")
    print(f"{'=' * 70}")
    print("\n観察してほしいポイント:")
    print("  [1] OI: right 列は CALL/PUT(大) か call/put(小) か")
    print("  [2] IV: underlying_price 列が「ある」か「ない」か ★")
    print("  [3] stock EOD: date 列の実値が 0512 か 0513 か ★★")
    print("      → option history の date と規約が一致するか不一致か")
    print("  [*] いずれかが HTTP 404 ならパス文字列の仮定が外れている")

    # このスクリプトは「観察」が目的。404 や 472 が混ざっていても
    # スクリプト自体は成功とみなし exit 0 を返す。
    # 唯一 exit 1 にするのは、全 probe が network_error だった場合
    # （= Terminal 未起動、観察そのものが成立していない）。
    all_network_failed = all(
        "network_error" in s for s in summaries
    )
    if all_network_failed:
        print("\n[FATAL] 全 probe が通信エラー。Theta Terminal が "
              "起動していない可能性が高い。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
