"""Step 1A (第2回): ThetaData option history IV エンドポイントの生レスポンス調査。

第1回（dump_history_raw.py）の結果と、その解釈:
    [1] /option/history/open_interest
        → status 200。symbol,expiration,strike,right,timestamp,open_interest
          right は "CALL"/"PUT" 大文字。expiration=* / strike=* OK。
          36 満期すべて取得（13,227 行）。再取得不要。
    [2] /option/history/greeks/implied_volatility
        → status 400。"Cannot specify '*' for the date"
          IV は expiration=* を受け付けない。成功レスポンス未確認。★今回の対象
    [3] /v3/hist/stock/eod (root / symbol 両方)
        → status 410 / 404。/v3/hist/stock/eod は旧 v2 系の遺物。
          v3 の正式パス不明。今回は保留（IV に underlying_price が
          同梱されていれば、そもそも stock EOD が不要になるため、
          IV の確認を先に済ませてから判断する）。

今回の唯一の目的:
    IV history の「成功レスポンスの列構成」を初めて見る。
    特に underlying_price 列が同梱されているか否か（★★★）。
        同梱あり → spot は IV history から取れる。stock EOD 不要。
        同梱なし → stock EOD の v3 正式パスをリサーチ依頼する必要あり。

第1回からの変更点（IV probe のみ）:
    - expiration: "*" → "2026-06-18"
        IV は expiration=* を拒否するため、実在する満期を 1 つ指定。
        2026-06-18 を選んだ理由:
          ・第1回 OI ダンプに存在が確認された満期（実在が確実）
          ・標準的な月次満期
          ・スクリプト実行日（2026-05-16）時点でも未失効
            → 「失効済み満期を history がどう扱うか」という未確認要素を
              持ち込まない。列構成を見るだけの probe にエッジケースは不要。
    - date: "20260513" 据え置き
        OI probe で status 200 が実証された唯一の date。
        目的は列構成の確認であり中身は問わないため、実証済みの date を使う。
        未来日・週末日のエッジケースを今ここで踏みに行かない。
    - strike: "*" 据え置き
        IV の 400 エラーが名指ししたのは expiration のみ。strike=* が
        通るか弾かれるかは未確認。残して試す:
          通る   → IV は満期ループ 36 回で済む
          弾かれる → IV は strike も指定が要る（数千リクエストの設計影響）
        どちらに転んでも設計判断に必須の収穫。

このスクリプトの設計方針（第1回と同一）:
    - gex_engine を import しない（依存ゼロ、httpx と標準ライブラリのみ）。
    - 一切パースしない。生テキストをそのまま保存し、ヘッダと先頭数行のみ表示。
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

# 第1回で status 200 が実証された唯一の date。据え置き。
#   申し送りマッピング: date=20260513 ↔ gex_history["2026.05.12"]（5/12 EOD）
#   第1回 OI ダンプの timestamp 列が 2026-05-13T06:30 で、
#   PC_PIPELINE 1.4 の「毎朝 06:30 ET に前営業日 EOD を報告」と一致 → 規約確定済み。
TARGET_OPTION_DATE = "20260513"

# IV は expiration=* を拒否するため、実在満期を 1 つ指定する。
# 第1回 OI ダンプに存在が確認された満期。実行日(5/16)時点でも未失効。
TARGET_EXPIRATION = "2026-06-18"

SYMBOL = "SPY"

OUTPUT_DIR = Path("./history_dump")

# 今回は IV probe のみ。
#   OI: 第1回で 36 満期取得済み、再取得不要。
#   stock EOD: IV の underlying_price 同梱を確認してから判断（保留）。
PROBES: list[dict] = [
    {
        "name": "2_option_history_iv_RETRY",
        "path": "/option/history/greeks/implied_volatility",
        "params": {
            "symbol": SYMBOL,
            "expiration": TARGET_EXPIRATION,   # 第1回の "*" から変更
            "strike": "*",                     # 据え置き（通るか弾かれるか観察）
            "date": TARGET_OPTION_DATE,
        },
    },
]

# 標準出力に表示する生テキストの先頭行数（保存ファイルには全文を残す）。
PREVIEW_LINES = 8


# ──────────────────────────────────────────────────────────
# 1 probe の実行（第1回と同一ロジック）
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
    # 200 でも 4xx でも保存する。4xx のボディも観察対象。
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

    print("Step 1A (2nd run): ThetaData option history IV raw dump")
    print(f"  base_url:    {BASE_URL}")
    print(f"  symbol:      {SYMBOL}")
    print(f"  date:        {TARGET_OPTION_DATE} (5/12 EOD, proven 200 in 1st run)")
    print(f"  expiration:  {TARGET_EXPIRATION} (real expiry, not yet expired on run date)")
    print(f"  strike:      * (kept, to observe whether IV accepts it)")
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
    print("  [A] status は 200 か。200 でなければ expiration 指定でも")
    print("      まだ何か足りない（エラー本文を読む）。")
    print("  [B] ヘッダ列名に underlying_price が「ある」か「ない」か ★★★")
    print("      ある → spot は IV history から取れる、stock EOD 不要")
    print("      ない → stock EOD の v3 正式パスをリサーチ依頼")
    print("  [C] IV の列名は implied_vol か（snapshot と同じか）")
    print("  [D] strike=* が通ったか（行数が複数 strike 分あるか）")
    print("  [E] right は CALL/PUT 大文字か（snapshot・OI history と同じか）")

    all_network_failed = all("network_error" in s for s in summaries)
    if all_network_failed:
        print("\n[FATAL] 全 probe が通信エラー。Theta Terminal 未起動の疑い。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
