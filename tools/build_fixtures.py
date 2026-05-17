"""フィクスチャ3点を実 API ダンプから切り出す（誤判断18 対策）。

入力（実 API ダンプ）:
  4_option_history_greeks_eod__status200.csv  (SPY/2026-06-18/20260512, 43列)
  1_option_history_open_interest__status200.csv (SPY/expiration=*/20260513, 6列)

出力（gex_engine/tests/fixtures/）:
  greeks_eod_normal.csv   greeks/eod 形式。normal + iv_zero + iv_err100 混在
  greeks_eod_partial.csv  上記から K=718 を抜いた片側欠落用
  oi_normal.csv           open_interest 形式。greeks_eod_normal と同一キー

3 ファイルは全て expiration=2026-06-18 / underlying≈738.18 の同一世界。
"""

from __future__ import annotations

import csv
from pathlib import Path

UPLOADS = Path("/mnt/user-data/uploads")
OUT = Path("/home/claude/gex_engine/tests/fixtures")
OUT.mkdir(parents=True, exist_ok=True)

GREEKS_DUMP = UPLOADS / "4_option_history_greeks_eod__status200.csv"
OI_DUMP = UPLOADS / "1_option_history_open_interest__status200.csv"
EXPIRATION = "2026-06-18"

# 選定した共通ストライク集合: (strike, right)
NORMAL_KEYS = [
    (710.0, "CALL"), (710.0, "PUT"),
    (718.0, "CALL"), (718.0, "PUT"),
    (740.0, "CALL"), (740.0, "PUT"),
    (760.0, "CALL"), (760.0, "PUT"),
]
IV_ZERO_KEYS = [
    (335.0, "CALL"),   # OI=9（小）── 通常の集計除外
    (550.0, "CALL"),   # OI=12325（大）── サイレント Wall 欠落検知 WARNING 用
]
IV_ERR100_KEYS = [
    (985.0, "CALL"),
    (990.0, "CALL"),
]
ALL_KEYS = NORMAL_KEYS + IV_ZERO_KEYS + IV_ERR100_KEYS

# iv_partial.csv で greeks/eod 側から抜くキー（OI 側には残す → 片側欠落）
PARTIAL_DROP = {(718.0, "CALL"), (718.0, "PUT")}


def load_greeks() -> tuple[list[str], dict]:
    with open(GREEKS_DUMP, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        idx = {}
        for r in reader:
            idx[(float(r["strike"]), r["right"])] = r
    return header, idx


def load_oi() -> tuple[list[str], dict]:
    with open(OI_DUMP, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        idx = {}
        for r in reader:
            if r["expiration"] == EXPIRATION:
                idx[(float(r["strike"]), r["right"])] = r
    return header, idx


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  書き出し: {path.name}  ({len(rows)} 行)")


def main() -> None:
    g_header, g_idx = load_greeks()
    o_header, o_idx = load_oi()

    # 全キーが両ダンプに存在することを確認（誤判断13/18: 推定で進めない）
    for k in ALL_KEYS:
        assert k in g_idx, f"greeks/eod に {k} が無い"
        assert k in o_idx, f"OI に {k} が無い"

    # greeks_eod_normal.csv: 全12キー、ダンプの行をそのまま採用
    g_rows = [g_idx[k] for k in ALL_KEYS]
    write_csv(OUT / "greeks_eod_normal.csv", g_header, g_rows)

    # greeks_eod_partial.csv: K=718 を抜く（片側欠落）
    g_partial = [g_idx[k] for k in ALL_KEYS if k not in PARTIAL_DROP]
    write_csv(OUT / "greeks_eod_partial.csv", g_header, g_partial)

    # oi_normal.csv: greeks_eod_normal と同一12キー、OI ダンプの実値
    o_rows = [o_idx[k] for k in ALL_KEYS]
    write_csv(OUT / "oi_normal.csv", o_header, o_rows)

    print("\n=== 検証 ===")
    print(f"greeks_eod_normal: {len(g_rows)} 行（normal 8 + iv_zero 2 + iv_err100 2）")
    print(f"greeks_eod_partial: {len(g_partial)} 行（K=718 CALL/PUT を除外）")
    print(f"oi_normal: {len(o_rows)} 行（greeks_eod_normal と同一キー）")


if __name__ == "__main__":
    main()
