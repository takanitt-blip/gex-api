"""migrate_zposition.py ─ 既存 gex_history.json を誤判断32 仕様へ移行（使い捨て）。

何をするか（ThetaData は叩かない・保存値だけで完結）:
  1. data_quality == "anomaly" のエントリ（Z∉[P,C]）を "ok" に変え、anomaly_detail を除去。
     ← Z と Wall の位置関係は品質欠陥ではなく regime 構造（PR3 で _assess_data_quality
       から除外済み）。当日満期除外（PR1/PR2）で大半が解消し、残った非整序も崩壊ではない。
  2. 全 v17 エントリに z_position（"inside"/"above_call"/"below_put"/None）を付与。
     ← 本番 serializer と同じ derive_z_position を再利用（独立再実装しない＝誤判断26）。
     保存済み C/Z/P は丸め済みなので、serializer（丸め値から派生）と一致する。
  3. data_quality == "data_error" は保全（触らない）。OLD（v17 でない）エントリは skip。

使い方（リポジトリ直下）:
  python tools/migrate_zposition.py            # dry-run（変更内容を表示するだけ）
  python tools/migrate_zposition.py --apply    # .bak を作って書き込み
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gex_engine.core.gex import derive_z_position  # noqa: E402

HISTORY = pathlib.Path(__file__).resolve().parents[1] / "gex_history.json"


def is_v17(entry: object) -> bool:
    return isinstance(entry, dict) and "data_quality" in entry


def _insert_z_position(entry: dict, zp) -> dict:
    """z_position を underlying_price の直後に挿入した新 dict を返す。

    go-forward の serializer（...underlying_price, z_position, total_gex...）と
    キー順を一致させ、履歴と新規 cron エントリの形を揃える。
    """
    out: dict = {}
    for k, v in entry.items():
        if k == "z_position":
            continue  # 既存があれば正位置へ入れ直す
        out[k] = v
        if k == "underlying_price":
            out["z_position"] = zp
    if "z_position" not in out:  # underlying_price 欠の異常系は末尾
        out["z_position"] = zp
    return out


def migrate(history: dict) -> tuple[dict, dict]:
    """history を移行し、サマリを返す。"""
    summary = {
        "total": len(history),
        "v17": 0,
        "anomaly_to_ok": 0,
        "data_error_kept": 0,
        "z_position": collections.Counter(),
        "samples": [],
    }
    for key, entry in history.items():
        if not is_v17(entry):
            continue
        summary["v17"] += 1
        dq = entry.get("data_quality")

        if dq == "anomaly":
            entry["data_quality"] = "ok"
            entry.pop("anomaly_detail", None)
            summary["anomaly_to_ok"] += 1
            if len(summary["samples"]) < 8:
                summary["samples"].append(key)
        elif dq == "data_error":
            summary["data_error_kept"] += 1  # 保全（触らない）

        zp = derive_z_position(
            entry.get("call_wall"), entry.get("put_wall"), entry.get("zero_gamma")
        )
        history[key] = _insert_z_position(entry, zp)
        summary["z_position"][zp] += 1

    return history, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="書き込む（既定は dry-run）")
    args = ap.parse_args()

    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    _, s = migrate(history)

    print(f"total keys      : {s['total']}")
    print(f"v17 entries     : {s['v17']}")
    print(f"anomaly -> ok   : {s['anomaly_to_ok']}")
    print(f"data_error kept : {s['data_error_kept']}")
    print(f"z_position dist : {dict(s['z_position'])}")
    if s["samples"]:
        print(f"anomaly->ok 例  : {s['samples']}")

    if not args.apply:
        print("\n[dry-run] 変更は書き込んでいません。--apply で .bak を作って書き込みます。")
        return

    backup = HISTORY.with_suffix(".json.bak")
    shutil.copy2(HISTORY, backup)
    HISTORY.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[applied] 書き込み完了。バックアップ: {backup.name}")


if __name__ == "__main__":
    main()
