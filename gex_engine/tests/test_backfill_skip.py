"""backfill_history の skip 判定（誤判断32 後の v2 移行）ユニットテスト。

- is_current_pipeline_entry : 通常 run の skip 集合 = v17 ∧ {rest, rest_backfill, rest_backfill_v2}
- is_recomputed_entry       : --force run の skip 集合 = v17 ∧ rest_backfill_v2 のみ

これにより --force は「v2 でない日（stale rest_backfill / rest / 不在）だけ再計算」になり、
分割実行の自動レジュームと、途中状態の stale/fixed 区別（provenance）が成立する。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.backfill_history import (  # noqa: E402
    BACKFILL_DATA_SOURCE,
    is_current_pipeline_entry,
    is_recomputed_entry,
)


def _v17(source):
    return {"data_quality": "ok", "data_source": source, "call_wall": 1.0}


def test_backfill_writes_v2_tag():
    assert BACKFILL_DATA_SOURCE == "rest_backfill_v2"


def test_current_pipeline_set_includes_all_three():
    # 通常 run は rest / rest_backfill / v2 をすべて「済み」とみなす（挙動維持）
    assert is_current_pipeline_entry(_v17("rest")) is True
    assert is_current_pipeline_entry(_v17("rest_backfill")) is True
    assert is_current_pipeline_entry(_v17("rest_backfill_v2")) is True


def test_current_pipeline_rejects_non_pipeline():
    assert is_current_pipeline_entry(_v17("mock")) is False
    assert is_current_pipeline_entry(_v17("unknown")) is False
    assert is_current_pipeline_entry({"data_source": "rest"}) is False  # data_quality 欠 = pre-v17
    assert is_current_pipeline_entry(None) is False
    assert is_current_pipeline_entry("regime") is False


def test_recomputed_set_is_v2_only():
    # force run は v2 のみ skip ＝ stale rest_backfill / rest は再計算対象
    assert is_recomputed_entry(_v17("rest_backfill_v2")) is True
    assert is_recomputed_entry(_v17("rest_backfill")) is False   # stale → 再計算
    assert is_recomputed_entry(_v17("rest")) is False            # cron（修正前後不明）→ 再計算
    assert is_recomputed_entry({"data_source": "rest_backfill_v2"}) is False  # data_quality 欠
    assert is_recomputed_entry(None) is False


def test_force_recomputes_stale_but_skips_v2():
    """force skip 集合 ⊊ 通常 skip 集合。stale 日は force で再計算、v2 日は skip。"""
    stale = _v17("rest_backfill")
    fixed = _v17("rest_backfill_v2")
    # 通常: 両方 skip
    assert is_current_pipeline_entry(stale) and is_current_pipeline_entry(fixed)
    # force: stale は再計算（skip しない）、fixed のみ skip
    assert is_recomputed_entry(stale) is False
    assert is_recomputed_entry(fixed) is True
