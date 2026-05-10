"""
gex_engine.io_layer の包括的なユニットテスト

テスト方針:
  - 純粋関数（serialize, scale, make_date_key 等）は単独でテスト
  - 履歴マージは新旧の組み合わせを網羅
  - atomic write は実ファイルで検証
  - 統合テストで Facade の挙動を確認
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

from gex_engine.io_layer import (
    save_gex_result,
    serialize_result,
    load_history,
    merge_entry,
    trim_history,
    write_json_atomic,
    make_date_key,
    make_timestamp,
    scale_total_gex,
)
from gex_engine.io_layer.history import _values_differ_meaningfully


# ============================================================
# テスト用のダミー GEXResult（実 GEXResult と同じフィールド構成）
# ============================================================
@dataclass
class FakeGEXResult:
    """gex_engine.core.result.GEXResult と同じフィールド構成のダミー。

    実 GEXResult は frozen=True だが、テスト用に書き換え可能にしておく。
    """
    symbol: str
    as_of: str
    underlying_price: float
    call_wall: float
    put_wall: float
    zero_gamma: Optional[float]
    max_pain: float                  # 実 GEXResult では Optional ではない
    total_gex: float
    n_contracts_used: int
    data_source: str
    # serialize_result が見るオプションフィールド（実 GEXResult にはない）
    regime: Optional[str] = None
    regime_text: Optional[str] = None


# ============================================================
# scale_total_gex
# ============================================================
class TestScaleTotalGex(unittest.TestCase):
    def test_basic_scaling(self):
        # raw=100, S=450 → 100 * 450^2 * 0.01 = 100 * 202500 * 0.01 = 202500
        self.assertEqual(scale_total_gex(100.0, 450.0), 202500.0)

    def test_zero_raw(self):
        self.assertEqual(scale_total_gex(0.0, 450.0), 0.0)

    def test_negative_raw_preserves_sign(self):
        # 符号が保たれることを確認（レジーム判定に必須）
        self.assertEqual(scale_total_gex(-100.0, 450.0), -202500.0)

    def test_typical_spy_scale(self):
        # SPY 典型値: raw_total_gex ≈ 数千〜数万、S ≈ 450
        # → スケール後は数億〜数十億のオーダー
        result = scale_total_gex(1000.0, 450.0)
        self.assertGreater(result, 1_000_000)
        self.assertLess(result, 100_000_000)


# ============================================================
# make_date_key（タイムゾーン処理）
# ============================================================
class TestMakeDateKey(unittest.TestCase):
    def test_et_basis_during_eastern_daylight(self):
        # 2026-05-09 22:30 UTC = 2026-05-09 18:30 ET (EDT, UTC-4)
        # → ET 基準で "2026.05.09"
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(make_date_key(utc), "2026.05.09")

    def test_et_basis_during_eastern_standard(self):
        # 2026-01-09 22:30 UTC = 2026-01-09 17:30 ET (EST, UTC-5)
        # → ET 基準で "2026.01.09"
        utc = datetime(2026, 1, 9, 22, 30, 0, tzinfo=timezone.utc)
        self.assertEqual(make_date_key(utc), "2026.01.09")

    def test_jst_morning_still_et_yesterday(self):
        # JST 翌日 07:30 = UTC 22:30 = ET 同日 17:30/18:30
        # → 日付は ET 基準で前日のまま（v11 の cron 想定通り）
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        # JST: 2026-05-10 07:30 だが、ET 基準で "2026.05.09"
        self.assertEqual(make_date_key(utc), "2026.05.09")

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2026, 5, 9, 22, 30, 0)  # tzinfo なし
        self.assertEqual(make_date_key(naive), "2026.05.09")

    def test_dot_separator_format(self):
        # EA 互換のため、ドット区切りでなければならない
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        key = make_date_key(utc)
        self.assertEqual(len(key), 10)
        self.assertEqual(key[4], ".")
        self.assertEqual(key[7], ".")

    def test_midnight_et_boundary(self):
        # ET 0:00 ちょうど（前日 5/9 の cron 後の境界）
        # 2026-05-10 04:00 UTC = 2026-05-10 00:00 ET (EDT)
        utc = datetime(2026, 5, 10, 4, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(make_date_key(utc), "2026.05.10")

        # その 1 分前（ET 23:59 = UTC 03:59）はまだ前日
        utc = datetime(2026, 5, 10, 3, 59, 0, tzinfo=timezone.utc)
        self.assertEqual(make_date_key(utc), "2026.05.09")


# ============================================================
# make_timestamp
# ============================================================
class TestMakeTimestamp(unittest.TestCase):
    def test_iso_8601_z_suffix(self):
        utc = datetime(2026, 5, 9, 22, 30, 15, tzinfo=timezone.utc)
        self.assertEqual(make_timestamp(utc), "2026-05-09T22:30:15Z")

    def test_microseconds_dropped(self):
        utc = datetime(2026, 5, 9, 22, 30, 15, 123456, tzinfo=timezone.utc)
        self.assertEqual(make_timestamp(utc), "2026-05-09T22:30:15Z")

    def test_naive_treated_as_utc(self):
        naive = datetime(2026, 5, 9, 22, 30, 15)
        self.assertEqual(make_timestamp(naive), "2026-05-09T22:30:15Z")


# ============================================================
# serialize_result
# ============================================================
class TestSerializeResult(unittest.TestCase):
    def _make_fake_result(self, **overrides):
        defaults = dict(
            symbol="SPY",
            as_of="2026-05-09T00:00:00",
            underlying_price=450.00,
            call_wall=465.00,
            put_wall=435.00,
            zero_gamma=441.69,
            max_pain=450.00,
            total_gex=6421.0,  # 素の単位
            n_contracts_used=12345,
            data_source="mock",
        )
        defaults.update(overrides)
        return FakeGEXResult(**defaults)

    def test_basic_dataclass_input(self):
        result = self._make_fake_result()
        utc = datetime(2026, 5, 9, 22, 30, 15, tzinfo=timezone.utc)
        out = serialize_result(result, now_utc=utc)

        # 価格水準（小数 2 桁）
        self.assertEqual(out["call_wall"], 465.00)
        self.assertEqual(out["put_wall"], 435.00)
        self.assertEqual(out["zero_gamma"], 441.69)
        self.assertEqual(out["max_pain"], 450.00)
        self.assertEqual(out["underlying_price"], 450.00)

        # スケール変換: 6421 * 450^2 * 0.01 = 6421 * 2025 = 13_002_525
        self.assertEqual(out["total_gex"], 13002525)

        # メタ
        self.assertEqual(out["data_source"], "mock")  # GEXResult 由来
        self.assertEqual(out["timestamp"], "2026-05-09T22:30:15Z")

        # 自動導出されたレジーム
        self.assertEqual(out["regime"], "range")
        self.assertIn("レンジ", out["regime_text"])

        # 分析用フィールド
        self.assertEqual(out["symbol"], "SPY")
        self.assertEqual(out["as_of"], "2026-05-09T00:00:00")
        self.assertEqual(out["n_contracts_used"], 12345)

    def test_dict_input(self):
        # dataclass 以外でも動くこと
        result_dict = {
            "symbol": "SPY",
            "as_of": "2026-05-09T00:00:00",
            "call_wall": 465.0,
            "put_wall": 435.0,
            "zero_gamma": 441.69,
            "max_pain": 450.0,
            "underlying_price": 450.0,
            "total_gex": 6421.0,
            "n_contracts_used": 12345,
            "data_source": "rest",
        }
        out = serialize_result(result_dict)
        self.assertEqual(out["call_wall"], 465.0)
        # data_source は dict から取れる
        self.assertEqual(out["data_source"], "rest")

    def test_zero_gamma_none_preserved(self):
        # Zero Gamma 解なし時の挙動（v11 仕様: None を返す）
        result = self._make_fake_result(zero_gamma=None)
        out = serialize_result(result)
        self.assertIsNone(out["zero_gamma"])

        # JSON シリアライズ時に null になることを確認
        json_str = json.dumps(out)
        self.assertIn('"zero_gamma": null', json_str)

    def test_max_pain_none_handled_defensively(self):
        # 実 GEXResult では max_pain は必ず float（None 不可）だが、
        # dict 経路で None を渡された場合に I/O 層が壊れないこと（防御）
        result_dict = {
            "symbol": "SPY",
            "as_of": "2026-05-09T00:00:00",
            "call_wall": 465.0,
            "put_wall": 435.0,
            "zero_gamma": 441.69,
            "max_pain": None,
            "underlying_price": 450.0,
            "total_gex": 6421.0,
            "n_contracts_used": 1000,
            "data_source": "mock",
        }
        out = serialize_result(result_dict)
        self.assertIsNone(out["max_pain"])

    def test_nan_normalized_to_none(self):
        # NaN は null に正規化
        result = self._make_fake_result(zero_gamma=float("nan"))
        out = serialize_result(result)
        self.assertIsNone(out["zero_gamma"])

    def test_inf_normalized_to_none(self):
        result = self._make_fake_result(zero_gamma=float("inf"))
        out = serialize_result(result)
        self.assertIsNone(out["zero_gamma"])

    def test_negative_total_gex_yields_trend_regime(self):
        result = self._make_fake_result(total_gex=-6421.0)
        out = serialize_result(result)
        self.assertEqual(out["regime"], "trend")
        self.assertIn("トレンド", out["regime_text"])

    def test_explicit_regime_overrides_derived(self):
        # GEXResult が regime を持っていればそれを尊重
        result = self._make_fake_result()
        result.regime = "custom"
        result.regime_text = "カスタムレジーム"
        out = serialize_result(result)
        self.assertEqual(out["regime"], "custom")
        self.assertEqual(out["regime_text"], "カスタムレジーム")

    def test_missing_required_fields_raises(self):
        with self.assertRaises(ValueError):
            serialize_result({"call_wall": 465.0})  # spot, total_gex 欠落

    def test_data_source_from_gex_result(self):
        # data_source 引数なし → GEXResult.data_source が使われる
        result = self._make_fake_result(data_source="rest")
        out = serialize_result(result)
        self.assertEqual(out["data_source"], "rest")

    def test_data_source_argument_overrides(self):
        # data_source 引数あり → 引数が優先（override）
        result = self._make_fake_result(data_source="mock")
        out = serialize_result(result, data_source="sdk")
        self.assertEqual(out["data_source"], "sdk")

    def test_data_source_fallback_to_unknown(self):
        # GEXResult が data_source を持たず、引数もない → "unknown"
        result_dict = {
            "symbol": "SPY",
            "as_of": "2026-05-09T00:00:00",
            "call_wall": 465.0,
            "put_wall": 435.0,
            "zero_gamma": 441.69,
            "max_pain": 450.0,
            "underlying_price": 450.0,
            "total_gex": 6421.0,
            "n_contracts_used": 100,
            # data_source なし
        }
        out = serialize_result(result_dict)
        self.assertEqual(out["data_source"], "unknown")

    def test_total_gex_rounded_to_integer(self):
        # 整数（小数 0 桁）に丸められること
        result = self._make_fake_result(total_gex=6421.789)
        out = serialize_result(result)
        # 6421.789 * 450^2 * 0.01 = 13_004_122.725 → 13_004_123
        self.assertEqual(out["total_gex"], 13004123)
        self.assertIsInstance(out["total_gex"], int)

    def test_output_is_json_serializable(self):
        result = self._make_fake_result(zero_gamma=None)
        out = serialize_result(result)
        # 例外が出ないこと（NaN/Inf が混入していたら json.dumps で死ぬ）
        json.dumps(out)


# ============================================================
# load_history
# ============================================================
class TestLoadHistory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "history.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_file_returns_empty_dict(self):
        self.assertEqual(load_history(self.path), {})

    def test_empty_file_returns_empty_dict(self):
        with open(self.path, "w") as f:
            pass
        self.assertEqual(load_history(self.path), {})

    def test_corrupted_json_returns_empty_dict(self):
        with open(self.path, "w") as f:
            f.write("{this is not valid json")
        self.assertEqual(load_history(self.path), {})

    def test_non_dict_top_level_returns_empty(self):
        with open(self.path, "w") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(load_history(self.path), {})

    def test_valid_history_loaded(self):
        data = {
            "2026.05.09": {"call_wall": 465.0, "put_wall": 435.0},
            "2026.05.08": {"call_wall": 463.0, "put_wall": 433.0},
        }
        with open(self.path, "w") as f:
            json.dump(data, f)
        loaded = load_history(self.path)
        self.assertEqual(loaded, data)


# ============================================================
# merge_entry
# ============================================================
class TestMergeEntry(unittest.TestCase):
    def _make_entry(self, call_wall=465.0):
        return {
            "call_wall": call_wall,
            "put_wall": 435.0,
            "zero_gamma": 441.69,
            "max_pain": 450.0,
            "underlying_price": 450.0,
            "total_gex": 13002525,
            "regime": "range",
            "regime_text": "レンジ相場・低ボラティリティ",
            "timestamp": "2026-05-09T22:30:15Z",
            "data_source": "mock",
        }

    def test_new_date_added(self):
        history = {}
        entry = self._make_entry()
        new_history, warning = merge_entry(history, "2026.05.09", entry)

        self.assertIn("2026.05.09", new_history)
        self.assertIsNone(warning)

    def test_same_date_same_values_no_warning(self):
        # 同日、同じ計算値（timestamp だけ違う）→ 警告なし
        old = self._make_entry()
        new = self._make_entry()
        new["timestamp"] = "2026-05-09T23:00:00Z"  # 違うタイムスタンプ

        history = {"2026.05.09": old}
        new_history, warning = merge_entry(history, "2026.05.09", new)

        self.assertIsNone(warning)
        # 上書きはされる（timestamp が新しいので）
        self.assertEqual(new_history["2026.05.09"]["timestamp"], "2026-05-09T23:00:00Z")

    def test_same_date_different_values_yields_warning(self):
        # 同日、違う計算値 → 警告
        old = self._make_entry(call_wall=465.0)
        new = self._make_entry(call_wall=470.0)

        history = {"2026.05.09": old}
        new_history, warning = merge_entry(history, "2026.05.09", new)

        self.assertIsNotNone(warning)
        self.assertIn("2026.05.09", warning)
        self.assertIn("call_wall=465", warning)
        self.assertIn("call_wall=470", warning)

        # 上書きされる
        self.assertEqual(new_history["2026.05.09"]["call_wall"], 470.0)

    def test_existing_history_preserved(self):
        # 別日のデータには触らない
        old_entry = self._make_entry(call_wall=460.0)
        new_entry = self._make_entry(call_wall=465.0)

        history = {"2026.05.08": old_entry}
        new_history, warning = merge_entry(history, "2026.05.09", new_entry)

        self.assertIn("2026.05.08", new_history)
        self.assertIn("2026.05.09", new_history)
        self.assertEqual(new_history["2026.05.08"]["call_wall"], 460.0)
        self.assertEqual(new_history["2026.05.09"]["call_wall"], 465.0)
        self.assertIsNone(warning)

    def test_original_history_not_mutated(self):
        # 入力 history が変更されないこと（イミュータブル性）
        history = {"2026.05.08": self._make_entry()}
        original_keys = set(history.keys())

        new_history, _ = merge_entry(history, "2026.05.09", self._make_entry())

        self.assertEqual(set(history.keys()), original_keys)
        self.assertNotIn("2026.05.09", history)


class TestValuesDifferMeaningfully(unittest.TestCase):
    def test_identical_dicts(self):
        a = {"call_wall": 465.0, "timestamp": "T1"}
        b = {"call_wall": 465.0, "timestamp": "T1"}
        self.assertFalse(_values_differ_meaningfully(a, b))

    def test_only_timestamp_differs(self):
        a = {"call_wall": 465.0, "timestamp": "T1", "data_source": "mock"}
        b = {"call_wall": 465.0, "timestamp": "T2", "data_source": "mock"}
        self.assertFalse(_values_differ_meaningfully(a, b))

    def test_only_data_source_differs(self):
        a = {"call_wall": 465.0, "timestamp": "T1", "data_source": "mock"}
        b = {"call_wall": 465.0, "timestamp": "T1", "data_source": "rest"}
        self.assertFalse(_values_differ_meaningfully(a, b))

    def test_calculation_value_differs(self):
        a = {"call_wall": 465.0, "timestamp": "T1"}
        b = {"call_wall": 470.0, "timestamp": "T1"}
        self.assertTrue(_values_differ_meaningfully(a, b))


# ============================================================
# trim_history
# ============================================================
class TestTrimHistory(unittest.TestCase):
    def test_no_limit_keeps_all(self):
        history = {f"2026.05.{i:02d}": {} for i in range(1, 31)}
        trimmed = trim_history(history, max_entries=None)
        self.assertEqual(len(trimmed), 30)

    def test_limit_keeps_latest_n(self):
        history = {f"2026.05.{i:02d}": {} for i in range(1, 31)}
        trimmed = trim_history(history, max_entries=10)
        self.assertEqual(len(trimmed), 10)
        # 最新 10 日（5月21日〜30日）が残る
        self.assertIn("2026.05.30", trimmed)
        self.assertIn("2026.05.21", trimmed)
        self.assertNotIn("2026.05.20", trimmed)

    def test_limit_larger_than_history_keeps_all(self):
        history = {f"2026.05.{i:02d}": {} for i in range(1, 5)}
        trimmed = trim_history(history, max_entries=100)
        self.assertEqual(len(trimmed), 4)


# ============================================================
# write_json_atomic
# ============================================================
class TestWriteJsonAtomic(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "out.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_basic_write(self):
        data = {"foo": "bar", "num": 42}
        write_json_atomic(self.path, data)

        with open(self.path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

    def test_japanese_preserved(self):
        data = {"regime_text": "レンジ相場・低ボラティリティ"}
        write_json_atomic(self.path, data)

        with open(self.path, encoding="utf-8") as f:
            content = f.read()
        # ensure_ascii=False なので日本語がそのまま
        self.assertIn("レンジ相場", content)

    def test_overwrite_existing(self):
        write_json_atomic(self.path, {"v": 1})
        write_json_atomic(self.path, {"v": 2})

        with open(self.path) as f:
            self.assertEqual(json.load(f), {"v": 2})

    def test_no_temp_file_left_behind_on_success(self):
        write_json_atomic(self.path, {"v": 1})
        # 一時ファイル（.tmp_xxx.json）が残っていないこと
        leftover = [f for f in os.listdir(self.tmpdir) if f.startswith(".tmp_")]
        self.assertEqual(leftover, [])

    def test_no_temp_file_left_behind_on_failure(self):
        # シリアライズ不可なオブジェクト
        class NotSerializable:
            pass

        with self.assertRaises(TypeError):
            write_json_atomic(self.path, {"obj": NotSerializable()})

        # 一時ファイルが掃除されていること
        leftover = [f for f in os.listdir(self.tmpdir) if f.startswith(".tmp_")]
        self.assertEqual(leftover, [])

    def test_nonexistent_directory_raises(self):
        bad_path = os.path.join(self.tmpdir, "nonexistent", "out.json")
        with self.assertRaises(OSError):
            write_json_atomic(bad_path, {"v": 1})


# ============================================================
# 統合テスト: save_gex_result（Facade）
# ============================================================
class TestSaveGexResultIntegration(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "gex_history.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_result(self, **overrides):
        defaults = dict(
            symbol="SPY",
            as_of="2026-05-09T00:00:00",
            underlying_price=450.00,
            call_wall=465.00,
            put_wall=435.00,
            zero_gamma=441.69,
            max_pain=450.00,
            total_gex=6421.0,
            n_contracts_used=12345,
            data_source="mock",
        )
        defaults.update(overrides)
        return FakeGEXResult(**defaults)

    def test_first_save_creates_file(self):
        result = self._make_result()
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)

        entry = save_gex_result(
            result, path=self.path, data_source="mock", now_utc=utc
        )

        self.assertTrue(os.path.exists(self.path))

        # ファイル内容を確認
        with open(self.path, encoding="utf-8") as f:
            history = json.load(f)
        self.assertEqual(set(history.keys()), {"2026.05.09"})
        self.assertEqual(history["2026.05.09"]["call_wall"], 465.00)
        self.assertEqual(history["2026.05.09"]["data_source"], "mock")

    def test_second_save_appends(self):
        # 1日目
        utc1 = datetime(2026, 5, 8, 22, 30, 0, tzinfo=timezone.utc)
        save_gex_result(
            self._make_result(call_wall=460.0),
            path=self.path, now_utc=utc1,
        )

        # 2日目
        utc2 = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        save_gex_result(
            self._make_result(call_wall=465.0),
            path=self.path, now_utc=utc2,
        )

        with open(self.path, encoding="utf-8") as f:
            history = json.load(f)

        self.assertEqual(set(history.keys()), {"2026.05.08", "2026.05.09"})
        self.assertEqual(history["2026.05.08"]["call_wall"], 460.0)
        self.assertEqual(history["2026.05.09"]["call_wall"], 465.0)

    def test_same_day_overwrite(self):
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)

        save_gex_result(self._make_result(call_wall=465.0),
                        path=self.path, now_utc=utc)
        save_gex_result(self._make_result(call_wall=470.0),
                        path=self.path, now_utc=utc)

        with open(self.path, encoding="utf-8") as f:
            history = json.load(f)

        # 上書きされている
        self.assertEqual(history["2026.05.09"]["call_wall"], 470.0)
        # キーは 1 つだけ
        self.assertEqual(len(history), 1)

    def test_corrupted_history_recovered(self):
        # 既存ファイルが破損していても続行できる
        with open(self.path, "w") as f:
            f.write("{garbage not json")

        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        save_gex_result(self._make_result(), path=self.path, now_utc=utc)

        with open(self.path, encoding="utf-8") as f:
            history = json.load(f)
        self.assertIn("2026.05.09", history)

    def test_max_entries_trimming(self):
        # 5 日分を保存し、max_entries=3 で切り詰め
        for day in range(5, 10):
            utc = datetime(2026, 5, day, 22, 30, 0, tzinfo=timezone.utc)
            save_gex_result(
                self._make_result(),
                path=self.path,
                now_utc=utc,
                max_entries=3,
            )

        with open(self.path, encoding="utf-8") as f:
            history = json.load(f)

        self.assertEqual(len(history), 3)
        self.assertIn("2026.05.07", history)
        self.assertIn("2026.05.08", history)
        self.assertIn("2026.05.09", history)
        self.assertNotIn("2026.05.05", history)
        self.assertNotIn("2026.05.06", history)

    def test_ea_compatible_json_format(self):
        """
        EA (Gex_visualizer.mq5) が期待するフォーマットを満たすこと:
          - トップレベルが dict
          - 日付キーが "YYYY.MM.DD" 形式
          - call_wall, put_wall, zero_gamma が数値で読める
        """
        utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
        save_gex_result(self._make_result(), path=self.path, now_utc=utc)

        with open(self.path, encoding="utf-8") as f:
            content = f.read()

        # 日付キーがドット区切りで含まれている
        self.assertIn('"2026.05.09"', content)

        # EA が読む 3 フィールドが含まれている
        self.assertIn('"call_wall":', content)
        self.assertIn('"put_wall":', content)
        self.assertIn('"zero_gamma":', content)

        # 追加フィールド（max_pain 等）も含まれている
        self.assertIn('"max_pain":', content)
        self.assertIn('"total_gex":', content)
        self.assertIn('"timestamp":', content)
        self.assertIn('"data_source":', content)


if __name__ == "__main__":
    unittest.main()
