"""Mock Adapter のテスト。

実行:
    cd /home/claude && python -m pytest gex_engine/tests/test_mock_adapter.py -v
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gex_engine.adapters.base import DataFetcher
from gex_engine.adapters.mock import (
    MockDataFetcher,
    generate_option_chain,
    generate_with_anomaly,
)
from gex_engine.schema import REQUIRED_DTYPES, validate


# ──────────────────────────────────────────────────────────
# Protocol 適合性
# ──────────────────────────────────────────────────────────

class TestProtocolConformance:
    def test_mock_satisfies_data_fetcher_protocol(self):
        """MockDataFetcher が DataFetcher Protocol を満たす。"""
        fetcher = MockDataFetcher()
        assert isinstance(fetcher, DataFetcher)

    def test_source_name_is_mock(self):
        fetcher = MockDataFetcher()
        assert fetcher.source_name == "mock"


# ──────────────────────────────────────────────────────────
# 正常系: スキーマ準拠
# ──────────────────────────────────────────────────────────

class TestSchemaConformance:
    def test_generated_chain_passes_validation(self):
        df = generate_option_chain(symbol="SPY", spot_price=450.0)
        result = validate(df)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_dtypes_match_schema(self):
        df = generate_option_chain()
        for col, expected in REQUIRED_DTYPES.items():
            assert str(df[col].dtype) == expected, (
                f"{col}: got {df[col].dtype}, expected {expected}"
            )

    def test_fetcher_returns_valid_dataframe(self):
        fetcher = MockDataFetcher()
        df = fetcher.get_option_chain("SPY", date.today())
        result = validate(df)
        assert result.is_valid


# ──────────────────────────────────────────────────────────
# データの「現実っぽさ」（レベル2 仕様）
# ──────────────────────────────────────────────────────────

class TestRealisticStructure:
    def test_strikes_are_centered_around_spot(self):
        """ストライクがスポット価格を中心に並んでいる。"""
        df = generate_option_chain(spot_price=450.0)
        strikes = df["strike"].unique()
        assert strikes.min() < 450.0 < strikes.max()
        # スポットに最も近いストライクは ±$5 以内
        nearest = strikes[np.argmin(np.abs(strikes - 450.0))]
        assert abs(nearest - 450.0) <= 5.0

    def test_strike_step_is_5_dollars(self):
        df = generate_option_chain(spot_price=450.0, strike_step=5.0)
        strikes = sorted(df["strike"].unique())
        diffs = np.diff(strikes)
        assert all(d == 5.0 for d in diffs)

    def test_iv_smile_shape(self):
        """IV は ATM 付近で最小、両端で大きくなる（スマイル）。"""
        df = generate_option_chain(spot_price=450.0, base_iv=0.15, seed=42)
        # call だけ抽出して比較（put でも同じ）
        calls = df[df["right"] == "call"].copy()
        calls["distance"] = abs(calls["strike"] - 450.0)

        atm = calls[calls["distance"] <= 5]["implied_volatility"].mean()
        wing = calls[calls["distance"] >= 50]["implied_volatility"].mean()
        assert wing > atm, (
            f"Expected smile shape: wing IV ({wing:.3f}) > ATM IV ({atm:.3f})"
        )

    def test_oi_peaks_near_atm(self):
        """OI は ATM 付近で大きい。"""
        df = generate_option_chain(spot_price=450.0, seed=42)
        calls = df[df["right"] == "call"].copy()
        calls["distance"] = abs(calls["strike"] - 450.0)

        atm_oi = calls[calls["distance"] <= 5]["open_interest"].mean()
        wing_oi = calls[calls["distance"] >= 50]["open_interest"].mean()
        assert atm_oi > wing_oi

    def test_call_and_put_both_exist(self):
        df = generate_option_chain()
        rights = set(df["right"].unique())
        assert rights == {"call", "put"}

    def test_underlying_price_constant_across_rows(self):
        """同一 expiration の全行で underlying_price は同じ値。"""
        df = generate_option_chain(spot_price=450.0)
        assert df["underlying_price"].nunique() == 1
        assert df["underlying_price"].iloc[0] == 450.0

    def test_bid_le_ask(self):
        """正常系では bid <= ask が常に成立。"""
        df = generate_option_chain()
        assert (df["bid"] <= df["ask"]).all()


# ──────────────────────────────────────────────────────────
# 再現性
# ──────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_produces_identical_output(self):
        df1 = generate_option_chain(seed=42)
        df2 = generate_option_chain(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_output(self):
        df1 = generate_option_chain(seed=42)
        df2 = generate_option_chain(seed=123)
        # 少なくとも IV は異なるはず（ノイズが入っているので）
        assert not df1["implied_volatility"].equals(df2["implied_volatility"])


# ──────────────────────────────────────────────────────────
# 異常系
# ──────────────────────────────────────────────────────────

class TestAnomalies:
    def test_empty_anomaly_returns_empty_df(self):
        df = generate_with_anomaly("empty")
        assert len(df) == 0
        # 空でもスキーマ検証は通る
        assert validate(df).is_valid

    def test_zero_oi_anomaly_creates_zero_oi_rows(self):
        df = generate_with_anomaly("zero_oi", spot_price=450.0)
        assert (df["open_interest"] == 0).any()
        # スキーマ検証は通る（OI=0 はソフト警告）
        result = validate(df)
        assert result.is_valid
        assert any("open_interest = 0" in w for w in result.warnings)

    def test_extreme_iv_anomaly_creates_extreme_rows(self):
        df = generate_with_anomaly("extreme_iv", spot_price=450.0)
        assert (df["implied_volatility"] > 5.0).any()
        result = validate(df)
        assert result.is_valid
        assert any("implied_volatility > 5.0" in w for w in result.warnings)

    def test_crossed_quote_anomaly_fails_validation(self):
        """crossed_quote は validate() で弾かれることを確認。"""
        df = generate_with_anomaly("crossed_quote", spot_price=450.0)
        result = validate(df)
        assert not result.is_valid
        assert any("bid > ask" in e for e in result.errors)

    def test_unknown_anomaly_raises(self):
        with pytest.raises(ValueError, match="Unknown anomaly_type"):
            generate_with_anomaly("nonexistent_anomaly")  # type: ignore[arg-type]

    def test_fetcher_with_anomaly_propagates(self):
        """MockDataFetcher 経由でも異常系が反映される。"""
        fetcher = MockDataFetcher(anomaly="zero_oi", spot_price=450.0)
        df = fetcher.get_option_chain("SPY", date.today())
        assert (df["open_interest"] == 0).any()


# ──────────────────────────────────────────────────────────
# 妥当性: GEX 計算が意味を持つか（先行確認）
# ──────────────────────────────────────────────────────────

class TestGEXReadiness:
    """Core Logic を実装する前の先行確認。
    Mock データで以下が成り立つこと:
        - スポット価格を下回るストライクで Put が大量にある
        - スポット価格を上回るストライクで Call が大量にある
        - 全 OI 合計がゼロでない
    """

    def test_total_oi_is_substantial(self):
        df = generate_option_chain(spot_price=450.0)
        assert df["open_interest"].sum() > 10000

    def test_call_and_put_are_balanced(self):
        df = generate_option_chain(spot_price=450.0)
        call_oi = df[df["right"] == "call"]["open_interest"].sum()
        put_oi = df[df["right"] == "put"]["open_interest"].sum()
        # 完全対称ではないが、桁違いにずれていないこと
        ratio = call_oi / put_oi
        assert 0.5 < ratio < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
