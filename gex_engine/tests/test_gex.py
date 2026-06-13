"""GEX 計算ロジックのテスト。"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from gex_engine.adapters.mock import MockDataFetcher, generate_option_chain
from gex_engine.core.gex import (
    calculate_all,
    calculate_gex_per_strike,
    find_call_wall,
    find_max_pain,
    find_put_wall,
    find_zero_gamma,
    _assess_data_quality,     
    derive_z_position,
    _find_call_wall_opt,      
    _find_put_wall_opt,
)
from gex_engine.core.result import GEXResult


# ──────────────────────────────────────────────────────────
# Per-Strike GEX
# ──────────────────────────────────────────────────────────

class TestGEXPerStrike:
    def test_returns_series_indexed_by_strike(self):
        df = generate_option_chain(spot_price=450.0)
        result = calculate_gex_per_strike(df, as_of=date.today())
        assert isinstance(result, pd.Series)
        assert result.index.name == "strike"

    def test_call_only_has_positive_gex(self):
        """Call だけのチェーンは Net GEX が全て正。"""
        df = generate_option_chain(spot_price=450.0)
        df_calls = df[df["right"] == "call"].copy()
        result = calculate_gex_per_strike(df_calls, as_of=date.today())
        assert (result >= 0).all()

    def test_put_only_has_negative_gex(self):
        """Put だけのチェーンは Net GEX が全て非正（負またはゼロ）。"""
        df = generate_option_chain(spot_price=450.0)
        df_puts = df[df["right"] == "put"].copy()
        result = calculate_gex_per_strike(df_puts, as_of=date.today())
        assert (result <= 0).all()

    def test_empty_dataframe_returns_empty_series(self):
        df = generate_option_chain().iloc[0:0]
        result = calculate_gex_per_strike(df, as_of=date.today())
        assert len(result) == 0


# ──────────────────────────────────────────────────────────
# Call Wall / Put Wall
# ──────────────────────────────────────────────────────────

class TestWalls:
    def test_call_wall_above_spot(self):
        df = generate_option_chain(spot_price=450.0)
        gex = calculate_gex_per_strike(df, as_of=date.today())
        cw = find_call_wall(gex, spot=450.0)
        assert cw >= 450.0

    def test_put_wall_below_spot(self):
        df = generate_option_chain(spot_price=450.0)
        gex = calculate_gex_per_strike(df, as_of=date.today())
        pw = find_put_wall(gex, spot=450.0)
        assert pw <= 450.0

    def test_no_positive_gex_falls_back_to_spot(self):
        """正の GEX が無いとき Call Wall は spot にフォールバック。"""
        # 全部 negative にする（Put のみ）
        gex = pd.Series(
            [-100.0, -200.0, -150.0],
            index=pd.Index([440.0, 450.0, 460.0], name="strike"),
            name="net_gex",
        )
        cw = find_call_wall(gex, spot=450.0)
        assert cw == 450.0


# ──────────────────────────────────────────────────────────
# Zero Gamma
# ──────────────────────────────────────────────────────────

class TestZeroGamma:
    def test_zero_gamma_within_strike_range(self):
        """Zero Gamma が見つかった場合、ストライク範囲内にある。"""
        df = generate_option_chain(spot_price=450.0)
        zg = find_zero_gamma(df, as_of=date.today())
        if zg is not None:
            k_min = df["strike"].min()
            k_max = df["strike"].max()
            assert k_min <= zg <= k_max

    def test_zero_gamma_satisfies_definition(self):
        """Zero Gamma で Net Gamma が実際にゼロ近傍。"""
        df = generate_option_chain(spot_price=450.0)
        zg = find_zero_gamma(df, as_of=date.today())
        if zg is not None:
            net_at_zg = calculate_gex_per_strike(
                df, as_of=date.today(), spot_override=zg
            ).sum()
            # スケールが大きいので相対誤差で見る
            total_abs = calculate_gex_per_strike(
                df, as_of=date.today(), spot_override=zg
            ).abs().sum()
            assert abs(net_at_zg) / total_abs < 0.01

    def test_realistic_mock_finds_zero_gamma(self):
        """現実的な Mock データで Zero Gamma が必ず見つかる。
        
        Mock は Call OI を spot+1%, Put OI を spot-1% にピークを持たせており、
        Net Gamma が spot 上下で符号反転するように設計されている。
        Zero Gamma が None になるのは設計バグのサイン。
        """
        df = generate_option_chain(spot_price=450.0, seed=42)
        zg = find_zero_gamma(df, as_of=date.today())
        assert zg is not None, (
            "Zero Gamma was not found in realistic Mock data. "
            "This indicates the Mock OI distribution doesn't cross zero."
        )
        # spot ± 10% の範囲内に収まることも確認
        assert 405.0 <= zg <= 495.0

    def test_realistic_mock_levels_relationship(self):
        """正常系で Put Wall <= Zero Gamma <= Call Wall が成立する。"""
        from gex_engine.core.gex import calculate_all
        df = generate_option_chain(spot_price=450.0, seed=42)
        result = calculate_all(df, as_of=date.today())
        assert result.zero_gamma is not None
        assert result.put_wall <= result.zero_gamma <= result.call_wall

    def test_empty_dataframe_returns_none(self):
        df = generate_option_chain().iloc[0:0]
        zg = find_zero_gamma(df, as_of=date.today())
        assert zg is None


# ──────────────────────────────────────────────────────────
# Max Pain
# ──────────────────────────────────────────────────────────

class TestMaxPain:
    def test_max_pain_in_strike_range(self):
        df = generate_option_chain(spot_price=450.0)
        mp = find_max_pain(df)
        assert df["strike"].min() <= mp <= df["strike"].max()

    def test_max_pain_with_synthetic_data(self):
        """OI 集中時の Max Pain が直感的な位置に来る。
        450 付近に大量の OI を集めれば Max Pain は 450 近辺。
        Mock のデフォルトは ATM 集中なので、Max Pain ≈ spot のはず。
        """
        df = generate_option_chain(spot_price=450.0, seed=42)
        mp = find_max_pain(df)
        # ATM 集中なので spot ± $20 以内には来るはず
        assert abs(mp - 450.0) <= 20.0

    def test_empty_returns_nan(self):
        df = generate_option_chain().iloc[0:0]
        mp = find_max_pain(df)
        assert np.isnan(mp)


# ──────────────────────────────────────────────────────────
# 統合: calculate_all
# ──────────────────────────────────────────────────────────

class TestCalculateAll:
    def test_returns_gex_result(self):
        df = generate_option_chain(spot_price=450.0)
        result = calculate_all(df, as_of=date.today(), data_source="mock")
        assert isinstance(result, GEXResult)

    def test_result_is_serializable(self):
        df = generate_option_chain(spot_price=450.0)
        result = calculate_all(df, as_of=date.today(), data_source="mock")
        d = result.to_dict()
        # 主要フィールドが入っている
        assert "call_wall" in d
        assert "put_wall" in d
        assert "zero_gamma" in d
        assert "max_pain" in d

    def test_levels_relationship(self):
        """正常系では Put Wall <= Zero Gamma <= Call Wall が期待される。
        ただし Mock では完全には保証されないため、
        各値が妥当な範囲内にあることだけ確認。
        """
        df = generate_option_chain(spot_price=450.0, seed=42)
        result = calculate_all(df, as_of=date.today())

        assert result.call_wall >= result.underlying_price
        assert result.put_wall <= result.underlying_price
        if result.zero_gamma is not None:
            assert result.put_wall <= result.zero_gamma <= result.call_wall

    def test_empty_dataframe_raises(self):
        df = generate_option_chain().iloc[0:0]
        with pytest.raises(ValueError):
            calculate_all(df, as_of=date.today())

    def test_works_with_mock_fetcher(self):
        """Mock Adapter 経由でも動く。"""
        fetcher = MockDataFetcher(spot_price=450.0)
        df = fetcher.get_option_chain("SPY", date.today())
        result = calculate_all(df, as_of=date.today(), data_source=fetcher.source_name)
        assert result.symbol == "SPY"
        assert result.data_source == "mock"


# ──────────────────────────────────────────────────────────
# 異常系の堅牢性
# ──────────────────────────────────────────────────────────

class TestRobustness:
    def test_handles_zero_oi_anomaly(self):
        """OI=0 が混じっても落ちない（schema は警告のみ通す）。"""
        fetcher = MockDataFetcher(spot_price=450.0, anomaly="zero_oi")
        df = fetcher.get_option_chain("SPY", date.today())
        result = calculate_all(df, as_of=date.today())
        # 計算結果が出る（NaN や inf にならない）
        assert np.isfinite(result.call_wall)
        assert np.isfinite(result.put_wall)
        assert np.isfinite(result.max_pain)

    def test_handles_extreme_iv_anomaly(self):
        """極端 IV が混じっても落ちない。"""
        fetcher = MockDataFetcher(spot_price=450.0, anomaly="extreme_iv")
        df = fetcher.get_option_chain("SPY", date.today())
        result = calculate_all(df, as_of=date.today())
        assert np.isfinite(result.call_wall)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ──────────────────────────────────────────────────────────
# data_quality 判定（v17）
# ──────────────────────────────────────────────────────────

class TestDataQuality:
    """_assess_data_quality の純粋ロジックと calculate_all への統合。"""

    # --- 純粋関数の単体テスト（判定順を含む） ---

    def test_ok_when_c_ge_z_ge_p(self):
        dq, detail = _assess_data_quality(call_wall=745.0, put_wall=730.0, zero_gamma=740.0)
        assert dq == "ok"
        assert detail is None

    def test_ok_at_boundary_z_equals_c(self):
        """Z == C は健全側（"ok"）。"""
        dq, _ = _assess_data_quality(call_wall=740.0, put_wall=730.0, zero_gamma=740.0)
        assert dq == "ok"

    def test_ok_at_boundary_z_equals_p(self):
        """Z == P は健全側（"ok"）。"""
        dq, _ = _assess_data_quality(call_wall=745.0, put_wall=730.0, zero_gamma=730.0)
        assert dq == "ok"

    def test_data_error_when_call_wall_missing(self):
        dq, detail = _assess_data_quality(call_wall=None, put_wall=730.0, zero_gamma=740.0)
        assert dq == "data_error"
        assert "call_wall" in detail

    def test_data_error_when_put_wall_missing(self):
        dq, detail = _assess_data_quality(call_wall=745.0, put_wall=None, zero_gamma=740.0)
        assert dq == "data_error"
        assert "put_wall" in detail

    def test_data_error_takes_precedence_over_anomaly(self):
        """Wall 不検出は zero_gamma の位置より優先（obs.E の誤分類を防ぐ）。"""
        dq, detail = _assess_data_quality(call_wall=None, put_wall=730.0, zero_gamma=999.0)
        assert dq == "data_error"

    def test_data_error_when_zero_gamma_none(self):
        """論点 c=c-1: zero_gamma 解なしは data_error。"""
        dq, detail = _assess_data_quality(call_wall=745.0, put_wall=730.0, zero_gamma=None)
        assert dq == "data_error"
        assert "zero_gamma" in detail

    def test_z_above_c_is_ok_not_anomaly(self):
        """誤判断32: Z > C は品質欠陥ではなく regime 構造 → "ok"（旧 anomaly 廃止）。"""
        dq, detail = _assess_data_quality(call_wall=740.0, put_wall=730.0, zero_gamma=742.0)
        assert dq == "ok"
        assert detail is None

    def test_z_below_p_is_ok_not_anomaly(self):
        """誤判断32: Z < P も "ok"（構造は z_position で記述）。"""
        dq, detail = _assess_data_quality(call_wall=745.0, put_wall=730.0, zero_gamma=725.0)
        assert dq == "ok"
        assert detail is None

    # --- z_position（地図の構造タグ。data_quality とは別レイヤー） ---

    def test_z_position_inside(self):
        assert derive_z_position(call_wall=745.0, put_wall=730.0, zero_gamma=740.0) == "inside"

    def test_z_position_boundary_is_inside(self):
        """Z==C / Z==P は inside（整序側）。"""
        assert derive_z_position(740.0, 730.0, 740.0) == "inside"
        assert derive_z_position(745.0, 730.0, 730.0) == "inside"

    def test_z_position_above_call(self):
        assert derive_z_position(call_wall=740.0, put_wall=730.0, zero_gamma=742.0) == "above_call"

    def test_z_position_below_put(self):
        assert derive_z_position(call_wall=745.0, put_wall=730.0, zero_gamma=725.0) == "below_put"

    def test_z_position_none_when_inputs_missing(self):
        assert derive_z_position(call_wall=None, put_wall=730.0, zero_gamma=740.0) is None
        assert derive_z_position(call_wall=745.0, put_wall=730.0, zero_gamma=None) is None

    # --- calculate_all への統合テスト ---

    def test_normal_mock_is_ok(self):
        """正常な Mock データは data_quality="ok"、anomaly_detail=None。"""
        df = generate_option_chain(spot_price=450.0, seed=42)
        result = calculate_all(df, as_of=date.today())
        assert result.data_quality == "ok"
        assert result.anomaly_detail is None

    def test_wall_fallback_yields_data_error_via_calculate_all(self):
        """Put のみのチェーンは spot 以上の正 GEX が無く Call Wall が
        フォールバック → calculate_all が data_error を返す。"""
        df = generate_option_chain(spot_price=450.0, seed=42)
        df_puts = df[df["right"] == "put"].copy()
        result = calculate_all(df_puts, as_of=date.today())
        assert result.data_quality == "data_error"
        assert result.call_wall == result.underlying_price

    # --- 内部ヘルパー（None 返し）の確認 ---

    def test_find_call_wall_opt_returns_none_on_fallback(self):
        gex = pd.Series(
            [-100.0, -200.0, -150.0],
            index=pd.Index([440.0, 450.0, 460.0], name="strike"),
            name="net_gex",
        )
        assert _find_call_wall_opt(gex, spot=450.0) is None

    def test_find_put_wall_opt_returns_none_on_fallback(self):
        gex = pd.Series(
            [100.0, 200.0, 150.0],
            index=pd.Index([440.0, 450.0, 460.0], name="strike"),
            name="net_gex",
        )
        assert _find_put_wall_opt(gex, spot=450.0) is None
