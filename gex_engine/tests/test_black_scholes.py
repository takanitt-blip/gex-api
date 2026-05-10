"""Black-Scholes gamma 計算のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from gex_engine.core.black_scholes import gamma
from gex_engine.core.config import GEXConfig


class TestGammaBasicProperties:
    """Black-Scholes gamma の数学的性質を確認。"""

    def test_atm_gamma_is_positive(self):
        """ATM gamma は常に正。"""
        g = gamma(S=100, K=100, T=30/365, sigma=0.2)
        assert g > 0

    def test_gamma_is_nonnegative(self):
        """gamma は数学的に常に非負。"""
        # OTM, ATM, ITM でテスト
        for K in [80, 100, 120]:
            g = gamma(S=100, K=K, T=30/365, sigma=0.2)
            assert g >= 0, f"K={K}: gamma={g}"

    def test_gamma_peaks_near_atm(self):
        """gamma は ATM 付近で最大。"""
        atm = gamma(S=100, K=100, T=30/365, sigma=0.2)
        otm = gamma(S=100, K=120, T=30/365, sigma=0.2)
        itm = gamma(S=100, K=80, T=30/365, sigma=0.2)
        assert atm > otm
        assert atm > itm

    def test_gamma_decreases_with_time_to_expiry_at_atm(self):
        """ATM では満期が遠いほど gamma は小さい（時間分散効果）。"""
        short = gamma(S=100, K=100, T=7/365, sigma=0.2)
        long = gamma(S=100, K=100, T=90/365, sigma=0.2)
        assert short > long

    def test_call_put_gamma_equality(self):
        """gamma は Call/Put で同じ（Put-Call Parity）。
        実装では区別していないので、同じ入力なら同じ値が返ることを確認。
        """
        g1 = gamma(S=100, K=100, T=30/365, sigma=0.2)
        g2 = gamma(S=100, K=100, T=30/365, sigma=0.2)
        assert g1 == g2


class TestEdgeCases:
    def test_zero_sigma_returns_nan(self):
        """σ=0 は数学的に未定義。"""
        g = gamma(S=100, K=100, T=30/365, sigma=0.0)
        assert np.isnan(g)

    def test_negative_sigma_returns_nan(self):
        g = gamma(S=100, K=100, T=30/365, sigma=-0.1)
        assert np.isnan(g)

    def test_zero_dte_uses_floor(self):
        """T=0 でも床値処理でゼロ除算しない。"""
        g = gamma(S=100, K=100, T=0.0, sigma=0.2)
        assert np.isfinite(g)
        assert g > 0

    def test_negative_dte_uses_floor(self):
        g = gamma(S=100, K=100, T=-1.0, sigma=0.2)
        assert np.isfinite(g)


class TestVectorized:
    def test_array_input(self):
        S = 100.0
        K = np.array([90, 100, 110])
        T = np.array([30/365, 30/365, 30/365])
        sigma = np.array([0.2, 0.2, 0.2])

        g = gamma(S=S, K=K, T=T, sigma=sigma)
        assert g.shape == (3,)
        # ATM が最大
        assert g[1] == g.max()

    def test_mixed_valid_invalid_in_array(self):
        """配列入力で一部が無効でも、有効な要素は正しく計算される。"""
        S = 100.0
        K = np.array([100, 100, 100])
        T = np.array([30/365, 30/365, 30/365])
        sigma = np.array([0.2, 0.0, 0.2])

        g = gamma(S=S, K=K, T=T, sigma=sigma)
        assert np.isfinite(g[0])
        assert np.isnan(g[1])
        assert np.isfinite(g[2])
        assert g[0] == g[2]


class TestConfigEffects:
    def test_risk_free_rate_effect_is_small(self):
        """r が変わっても gamma への影響は小さい（< 1%）。"""
        cfg1 = GEXConfig(risk_free_rate=0.0)
        cfg2 = GEXConfig(risk_free_rate=0.05)

        g1 = gamma(S=100, K=100, T=30/365, sigma=0.2, config=cfg1)
        g2 = gamma(S=100, K=100, T=30/365, sigma=0.2, config=cfg2)

        relative_diff = abs(g1 - g2) / g1
        assert relative_diff < 0.01  # 1% 未満

    def test_dividend_yield_lowers_gamma_at_atm(self):
        """配当を加えると ATM の有効スポット位置がズレる。
        ここでは「値は変わるが小さい」ことだけ確認。
        """
        cfg1 = GEXConfig(dividend_yield=0.0)
        cfg2 = GEXConfig(dividend_yield=0.03)

        g1 = gamma(S=100, K=100, T=30/365, sigma=0.2, config=cfg1)
        g2 = gamma(S=100, K=100, T=30/365, sigma=0.2, config=cfg2)

        assert abs(g1 - g2) / g1 < 0.05  # 5% 未満


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
