"""Black-Scholes モデルによる gamma 計算。

数式（数式設計議論で確定）:

    γ  =  N'(d1) / (S × σ × √T)

    d1 = [ln(S/K) + (r - q + σ²/2) × T] / (σ × √T)

    where:
        N'(x) = (1/√(2π)) × exp(-x²/2)   標準正規分布の PDF
        S: スポット価格
        K: ストライク
        σ: IV (decimal、0.15 = 15%)
        T: 満期までの時間（年単位）
        r: 無リスク金利
        q: 配当利回り

重要な性質:
    Call と Put の gamma は等しい（Put-Call Parity）
    → 関数は 1 つだけ提供すれば十分
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from gex_engine.core.config import DEFAULT_CONFIG, GEXConfig


def gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    sigma: float | np.ndarray,
    config: GEXConfig = DEFAULT_CONFIG,
) -> float | np.ndarray:
    """Black-Scholes gamma を計算する。

    スカラーでも numpy 配列でも動作する（ベクトル化対応）。

    Args:
        S: スポット価格
        K: ストライク
        T: 満期までの時間（年単位）。0 や負の値は config の最小値で置換
        sigma: IV（decimal）
        config: 計算設定。デフォルトは DEFAULT_CONFIG

    Returns:
        gamma 値（同じ shape）

    Notes:
        - σ <= 0 の入力では NaN を返す（数学的に未定義）
        - T < min_time_to_expiry の場合は床値で置換
        - Call/Put 共通（同じ式）
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # T の床値処理（0DTE 対応）
    T = np.maximum(T, config.min_time_to_expiry)

    # σ <= 0 を NaN マスクとして扱う
    # （素直に計算すると warning が出るので明示的に処理）
    valid = sigma > 0

    # d1 計算（無効な要素は後で NaN で上書きするので、
    # 一時的にダミー値を入れて警告を抑制）
    sigma_safe = np.where(valid, sigma, 1.0)

    d1 = (
        np.log(S / K) + (config.risk_free_rate - config.dividend_yield
                         + 0.5 * sigma_safe ** 2) * T
    ) / (sigma_safe * np.sqrt(T))

    result = norm.pdf(d1) / (S * sigma_safe * np.sqrt(T))

    # 無効入力は NaN に
    result = np.where(valid, result, np.nan)

    # スカラー入力にはスカラーで返す
    if result.ndim == 0:
        return float(result)
    return result
