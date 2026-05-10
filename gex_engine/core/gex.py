"""GEX 計算のコアロジック。

含まれる関数:
    - calculate_gex_per_strike()  : ストライク別の Net GEX
    - find_call_wall()             : Call Wall 検出
    - find_put_wall()              : Put Wall 検出
    - find_zero_gamma()            : Brent 法で Zero Gamma
    - find_max_pain()              : Max Pain
    - calculate_all()              : 上記を全部まとめて GEXResult を返す

数式（議論で確定）:
    GEX_per_option = γ × OI × contract_size × sign
    sign = +1 (call), -1 (put)   ← ディーラー視点
    
    × S や × S² は呼び出し側の表示時に追加（内部は素の単位）
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from gex_engine.core.black_scholes import gamma
from gex_engine.core.config import DEFAULT_CONFIG, GEXConfig
from gex_engine.core.result import GEXResult
from gex_engine.schema import validate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# T (時間) を expiration から計算
# ──────────────────────────────────────────────────────────

def _compute_time_to_expiry(
    expiration: pd.Series,
    as_of: date,
    config: GEXConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """満期までの時間 T を年単位で計算。

    暦日ベース、最小値は config.min_time_to_expiry。
    """
    as_of_ts = pd.Timestamp(as_of)
    days_to_exp = (expiration - as_of_ts).dt.days.to_numpy(dtype=float)
    T = days_to_exp / 365.0
    T = np.maximum(T, config.min_time_to_expiry)
    return T


# ──────────────────────────────────────────────────────────
# ストライク別 Net GEX
# ──────────────────────────────────────────────────────────

def calculate_gex_per_strike(
    df: pd.DataFrame,
    as_of: date,
    spot_override: Optional[float] = None,
    config: GEXConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """ストライク別の Net GEX を計算する。

    Args:
        df: schema 準拠の DataFrame
        as_of: 基準日
        spot_override: スポット価格を上書き（Zero Gamma 探索で使用）。
                       None なら df["underlying_price"] の値を使う。
        config: 計算設定

    Returns:
        index=strike, values=Net GEX (call は +、put は -) の Series
    """
    if len(df) == 0:
        return pd.Series(dtype=float, name="net_gex")

    S = (
        spot_override
        if spot_override is not None
        else float(df["underlying_price"].iloc[0])
    )
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["implied_volatility"].to_numpy(dtype=float)
    oi = df["open_interest"].to_numpy(dtype=float)
    sign = np.where(df["right"].to_numpy() == "call", 1.0, -1.0)

    T = _compute_time_to_expiry(df["expiration"], as_of, config)

    g = gamma(S=S, K=K, T=T, sigma=sigma, config=config)

    # NaN（IV<=0 等）は寄与ゼロとして扱う
    g = np.nan_to_num(g, nan=0.0)

    gex_per_option = sign * g * oi * config.contract_size

    # ストライクで集約
    out = pd.DataFrame({"strike": K, "gex": gex_per_option})
    return out.groupby("strike")["gex"].sum().rename("net_gex")


# ──────────────────────────────────────────────────────────
# Call Wall / Put Wall
# ──────────────────────────────────────────────────────────

def find_call_wall(gex_by_strike: pd.Series, spot: float) -> float:
    """Call Wall: スポット以上で Net GEX が最大（正）のストライク。

    解が無ければ spot を返す（フォールバック）。
    """
    above = gex_by_strike[(gex_by_strike.index >= spot) & (gex_by_strike > 0)]
    if above.empty:
        logger.warning("No positive GEX above spot; returning spot as Call Wall")
        return spot
    return float(above.idxmax())


def find_put_wall(gex_by_strike: pd.Series, spot: float) -> float:
    """Put Wall: スポット以下で Net GEX が最小（最も負）のストライク。

    解が無ければ spot を返す（フォールバック）。
    """
    below = gex_by_strike[(gex_by_strike.index <= spot) & (gex_by_strike < 0)]
    if below.empty:
        logger.warning("No negative GEX below spot; returning spot as Put Wall")
        return spot
    return float(below.idxmin())


# ──────────────────────────────────────────────────────────
# Zero Gamma（Brent 法）
# ──────────────────────────────────────────────────────────

def find_zero_gamma(
    df: pd.DataFrame,
    as_of: date,
    config: GEXConfig = DEFAULT_CONFIG,
) -> Optional[float]:
    """Zero Gamma: Net Gamma がゼロになるスポット価格を探索。

    Net Gamma(S*) = Σ_calls γ(S*) × OI - Σ_puts γ(S*) × OI = 0
    を満たす S* を Brent 法で求める。

    探索範囲: ストライクの min/max（フォールバックで spot ± 20%）

    Args:
        df: schema 準拠の DataFrame
        as_of: 基準日
        config: 計算設定

    Returns:
        Zero Gamma の S* 値。解なしなら None。
    """
    if len(df) == 0:
        return None

    spot = float(df["underlying_price"].iloc[0])
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["implied_volatility"].to_numpy(dtype=float)
    oi = df["open_interest"].to_numpy(dtype=float)
    sign = np.where(df["right"].to_numpy() == "call", 1.0, -1.0)
    T = _compute_time_to_expiry(df["expiration"], as_of, config)

    def net_gamma(s_star: float) -> float:
        """与えられた s_star でのネットガンマ。"""
        g = gamma(S=s_star, K=K, T=T, sigma=sigma, config=config)
        g = np.nan_to_num(g, nan=0.0)
        return float(np.sum(sign * g * oi))

    # 探索範囲: ストライクの min/max
    s_low = max(float(K.min()), spot * (1 - config.zero_gamma_search_pct))
    s_high = min(float(K.max()), spot * (1 + config.zero_gamma_search_pct))

    if s_low >= s_high:
        logger.warning(
            "Invalid Zero Gamma search range: low=%s, high=%s",
            s_low, s_high,
        )
        return None

    f_low = net_gamma(s_low)
    f_high = net_gamma(s_high)

    if f_low * f_high > 0:
        # 両端で同符号 → 範囲内に解なし
        logger.info(
            "No sign change in Zero Gamma search range: f(%.2f)=%.2e, f(%.2f)=%.2e",
            s_low, f_low, s_high, f_high,
        )
        return None

    try:
        zero_g = brentq(net_gamma, s_low, s_high, xtol=0.01)
        return float(zero_g)
    except (ValueError, RuntimeError) as e:
        logger.warning("Brent solver failed: %s", e)
        return None


# ──────────────────────────────────────────────────────────
# Max Pain
# ──────────────────────────────────────────────────────────

def find_max_pain(df: pd.DataFrame) -> float:
    """Max Pain: オプション買い手の総損失を最小化するストライク。

    各候補ストライク K* について、
        cost(K*) = Σ_calls max(K* - strike, 0) × OI
                 + Σ_puts max(strike - K*, 0) × OI
    を計算し、最小の K* を返す。

    Notes:
        厳密には買い手の "intrinsic value at expiry" を最大化する K*
        ではなく、ディーラー（売り手）の支払い額を最小化する K*。
        実務上は「全オプションが同時満期したと仮定した場合の
        ペイオフ最小ストライク」として使われる。

    Args:
        df: schema 準拠の DataFrame

    Returns:
        Max Pain ストライク。データなしなら NaN。
    """
    if len(df) == 0:
        return float("nan")

    candidates = np.sort(df["strike"].unique())

    calls = df[df["right"] == "call"]
    puts = df[df["right"] == "put"]

    call_strikes = calls["strike"].to_numpy()
    call_oi = calls["open_interest"].to_numpy()
    put_strikes = puts["strike"].to_numpy()
    put_oi = puts["open_interest"].to_numpy()

    # 各候補 K* でのコスト計算（ベクトル化）
    # shape: (n_candidates, n_options)
    K_star = candidates[:, np.newaxis]

    call_payoff = np.maximum(K_star - call_strikes, 0) * call_oi
    put_payoff = np.maximum(put_strikes - K_star, 0) * put_oi

    total_cost = call_payoff.sum(axis=1) + put_payoff.sum(axis=1)

    return float(candidates[np.argmin(total_cost)])


# ──────────────────────────────────────────────────────────
# 統合: 全部計算して GEXResult を返す
# ──────────────────────────────────────────────────────────

def calculate_all(
    df: pd.DataFrame,
    as_of: date,
    data_source: str = "mock",
    config: GEXConfig = DEFAULT_CONFIG,
) -> GEXResult:
    """DataFrame から全水準を計算して GEXResult を返す。

    Args:
        df: schema 準拠の DataFrame
        as_of: 基準日
        data_source: "mock" / "rest" / "sdk"
        config: 計算設定

    Raises:
        SchemaValidationError: スキーマ検証で致命エラー
        ValueError: データが空、または必須情報が欠ける

    Returns:
        GEXResult
    """
    # スキーマ検証（致命エラーは例外）
    validate(df).raise_if_invalid()

    if len(df) == 0:
        raise ValueError("Cannot calculate GEX from empty DataFrame")

    symbol = str(df["symbol"].iloc[0])
    spot = float(df["underlying_price"].iloc[0])

    gex_by_strike = calculate_gex_per_strike(df, as_of, config=config)
    call_wall = find_call_wall(gex_by_strike, spot)
    put_wall = find_put_wall(gex_by_strike, spot)
    zero_gamma = find_zero_gamma(df, as_of, config=config)
    max_pain = find_max_pain(df)
    total_gex = float(gex_by_strike.sum())

    return GEXResult(
        symbol=symbol,
        as_of=datetime.combine(as_of, datetime.min.time()).isoformat(),
        underlying_price=spot,
        call_wall=call_wall,
        put_wall=put_wall,
        zero_gamma=zero_gamma,
        max_pain=max_pain,
        total_gex=total_gex,
        n_contracts_used=int(df["open_interest"].sum()),
        data_source=data_source,
    )
