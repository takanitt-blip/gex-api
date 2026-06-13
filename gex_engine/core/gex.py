"""GEX 計算のコアロジック。

含まれる関数:
    - calculate_gex_per_strike()  : ストライク別の Net GEX
    - find_call_wall()             : Call Wall 検出（公開・float 返し、フォールバック spot）
    - find_put_wall()              : Put Wall 検出（公開・float 返し、フォールバック spot）
    - find_zero_gamma()            : Brent 法で Zero Gamma
    - find_max_pain()              : Max Pain
    - calculate_all()              : 上記を全部まとめて GEXResult を返す

数式（議論で確定）:
    GEX_per_option = γ × OI × contract_size × sign
    sign = +1 (call), -1 (put)   ← ディーラー視点

    × S や × S² は呼び出し側の表示時に追加（内部は素の単位）

v17 変更（data_quality 導入）:
    - 内部ヘルパー _find_call_wall_opt / _find_put_wall_opt を追加。
      Wall が見つからなければ None を返す（calculate_all が品質判定に使う）。
    - 公開 find_call_wall / find_put_wall はこれを包んで従来どおり
      「None→spot ＋ WARNING ログ」を返す（戻り値・挙動とも不変）。
    - _assess_data_quality() で C/Z/P から data_quality / anomaly_detail を判定。
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
#
# 設計メモ（v17）:
#   Wall は spot 起点に固定して探索する（Call Wall ≥ spot、Put Wall ≤ spot）。
#   これは PC_CORE §3.2「Call Wall = Pr+α / Put Wall = Pr−β」の定義どおりで、
#   構造的に C ≥ spot ≥ P ＝ C ≥ P が常に成立する。よって PC_CORE §3.3 の
#   「P > C → data_error」は本実装では発火しない。
#   本実装での「データ品質エラー」の真の信号は『Wall が見つからない
#   （= フォールバックで spot を返す）』こと。これを観測するため、
#   None 返しの内部版 _find_*_wall_opt を分離する。

def _find_call_wall_opt(gex_by_strike: pd.Series, spot: float) -> Optional[float]:
    """Call Wall 候補。スポット以上で Net GEX が最大（正）のストライク。

    見つからなければ None（呼び出し側＝calculate_all が data_quality 判定に使う）。
    """
    above = gex_by_strike[(gex_by_strike.index >= spot) & (gex_by_strike > 0)]
    if above.empty:
        return None
    return float(above.idxmax())


def _find_put_wall_opt(gex_by_strike: pd.Series, spot: float) -> Optional[float]:
    """Put Wall 候補。スポット以下で Net GEX が最小（最も負）のストライク。

    見つからなければ None。
    """
    below = gex_by_strike[(gex_by_strike.index <= spot) & (gex_by_strike < 0)]
    if below.empty:
        return None
    return float(below.idxmin())


def find_call_wall(gex_by_strike: pd.Series, spot: float) -> float:
    """Call Wall: スポット以上で Net GEX が最大（正）のストライク。

    解が無ければ spot を返す（フォールバック）。
    （挙動は v17 前と不変。内部で _find_call_wall_opt を包むだけ。）
    """
    cw = _find_call_wall_opt(gex_by_strike, spot)
    if cw is None:
        logger.warning("No positive GEX above spot; returning spot as Call Wall")
        return spot
    return cw


def find_put_wall(gex_by_strike: pd.Series, spot: float) -> float:
    """Put Wall: スポット以下で Net GEX が最小（最も負）のストライク。

    解が無ければ spot を返す（フォールバック）。
    （挙動は v17 前と不変。内部で _find_put_wall_opt を包むだけ。）
    """
    pw = _find_put_wall_opt(gex_by_strike, spot)
    if pw is None:
        logger.warning("No negative GEX below spot; returning spot as Put Wall")
        return spot
    return pw


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
# data_quality 判定（v17、PC_CORE §3）
# ──────────────────────────────────────────────────────────

def _assess_data_quality(
    call_wall: Optional[float],
    put_wall: Optional[float],
    zero_gamma: Optional[float],
) -> tuple[str, Optional[str]]:
    """C/Z/P から地図の品質を判定する純粋関数。

    引数の call_wall / put_wall は **_find_*_wall_opt の戻り値**（None 可）を
    そのまま渡す。None は「Wall が見つからずフォールバックした」を意味する。

    判定順（前段が優先 ─ 後段の誤分類を防ぐ）:
      1. data_error : call_wall か put_wall が None（Wall 不検出）。
                      ← 現行 spot 固定探索では P>C が起きないため、これが
                        真のデータ品質エラー信号。Wall 不検出を放置して
                        zero_gamma と比較すると、データ問題を anomaly と
                        誤読する（obs.E の罠）。だから最優先で弾く。
      2. data_error : zero_gamma が None（Brent 解なし ＝ 地図が regime 分割に
                      使えない）。論点c=c-1。
      3. ok          : それ以外。Z と Wall の位置関係（Z∉[P,C]）は regime 構造で
                      あって品質欠陥ではないため判定しない（誤判断32：当日満期
                      混入が壁を spot にピンさせ Z∉[P,C] を量産していたが、当日
                      満期除外で大半が解消。残る非整序も崩壊ではなく実在配置）。
                      構造は z_position（C/Z/P からの派生）で記述する。

    Returns:
        (data_quality, anomaly_detail)
        正常時は ("ok", None)。
    """
    # 1. Wall フォールバック
    if call_wall is None or put_wall is None:
        missing = [
            name
            for name, value in (("call_wall", call_wall), ("put_wall", put_wall))
            if value is None
        ]
        return "data_error", f"wall not found (fell back to spot): {', '.join(missing)}"

    # 2. zero_gamma 解なし
    if zero_gamma is None:
        return (
            "data_error",
            "zero_gamma not found (no net-gamma sign change in search range)",
        )

    # 3. 正常。Z と Wall の位置関係（Z∉[P,C]）は regime 構造であって品質欠陥では
    #    ないため、ここでは判定しない（誤判断32）。構造は z_position（C/Z/P からの
    #    派生）で記述し、data_quality は {ok, data_error} に限定する。
    return "ok", None


def derive_z_position(
    call_wall: Optional[float],
    put_wall: Optional[float],
    zero_gamma: Optional[float],
) -> Optional[str]:
    """Zero Gamma と Wall レンジの位置関係を記述する純粋関数（誤判断32）。

    data_quality（品質欠陥）とは別レイヤーの「地図の構造タグ」。整序
    （P ≤ Z ≤ C）か、Z が Wall レンジから出た非整序かを表す。優位性検証の層別
    と、EA の環境判別（非整序日は壁をレンジ境界として使う確信度を下げる）に使う。

    Returns:
        "inside"      : P <= Z <= C（整序。4区分がそのまま使える）
        "above_call"  : Z > C（非整序。ネガγ寄りで直上にレジ）
        "below_put"   : Z < P（非整序。ポジγ寄りで直下にサポート）
        None          : C/P/Z のいずれかが None（= data_error。判定不能）
    """
    if call_wall is None or put_wall is None or zero_gamma is None:
        return None
    if zero_gamma > call_wall:
        return "above_call"
    if zero_gamma < put_wall:
        return "below_put"
    return "inside"


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
        as_of: 基準日（= 実取引日 T。a7-A 以降、呼び出し側が df["trade_date"]
               から取り出して渡す。today を直接渡さないこと ─ obs.F）
        data_source: "mock" / "rest" / "sdk"
        config: 計算設定

    Raises:
        SchemaValidationError: スキーマ検証で致命エラー
        ValueError: データが空、または必須情報が欠ける

    Returns:
        GEXResult（data_quality / anomaly_detail を含む）
    """
    # スキーマ検証（致命エラーは例外）
    validate(df).raise_if_invalid()

    if len(df) == 0:
        raise ValueError("Cannot calculate GEX from empty DataFrame")

    # 当日満期/期限切れ（expiration <= as_of, DTE<=0）を除外（誤判断32）。
    # 地図は EOD(T) から計算され翌セッション T+1 をガバナンスする。expiration <= as_of の
    # 建玉は T 引けで消滅済み＝翌セッションには存在せず、かつ T→0 で γ が床値まで爆発する
    # 退化 greeks。これらが per-strike Net GEX の argmax を spot 近傍にピンさせ、Zero Gamma
    # （本体は長期物の符号反転）との DTE 不整合から Z∉[P,C] anomaly を量産していた。
    # T+1 満期（セッションの 0DTE）は expiration > as_of なので DTE=1 で保持される。
    as_of_ts = pd.Timestamp(as_of)
    df = df[df["expiration"] > as_of_ts]
    if len(df) == 0:
        raise ValueError(
            f"No live options after excluding expiration <= as_of ({as_of_ts.date()})"
        )

    symbol = str(df["symbol"].iloc[0])
    spot = float(df["underlying_price"].iloc[0])

    gex_by_strike = calculate_gex_per_strike(df, as_of, config=config)

    # Wall は None 可の内部版で取得（data_quality 判定に None を使うため）
    call_wall_opt = _find_call_wall_opt(gex_by_strike, spot)
    put_wall_opt = _find_put_wall_opt(gex_by_strike, spot)
    zero_gamma = find_zero_gamma(df, as_of, config=config)
    max_pain = find_max_pain(df)
    total_gex = float(gex_by_strike.sum())

    # data_quality 判定（None ＝ Wall 不検出 を判定に使う）
    data_quality, anomaly_detail = _assess_data_quality(
        call_wall_opt, put_wall_opt, zero_gamma
    )
    if data_quality != "ok":
        # 「異常を検出した」事実をログに残す（PC_VALIDATION §1.6 / §3.4 の方針）
        logger.warning("data_quality=%s: %s", data_quality, anomaly_detail)

    # GEXResult の数値フィールドは従来どおり float（None はフォールバック spot）。
    # ← JSON 出力の数値・既存テスト（np.isfinite）の後方互換を保つ。
    call_wall = call_wall_opt if call_wall_opt is not None else spot
    put_wall = put_wall_opt if put_wall_opt is not None else spot

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
        data_quality=data_quality,
        anomaly_detail=anomaly_detail,
    )
