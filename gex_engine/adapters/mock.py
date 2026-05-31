"""Mock Adapter: 契約前の開発・テスト用のダミーデータ生成。

設計方針（議論で確定）:
    - レベル2: 構造を持ったダミー（完全ランダムではない）
    - ストライクは $5 刻み、スポット ±20% の範囲
    - IV はスマイル形状（ATM で低く、両端で高い）
    - OI は ATM 付近で大きく、ITM/OTM ほど小さい
    - seed 必須（再現性確保）

このモックの目的は「現実の完璧な模倣」ではなく、
Core Logic（GEX 計算、Zero Gamma 探索、Max Pain）が
意味のある結果を返せる状態を作ること。

YAGNI 警告:
    複数 expiration、配当、金利の動的化などは現状不要。
    必要になってから追加する。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from gex_engine.schema import coerce_to_schema, empty_dataframe

AnomalyType = Literal[
    "zero_oi",        # 一部の OI を 0 に
    "extreme_iv",     # 一部の IV を 6.0 (600%) に
    "crossed_quote",  # 一部で bid > ask
    "empty",          # 休場日想定（空 DF）
]


# ──────────────────────────────────────────────────────────
# データ生成関数（純粋関数、Adapter から独立してテスト可能）
# ──────────────────────────────────────────────────────────

def generate_option_chain(
    symbol: str = "SPY",
    spot_price: float = 450.0,
    trade_date: date | None = None,
    expiration: date | None = None,
    strike_step: float = 5.0,
    strike_range_pct: float = 0.20,
    base_iv: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """構造を持ったダミーオプションチェーンを生成する。

    Args:
        symbol: シンボル名
        spot_price: 現在のスポット価格
        trade_date: Adapter が解釈した取引日 T。None なら今日。
            schema.REQUIRED_DTYPES の trade_date 列に全行同じ値で入る。
        expiration: 満期日。None なら 30 日後の金曜日
        strike_step: ストライクの刻み幅（$）
        strike_range_pct: スポット価格の何 % まで広げるか（片側）
        base_iv: ATM の IV（decimal、0.15 = 15%）
        seed: 乱数シード（再現性確保）

    Returns:
        schema.REQUIRED_DTYPES に準拠した DataFrame。
        Call と Put の両方を含む。
    """
    rng = np.random.default_rng(seed)

    if trade_date is None:
        trade_date = datetime.now().date()
    if expiration is None:
        expiration = _next_friday(days_ahead=30)

    # ── ストライクグリッドの生成 ──
    # スポット価格を $5 刻みグリッドにスナップ（実際の取引所と同様）
    spot_snapped = round(spot_price / strike_step) * strike_step
    span = spot_price * strike_range_pct
    n_strikes_each_side = int(span / strike_step)
    strikes = np.array([
        spot_snapped + i * strike_step
        for i in range(-n_strikes_each_side, n_strikes_each_side + 1)
    ])

    rows = []
    for strike in strikes:
        for right in ("call", "put"):
            row = _generate_single_option(
                symbol=symbol,
                spot_price=spot_price,
                strike=strike,
                right=right,
                expiration=expiration,
                base_iv=base_iv,
                rng=rng,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    # 取引日 T を全行に持たせる（誤判断25, 2026-05-24）
    # schema.REQUIRED_DTYPES の trade_date 列。詳細は schema.py のメモ参照。
    df["trade_date"] = pd.Timestamp(trade_date)
    return coerce_to_schema(df)


def generate_with_anomaly(
    anomaly_type: AnomalyType,
    **kwargs,
) -> pd.DataFrame:
    """異常系データを生成する。

    Core Logic の堅牢性テスト用。
    通常の generate_option_chain() に異常を「注入」する形で実装。

    Args:
        anomaly_type: 異常の種類
        **kwargs: generate_option_chain() に渡す引数

    Returns:
        異常を含む DataFrame（"empty" の場合は空 DF）
    """
    if anomaly_type == "empty":
        # empty_dataframe() は schema 由来で trade_date 列を含む（γ-1）
        return empty_dataframe()

    df = generate_option_chain(**kwargs)
    n = len(df)

    if anomaly_type == "zero_oi":
        # 全行の 30% の OI を 0 に
        n_corrupt = max(1, n // 3)
        idx = np.random.default_rng(seed=99).choice(n, n_corrupt, replace=False)
        df.loc[idx, "open_interest"] = 0

    elif anomaly_type == "extreme_iv":
        # 全行の 10% の IV を 6.0 (600%) に
        n_corrupt = max(1, n // 10)
        idx = np.random.default_rng(seed=99).choice(n, n_corrupt, replace=False)
        df.loc[idx, "implied_volatility"] = 6.0

    elif anomaly_type == "crossed_quote":
        # 1 行だけ bid > ask に
        df.loc[0, "bid"] = 5.0
        df.loc[0, "ask"] = 1.0

    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    return df


# ──────────────────────────────────────────────────────────
# 内部ヘルパー
# ──────────────────────────────────────────────────────────

def _generate_single_option(
    symbol: str,
    spot_price: float,
    strike: float,
    right: str,
    expiration: date,
    base_iv: float,
    rng: np.random.Generator,
) -> dict:
    """1 オプションコントラクトの行を生成。"""
    # ── IV: スマイル形状（ATM 低く、両端高い） ──
    # moneyness = log(K/S)。ATM で 0、ITM/OTM で離れる。
    moneyness = np.log(strike / spot_price)
    # 二次関数的なスマイル: iv = base + smile_factor * moneyness^2
    smile_factor = 0.5
    iv = base_iv + smile_factor * (moneyness ** 2)
    # 微小なノイズを乗せる（実物の歪みっぽさ）
    iv += rng.normal(0, 0.005)
    iv = max(0.01, iv)  # 数学的下限

    # ── OI: 実市場に倣った非対称構造 ──
    # 実際の SPY オプション市場では:
    #   ・Call OI のピークは spot より上（OTM Call にヘッジ・投機が積まれる）
    #   ・Put OI のピークは spot より下（OTM Put にヘッジ需要）
    # これにより Net Gamma が spot 上下で符号反転し、Zero Gamma が spot 付近に出る。
    #
    # 数学的に対称な Mock にしないのは、現実構造を反映するためであり、
    # カーブフィッティングではない。
    oi_peak = 5000
    oi_width = spot_price * 0.05  # 標準偏差: スポットの 5%

    if right == "call":
        # Call OI のピークは spot より +1% 程度上
        peak_strike = spot_price * 1.01
    else:
        # Put OI のピークは spot より -1% 程度下
        peak_strike = spot_price * 0.99

    distance_from_peak = strike - peak_strike
    oi_mean = oi_peak * np.exp(-(distance_from_peak ** 2) / (2 * oi_width ** 2))
    # ガウス的なばらつき（負にならない）
    oi = max(0, int(rng.normal(oi_mean, oi_mean * 0.2 + 50)))

    # ── 価格（bid/ask）: 簡易な内在価値 + 時間価値 ──
    # 厳密な BS は不要。Core Logic は IV と OI と underlying_price しか
    # 使わない（gamma を自前計算するため）。bid/ask は飾り。
    if right == "call":
        intrinsic = max(0, spot_price - strike)
    else:
        intrinsic = max(0, strike - spot_price)
    time_value = spot_price * iv * 0.05  # 適当な time value 近似
    mid = intrinsic + time_value
    spread = max(0.01, mid * 0.02)  # 2% スプレッド
    bid = max(0.0, mid - spread / 2)
    ask = mid + spread / 2

    return {
        "symbol": symbol,
        "expiration": pd.Timestamp(expiration),
        "strike": float(strike),
        "right": right,
        "bid": float(bid),
        "ask": float(ask),
        "implied_volatility": float(iv),
        "open_interest": int(oi),
        "underlying_price": float(spot_price),
    }


def _next_friday(days_ahead: int = 30) -> date:
    """指定日数後以降の最初の金曜日を返す。"""
    target = datetime.now().date() + timedelta(days=days_ahead)
    # 金曜は weekday() == 4
    days_to_friday = (4 - target.weekday()) % 7
    return target + timedelta(days=days_to_friday)


# ──────────────────────────────────────────────────────────
# Adapter 本体
# ──────────────────────────────────────────────────────────

class MockDataFetcher:
    """DataFetcher Protocol の Mock 実装。

    使用例:
        fetcher = MockDataFetcher(seed=42)
        df = fetcher.get_option_chain("SPY", date.today())

    異常系を試したい場合:
        fetcher = MockDataFetcher(anomaly="zero_oi")
        df = fetcher.get_option_chain("SPY", date.today())
    """

    source_name = "mock"

    def __init__(
        self,
        spot_price: float = 450.0,
        seed: int = 42,
        anomaly: AnomalyType | None = None,
    ):
        """
        Args:
            spot_price: モックで使うスポット価格
            seed: 乱数シード
            anomaly: 異常系シナリオ。None なら正常系
        """
        self.spot_price = spot_price
        self.seed = seed
        self.anomaly = anomaly

    def get_option_chain(
        self, symbol: str, as_of: date
    ) -> pd.DataFrame:
        """オプションチェーンを返す。

        as_of は trade_date 列としてデータに反映される（誤判断25, 2026-05-24）。
        Mock では「前営業日」概念がないので as_of そのものを trade_date に
        使う（rest と違って T 解決ロジックを持たない）。
        スポット価格・OI/IV 分布は as_of に依存しない（再現性確保のため）。
        """
        if self.anomaly is not None:
            return generate_with_anomaly(
                self.anomaly,
                symbol=symbol,
                spot_price=self.spot_price,
                trade_date=as_of,
                seed=self.seed,
            )
        return generate_option_chain(
            symbol=symbol,
            spot_price=self.spot_price,
            trade_date=as_of,
            seed=self.seed,
        )


    def schedule_type_on(self, target: date) -> str:
        """DataFetcher Protocol: 指定日の市場スケジュール type（素朴版）。

        Mock は実カレンダーを持たないため、平日 -> "open" / 土日 -> "weekend"。
        market_calendar.next_business_day の schedule_lookup として注入される。
        """
        # weekday(): Mon=0 .. Fri=4, Sat=5, Sun=6
        return "weekend" if target.weekday() >= 5 else "open"