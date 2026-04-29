import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, date
from scipy.stats import norm

# ==========================================
# 設定パラメータ
# ==========================================
DEFAULT_TICKER       = "SPY"
DEFAULT_HISTORY_FILE = "gex_history.json"
ATM_RANGE_INNER      = 0.05  # Inner Band: ±5%（短期の壁）
ATM_RANGE_OUTER      = 0.10  # Outer Band: ±10%（最終防衛ライン）
RISK_FREE_RATE       = 0.05
CONTRACT_SIZE        = 100
MAX_DTE              = 60

# ==========================================
# 金融工学関数
# ==========================================
def bs_gamma(S, K, T, r, sigma):
    """Black-Scholesモデルによるガンマの計算（スカラー版）"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def time_to_expiry(expiry_str):
    """満期までの年率換算時間を計算（0DTE対応済み）"""
    exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp_date - date.today()).days
    if days <= 0:
        days = 0.5
    return days / 365.0


def find_true_zero_gamma(current_spot, options_data):
    """numpy ベクトル演算で Zero Gamma を高速計算"""
    if not options_data:
        return current_spot

    sim_spots = np.arange(current_spot * 0.95, current_spot * 1.05, 0.25)

    K    = np.array([o['K']    for o in options_data])
    T    = np.array([o['T']    for o in options_data])
    iv   = np.array([o['iv']   for o in options_data])
    oi   = np.array([o['oi']   for o in options_data])
    sign = np.array([o['sign'] for o in options_data])

    S_grid = sim_spots[:, np.newaxis]

    d1 = (np.log(S_grid / K) + (RISK_FREE_RATE + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    gamma = norm.pdf(d1) / (S_grid * iv * np.sqrt(T))
    gex = sign * gamma * oi * CONTRACT_SIZE * (S_grid ** 2) * 0.01

    total_gex = gex.sum(axis=1)
    best_idx = np.argmin(np.abs(total_gex))
    return float(sim_spots[best_idx])


# ==========================================
# Wall 算出ロジック
# ==========================================
def find_walls(gex_by_strike, S):
    """
    現在価格より上でプラスGEX最大 = Call Wall
    現在価格より下でマイナスGEX最大 = Put Wall
    """
    positive_above = gex_by_strike[(gex_by_strike > 0) & (gex_by_strike.index >= S)]
    call_wall = float(positive_above.idxmax()) if not positive_above.empty else S

    negative_below = gex_by_strike[(gex_by_strike < 0) & (gex_by_strike.index <= S)]
    put_wall = float(negative_below.idxmin()) if not negative_below.empty else S

    return call_wall, put_wall


# ==========================================
# メイン計算ロジック
# ==========================================
def calculate_gex(ticker_symbol):
    print(f"--- {ticker_symbol} の GEX 計算を開始します ---")
    ticker = yf.Ticker(ticker_symbol)

    hist = ticker.history(period="5d")
    if hist.empty:
        raise Exception("価格データが取得できませんでした。")
    S = float(hist["Close"].iloc[-1])
    print(f"  現在価格: {S:.2f}")

    expirations = ticker.options
    if not expirations:
        raise Exception("オプションデータが取得できませんでした。")

    records = []
    options_data = []

    for exp_date in expirations:
        exp_d = datetime.strptime(exp_date, "%Y-%m-%d").date()
        days_to_exp = (exp_d - date.today()).days
        if days_to_exp > MAX_DTE:
            continue

        T = time_to_expiry(exp_date)
        try:
            chain = ticker.option_chain(exp_date)
        except Exception as e:
            print(f"  警告: {exp_date} 取得失敗 - {e}")
            continue

        for flag, df_opt, sign in [("call", chain.calls, 1), ("put", chain.puts, -1)]:
            for _, row in df_opt.iterrows():
                K  = float(row["strike"])
                iv = float(row["impliedVolatility"]) if row["impliedVolatility"] > 0 else None
                oi = float(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0
                vol = float(row["volume"]) if not pd.isna(row["volume"]) else 0

                contracts = oi if oi > 0 else vol

                if contracts == 0 or iv is None:
                    continue
                if iv < 0.01 or iv > 3.0:
                    continue

                # Outer 範囲（±10%）でフィルタ
                moneyness = abs(K - S) / S
                if moneyness > ATM_RANGE_OUTER:
                    continue

                gamma = bs_gamma(S, K, T, RISK_FREE_RATE, iv)
                gex = sign * gamma * contracts * CONTRACT_SIZE * (S ** 2) * 0.01

                records.append({
                    "strike": K,
                    "gex": gex,
                    "moneyness": moneyness,
                })
                options_data.append({
                    "K": K, "T": T, "iv": iv, "oi": contracts, "sign": sign
                })

    if not records:
        raise Exception("GEX を計算できる有効なデータがありませんでした。")

    df = pd.DataFrame(records)

    # ── Outer Band（±10%）──
    gex_outer = df.groupby("strike")["gex"].sum()
    total_gex = float(gex_outer.sum())

    # デバッグ: Outer のGEX分布
    print("\n  [DEBUG Outer] プラスGEX上位5（現在価格より上）:")
    pos_above_o = gex_outer[(gex_outer > 0) & (gex_outer.index >= S)].sort_values(ascending=False)
    print(pos_above_o.head())

    print("  [DEBUG Outer] マイナスGEX上位5（現在価格より下）:")
    neg_below_o = gex_outer[(gex_outer < 0) & (gex_outer.index <= S)].sort_values()
    print(neg_below_o.head())

    call_wall_outer, put_wall_outer = find_walls(gex_outer, S)

    # ── Inner Band（±5%）──
    df_inner = df[df["moneyness"] <= ATM_RANGE_INNER]
    if not df_inner.empty:
        gex_inner = df_inner.groupby("strike")["gex"].sum()

        # デバッグ: Inner のGEX分布
        print("\n  [DEBUG Inner] プラスGEX上位5（現在価格より上）:")
        pos_above_i = gex_inner[(gex_inner > 0) & (gex_inner.index >= S)].sort_values(ascending=False)
        print(pos_above_i.head())

        print("  [DEBUG Inner] マイナスGEX上位5（現在価格より下）:")
        neg_below_i = gex_inner[(gex_inner < 0) & (gex_inner.index <= S)].sort_values()
        print(neg_below_i.head())

        call_wall_inner, put_wall_inner = find_walls(gex_inner, S)
    else:
        call_wall_inner = S
        put_wall_inner = S

    # ── Zero Gamma（ベクトル演算版）──
    print("\n  Zero Gamma をベクトル演算で計算中...")
    zero_gamma = find_true_zero_gamma(S, options_data)

    # ── レジーム判定 ──
    if total_gex > 0:
        regime = "range"
        regime_text = "レンジ相場・低ボラティリティ"
    else:
        regime = "trend"
        regime_text = "トレンド相場・高ボラティリティ"

    print(f"\n  [Inner] Call Wall: {call_wall_inner:.2f}  Put Wall: {put_wall_inner:.2f}")
    print(f"  [Outer] Call Wall: {call_wall_outer:.2f}  Put Wall: {put_wall_outer:.2f}")
    print(f"  Zero Gamma : {zero_gamma:.2f}")
    print(f"  Total GEX  : {total_gex:+,.0f}  → {regime_text}")

    return {
        "call_wall_inner":  round(call_wall_inner,  2),
        "put_wall_inner":   round(put_wall_inner,   2),
        "call_wall_outer":  round(call_wall_outer,  2),
        "put_wall_outer":   round(put_wall_outer,   2),
        "zero_gamma":       round(zero_gamma,        2),
        "underlying_price": round(S,                 2),
        "total_gex":        round(total_gex,         2),
        "regime":           regime,
        "regime_text":      regime_text,
    }


# ==========================================
# エントリーポイント
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="GEX トラッカー（2バンド版・デバッグ付き）")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--output", default=DEFAULT_HISTORY_FILE)
    args = parser.parse_args()

    try:
        new_data = calculate_gex(args.ticker)
        today_str = datetime.now().strftime("%Y.%m.%d")

        history = {}
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    print("  警告: 履歴ファイルが破損していたため新規作成します。")

        history[today_str] = new_data

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        print(f"\n✅ {today_str} のデータを '{args.output}' に保存しました。")

    except Exception as e:
        print(f"❌ エラー: {e}")


if __name__ == "__main__":
    main()
