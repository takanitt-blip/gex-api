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
ATM_RANGE            = 0.10  # 上下10%の範囲に絞り、Far OTMの宝くじノイズを完全排除
RISK_FREE_RATE       = 0.05  # リスクフリーレート（5%）
CONTRACT_SIZE        = 100   # 1契約あたりの株数
MAX_DTE              = 60    # 対象とする最大残存日数 (0〜60日の短期オプションに絞る)

# ==========================================
# 金融工学関数
# ==========================================
def bs_gamma(S, K, T, r, sigma):
    """Black-Scholesモデルによるガンマの計算"""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def time_to_expiry(expiry_str):
    """満期までの年率換算時間を計算（0DTE対応済み）"""
    exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp_date - date.today()).days
    if days <= 0:
        days = 0.5  # 0DTE(満期日当日)は半日(0.5日)として計算に含める
    return days / 365.0

def calc_total_gex_for_spot(spot, options_data):
    """特定の原資産価格(spot)におけるトータルGEXをシミュレーション計算する"""
    total_gex = 0.0
    for opt in options_data:
        gamma = bs_gamma(spot, opt['K'], opt['T'], RISK_FREE_RATE, opt['iv'])
        total_gex += opt['sign'] * gamma * opt['oi'] * CONTRACT_SIZE * spot
    return total_gex

def find_true_zero_gamma(current_spot, options_data):
    """
    原資産価格(S)を上下に動かし、Total GEXが0を跨ぐ(反転する)価格を探す。
    """
    # シミュレーション範囲をオプションデータの範囲内に限定
    sim_spots = np.arange(current_spot * 0.95, current_spot * 1.05, 0.25)

    best_spot = current_spot
    min_gex_abs = float("inf")

    for s in sim_spots:
        gex = calc_total_gex_for_spot(s, options_data)
        if abs(gex) < min_gex_abs:
            min_gex_abs = abs(gex)
            best_spot = s

    return best_spot

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
        # フィルター1：DTE(残存日数)が遠すぎる長期オプション（LEAPS）を除外
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

                # フィルター2：yfinance特有の「早朝OI消失バグ」対策
                contracts = oi if oi > 0 else vol

                if contracts == 0 or iv is None:
                    continue

                # フィルター3：IVの異常値を弾く
                if iv < 0.01 or iv > 3.0:
                    continue

                # フィルター4：DITM(ディープ・イン・ザ・マネー)のノイズを弾く
                if flag == "call" and K < S * 0.95:
                    continue
                if flag == "put" and K > S * 1.05:
                    continue

                # フィルター5：ATM_RANGE外のFar OTMを弾く
                if abs(K - S) / S > ATM_RANGE:
                    continue

                gamma = bs_gamma(S, K, T, RISK_FREE_RATE, iv)
                gex = sign * gamma * contracts * CONTRACT_SIZE * S

                records.append({"strike": K, "gex": gex})
                options_data.append({
                    "K": K, "T": T, "iv": iv, "oi": contracts, "sign": sign
                })

    if not records:
        raise Exception("GEX を計算できる有効なデータがありませんでした。")

    df = pd.DataFrame(records)
    gex_by_strike = df.groupby("strike")["gex"].sum()
    total_gex = float(gex_by_strike.sum())

    # Call Wall: 現在価格より上で、プラスGEXが最大のストライク
    positive_above = gex_by_strike[(gex_by_strike > 0) & (gex_by_strike.index >= S)]
    call_wall = float(positive_above.idxmax()) if not positive_above.empty else S

    # Put Wall: 現在価格より下で、マイナスGEXの絶対値が最大のストライク
    negative_below = gex_by_strike[(gex_by_strike < 0) & (gex_by_strike.index <= S)]
    put_wall = float(negative_below.idxmin()) if not negative_below.empty else S

    # 真のZero Gammaの計算 (シミュレーション)
    print("  Zero Gamma をシミュレーション計算中...")
    zero_gamma = find_true_zero_gamma(S, options_data)

    # レジーム判定
    if total_gex > 0:
        regime = "range"
        regime_text = "レンジ相場・低ボラティリティ"
    else:
        regime = "trend"
        regime_text = "トレンド相場・高ボラティリティ"

    print(f"  Call Wall : {call_wall:.2f}")
    print(f"  Put Wall  : {put_wall:.2f}")
    print(f"  Zero Gamma: {zero_gamma:.2f}")
    print(f"  Total GEX : {total_gex:+,.0f}  → {regime_text}")

    return {
        "call_wall":        round(call_wall,  2),
        "put_wall":         round(put_wall,   2),
        "zero_gamma":       round(zero_gamma, 2),
        "underlying_price": round(S,          2),
        "total_gex":        round(total_gex,  2),
        "regime":           regime,
        "regime_text":      regime_text,
    }

# ==========================================
# エントリーポイント
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="GEX トラッカー（最終・完全版）")
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

        print(f"✅ {today_str} のデータを '{args.output}' に保存しました。")

    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()
