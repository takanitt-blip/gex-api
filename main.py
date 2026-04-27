from fastapi import FastAPI
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm

# FastAPIのアプリケーションを作成
app = FastAPI(title="GEX & Zero Gamma API", description="SPYのオプション防衛ラインを計算するAPI")

# ブラック・ショールズのGamma計算関数
def calc_gamma(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = np.exp(-0.5 * d1 ** 2) / (S * sigma * np.sqrt(2 * np.pi * T))
    return gamma

# 「/api/gex」というURLにアクセスが来たときに実行される関数
@app.get("/api/gex")
def get_gex_data(ticker_symbol: str = "SPY"):
    try:
        # 1. データの取得
        ticker = yf.Ticker(ticker_symbol)
        spot_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # 直近の満期日を取得
        expirations = ticker.options
        if not expirations:
            return {"error": "オプションデータが見つかりません"}
            
        nearest_expiry = expirations[0] 
        opt_chain = ticker.option_chain(nearest_expiry)
        calls, puts = opt_chain.calls, opt_chain.puts

        T = 1.0 / 365.0 
        total_gex_by_strike = {}

        # 2. コールのGEX計算
        for _, row in calls.iterrows():
            K, OI, IV = row['strike'], row['openInterest'], row['impliedVolatility']
            if pd.isna(OI) or OI == 0 or pd.isna(IV): continue
            gamma = calc_gamma(spot_price, K, T, IV)
            total_gex_by_strike[K] = total_gex_by_strike.get(K, 0) + (OI * gamma * 100 * spot_price)

        # 3. プットのGEX計算 (マイナス影響)
        for _, row in puts.iterrows():
            K, OI, IV = row['strike'], row['openInterest'], row['impliedVolatility']
            if pd.isna(OI) or OI == 0 or pd.isna(IV): continue
            gamma = calc_gamma(spot_price, K, T, IV)
            total_gex_by_strike[K] = total_gex_by_strike.get(K, 0) - (OI * gamma * 100 * spot_price)

        # 4. 指標の特定
        call_wall = max(total_gex_by_strike, key=total_gex_by_strike.get)
        put_wall = min(total_gex_by_strike, key=total_gex_by_strike.get)
        
        # 5. Zero Gammaの推定
        zero_gamma = None
        for sim_price in np.arange(spot_price * 1.05, spot_price * 0.95, -0.5):
            sim_total_gex = sum(
                gex * (calc_gamma(sim_price, K, T, 0.15) / (calc_gamma(spot_price, K, T, 0.15) + 1e-9))
                for K, gex in total_gex_by_strike.items()
            )
            if sim_total_gex < 0:
                zero_gamma = round(sim_price, 2)
                break

        # FastAPIはreturnした辞書を自動的にJSON形式にして返却します
        return {
            "symbol": ticker_symbol,
            "spot_price": round(spot_price, 2),
            "expiry": nearest_expiry,
            "metrics": {
                "call_wall": call_wall,
                "put_wall": put_wall,
                "zero_gamma": zero_gamma if zero_gamma else "Not Found"
            }
        }
        
    except Exception as e:
        return {"error": str(e)}