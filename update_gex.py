import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# 保存する履歴ファイル名
HISTORY_FILE = "gex_history.json"

def calculate_gex():
    ticker_symbol = "SPY"
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. 最も近い満期日（限月）を取得
    expirations = ticker.options
    if not expirations:
        raise Exception("オプションデータが見つかりません。")
    
    target_date = expirations[0]  # 当日または直近の満期
    opt_chain = ticker.option_chain(target_date)
    calls = opt_chain.calls
    puts = opt_chain.puts
    
    # 現在の価格（計算用の中間値として使用）
    current_price = ticker.history(period="1d")['Close'].iloc[-1]

    # --- ガンマの簡易計算ロジック（Renderで使っていたもの） ---
    # Call Wall: Open Interest（建玉）が最大の権利行使価格
    call_wall = float(calls.loc[calls['openInterest'].idxmax()]['strike'])
    
    # Put Wall: Open Interest（建玉）が最大の権利行使価格
    put_wall = float(puts.loc[puts['openInterest'].idxmax()]['strike'])
    
    # Zero Gamma: 簡易版（CallとPutの建玉の差が最小になる点など、以前のロジックに合わせます）
    # ここでは例として、建玉の合計が最も大きい付近を算出するロジック、
    # または以前のコードがあればそれを反映させます。
    # 一旦、平均値に近い統計的ポイントをZero Gammaとして計算
    zero_gamma = (call_wall + put_wall) / 2 

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(zero_gamma, 2),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    try:
        print("データ取得中...")
        new_data = calculate_gex()
        today_str = datetime.now().strftime("%Y.%m.%d")
        
        history = {}
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except:
                history = {}
        
        # 今日の日付で保存
        history[today_str] = new_data
        
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        print(f"✅ {today_str} の履歴を更新しました！")
        print(f"Call Wall: {new_data['call_wall']}, Put Wall: {new_data['put_wall']}")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
