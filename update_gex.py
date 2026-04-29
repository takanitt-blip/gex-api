import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime

TICKER_SYMBOL = "SPY"
HISTORY_FILE = "gex_history.json"

def calculate_gex():
    print(f"--- {TICKER_SYMBOL} の解析（原点回帰モード）を開始します ---")
    ticker = yf.Ticker(TICKER_SYMBOL)
    
    # 1. 満期日を取得（先の方まで見すぎないよう、直近10個に限定）
    expirations = ticker.options[:10]
    if not expirations:
        raise Exception("オプションデータが取得できませんでした。")
    
    # 現在価格を取得
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    
    all_calls = []
    all_puts = []

    # 2. データを集計（以前のシンプルかつ強力な方法）
    for date in expirations:
        try:
            opt = ticker.option_chain(date)
            all_calls.append(opt.calls[['strike', 'openInterest']])
            all_puts.append(opt.puts[['strike', 'openInterest']])
        except:
            continue

    df_calls = pd.concat(all_calls)
    df_puts = pd.concat(all_puts)

    # 3. 権利行使価格ごとに合計して、単純に「最大値」を見つける
    call_oi_sum = df_calls.groupby('strike')['openInterest'].sum()
    put_oi_sum = df_puts.groupby('strike')['openInterest'].sum()

    # 最大建玉（OI）の価格を特定
    call_wall = float(call_oi_sum.idxmax())
    put_wall = float(put_oi_sum.idxmax())
    
    # Zero Gamma (以前と同じく、CallとPutの均衡点付近を採用)
    zero_gamma = (call_wall + put_wall) / 2

    print(f"【計算結果】Current: {current_price:.2f}, CallWall: {call_wall}, PutWall: {put_wall}")

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(zero_gamma, 2),
        "underlying_price": round(current_price, 2)
    }

def main():
    try:
        new_data = calculate_gex()
        today_str = datetime.now().strftime("%Y.%m.%d")
        history = {}
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                try: history = json.load(f)
                except: history = {}
        
        history[today_str] = new_data
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        print(f"✅ {today_str} のデータを保存しました。")
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()
