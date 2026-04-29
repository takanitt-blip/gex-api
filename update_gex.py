import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# --- 設定 ---
TICKER_SYMBOL = "SPY"
HISTORY_FILE = "gex_history.json"

def calculate_gex():
    print(f"--- {TICKER_SYMBOL} の全オプションデータ解析を開始します ---")
    ticker = yf.Ticker(TICKER_SYMBOL)
    
    # 1. すべての満期日を取得
    expirations = ticker.options
    if not expirations:
        raise Exception("オプションデータが取得できませんでした。")
    
    # 計算時点の現在価格を取得（MT5側のスケール調整用）
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"現在価格: {current_price:.2f}")

    all_calls = []
    all_puts = []

    # 2. 全ての満期日をループしてデータを集計
    for date in expirations:
        try:
            opt_chain = ticker.option_chain(date)
            all_calls.append(opt_chain.calls[['strike', 'openInterest']])
            all_puts.append(opt_chain.puts[['strike', 'openInterest']])
            print(f"取得済み: {date}")
        except Exception as e:
            print(f"スキップ: {date} (エラー: {e})")

    # データフレームの統合
    df_calls = pd.concat(all_calls)
    df_puts = pd.concat(all_puts)

    # 3. 権利行使価格（Strike）ごとに建玉（Open Interest）を合算
    call_oi_sum = df_calls.groupby('strike')['openInterest'].sum()
    put_oi_sum = df_puts.groupby('strike')['openInterest'].sum()

    # --- Call Wall & Put Wall の算出 ---
    # 市場全体で最も建玉が集中している価格を特定
    call_wall = float(call_oi_sum.idxmax())
    put_wall = float(put_oi_sum.idxmax())

    # --- Zero Gamma (Proxy) の算出 ---
    # Net OI (Call OI - Put OI) が 0 を跨ぐ、あるいは最小になるポイントを特定
    # これにより「強気と弱気の分岐点」が正確に出ます
    all_strikes = sorted(list(set(df_calls['strike']) | set(df_puts['strike'])))
    net_oi = []
    for s in all_strikes:
        c = call_oi_sum.get(s, 0)
        p = put_oi_sum.get(s, 0)
        net_oi.append(abs(c - p))
    
    # 最も均衡しているポイント
    zero_gamma = float(all_strikes[np.argmin(net_oi)])

    print(f"解析完了: Call Wall={call_wall}, Put Wall={put_wall}, Zero Gamma={zero_gamma}")

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(zero_gamma, 2),
        "underlying_price": round(current_price, 2),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def main():
    try:
        new_data = calculate_gex()
        today_str = datetime.now().strftime("%Y.%m.%d")
        
        # 履歴の読み込みと更新
        history = {}
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except:
                    history = {}
        
        history[today_str] = new_data
        
        # 保存
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        print(f"✅ {today_str} のデータを保存しました。")
        
    except Exception as e:
        print(f"❌ 致命的なエラー: {e}")

if __name__ == "__main__":
    main()
