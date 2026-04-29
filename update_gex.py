import yfinance as yf
import pandas as pd
import json
import os
import argparse
from datetime import datetime

DEFAULT_TICKER = "SPY"
DEFAULT_HISTORY_FILE = "gex_history.json"
ATM_RANGE = 0.15  # 現在価格の ±15% 以内のストライクに絞る


def calculate_gex(ticker_symbol: str) -> dict:
    print(f"--- {ticker_symbol} の解析を開始します ---")
    ticker = yf.Ticker(ticker_symbol)

    # 現在価格を取得（5日分取得して週末・祝日でも安全に対応）
    hist = ticker.history(period="5d")
    if hist.empty:
        raise Exception("価格データが取得できませんでした。ティッカーシンボルを確認してください。")
    current_price = float(hist["Close"].iloc[-1])
    print(f"  現在価格: {current_price:.2f}")

    # 満期日を取得（直近10件に限定）
    expirations = ticker.options[:10]
    if not expirations:
        raise Exception("オプションデータが取得できませんでした。")

    all_calls = []
    all_puts = []

    # volume ベースで集計（OIはGitHub Actions環境で0になるため）
    for date in expirations:
        try:
            opt = ticker.option_chain(date)

            calls = opt.calls[["strike", "volume"]].copy()
            puts  = opt.puts[["strike", "volume"]].copy()

            # volume が NaN のものは 0 に置換
            calls["volume"] = calls["volume"].fillna(0)
            puts["volume"]  = puts["volume"].fillna(0)

            all_calls.append(calls)
            all_puts.append(puts)
        except Exception as e:
            print(f"  警告: {date} の取得失敗 - {e}")
            continue

    if not all_calls or not all_puts:
        raise Exception("有効なオプションデータが1件もありませんでした。")

    df_calls = pd.concat(all_calls, ignore_index=True)
    df_puts  = pd.concat(all_puts,  ignore_index=True)

    # ATM ±15% に絞ってノイズを低減
    df_calls = df_calls[
        abs(df_calls["strike"] - current_price) / current_price <= ATM_RANGE
    ]
    df_puts = df_puts[
        abs(df_puts["strike"] - current_price) / current_price <= ATM_RANGE
    ]

    if df_calls.empty or df_puts.empty:
        raise Exception(f"ATM ±{int(ATM_RANGE*100)}% 以内にオプションデータがありませんでした。")

    # ストライクごとに volume を合計
    call_vol_sum = df_calls.groupby("strike")["volume"].sum()
    put_vol_sum  = df_puts.groupby("strike")["volume"].sum()

    # volume が全部 0 の場合はエラー
    if call_vol_sum.sum() == 0 or put_vol_sum.sum() == 0:
        raise Exception(
            "volume が全て0です。市場クローズ中か、データ取得タイミングの問題の可能性があります。"
        )

    call_wall = float(call_vol_sum.idxmax())
    put_wall  = float(put_vol_sum.idxmax())

    # Zero Gamma: volume 加重平均
    call_vol_at_wall = float(call_vol_sum[call_wall])
    put_vol_at_wall  = float(put_vol_sum[put_wall])
    total_vol = call_vol_at_wall + put_vol_at_wall
    zero_gamma = (
        (call_wall * call_vol_at_wall + put_wall * put_vol_at_wall) / total_vol
        if total_vol > 0
        else (call_wall + put_wall) / 2
    )

    print(
        f"  CallWall: {call_wall:.2f} (volume: {int(call_vol_at_wall):,})\n"
        f"  PutWall:  {put_wall:.2f} (volume: {int(put_vol_at_wall):,})\n"
        f"  ZeroGamma:{zero_gamma:.2f}"
    )

    return {
        "call_wall":        round(call_wall,    2),
        "put_wall":         round(put_wall,     2),
        "zero_gamma":       round(zero_gamma,   2),
        "underlying_price": round(current_price, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="GEX トラッカー")
    parser.add_argument("--ticker", default=DEFAULT_TICKER,
                        help=f"ティッカーシンボル (デフォルト: {DEFAULT_TICKER})")
    parser.add_argument("--output", default=DEFAULT_HISTORY_FILE,
                        help=f"履歴JSONファイル名 (デフォルト: {DEFAULT_HISTORY_FILE})")
    args = parser.parse_args()

    try:
        new_data = calculate_gex(args.ticker)

        # ⚠️ キー形式を "YYYY.MM.DD" に固定（EAの日付パース形式に合わせる）
        today_str = datetime.now().strftime("%Y.%m.%d")

        history = {}
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    print("  警告: 既存の履歴ファイルが破損していたため、新規作成します。")
                    history = {}

        history[today_str] = new_data

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        print(f"✅ {today_str} のデータを '{args.output}' に保存しました。")

    except Exception as e:
        print(f"❌ エラー: {e}")


if __name__ == "__main__":
    main()
