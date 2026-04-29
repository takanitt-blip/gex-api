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

    # 1. 現在価格を取得（5日分取得して週末・祝日でも安全に対応）
    hist = ticker.history(period="5d")
    if hist.empty:
        raise Exception("価格データが取得できませんでした。ティッカーシンボルを確認してください。")
    current_price = float(hist["Close"].iloc[-1])

    # 2. 満期日を取得（直近10件に限定してAPI負荷を抑える）
    expirations = ticker.options[:10]
    if not expirations:
        raise Exception("オプションデータが取得できませんでした。")

    all_calls = []
    all_puts = []

    # 3. データを集計（失敗した満期日はスキップしてログに残す）
    for date in expirations:
        try:
            opt = ticker.option_chain(date)
            all_calls.append(opt.calls[["strike", "openInterest"]])
            all_puts.append(opt.puts[["strike", "openInterest"]])
        except Exception as e:
            print(f"  警告: {date} の取得失敗 - {e}")
            continue

    # 4. 有効データが1件もなければ例外を発生させる
    if not all_calls or not all_puts:
        raise Exception("有効なオプションデータが1件もありませんでした。")

    df_calls = pd.concat(all_calls, ignore_index=True)
    df_puts = pd.concat(all_puts, ignore_index=True)

    # 5. ATM 周辺（±15%）のストライクに絞ってノイズを低減
    atm_mask_calls = (
        abs(df_calls["strike"] - current_price) / current_price <= ATM_RANGE
    )
    atm_mask_puts = (
        abs(df_puts["strike"] - current_price) / current_price <= ATM_RANGE
    )
    df_calls = df_calls[atm_mask_calls]
    df_puts = df_puts[atm_mask_puts]

    if df_calls.empty or df_puts.empty:
        raise Exception(f"ATM ±{int(ATM_RANGE*100)}% 以内にオプションデータがありませんでした。")

    # 6. 権利行使価格ごとに OI を合計して、最大OIのストライクを特定
    call_oi_sum = df_calls.groupby("strike")["openInterest"].sum()
    put_oi_sum = df_puts.groupby("strike")["openInterest"].sum()

    call_wall = float(call_oi_sum.idxmax())
    put_wall = float(put_oi_sum.idxmax())

    print(f"取得した現在価格: {current_price}")
    print(f"ATMフィルタ後のCallストライク一覧:\n{call_oi_sum.sort_index()}")
    print(f"ATMフィルタ後のPutストライク一覧:\n{put_oi_sum.sort_index()}")

    # 7. Zero Gamma: OI 加重平均で算出（単純平均より精度が高い）
    call_oi_at_wall = float(call_oi_sum[call_wall])
    put_oi_at_wall = float(put_oi_sum[put_wall])
    total_oi = call_oi_at_wall + put_oi_at_wall
    zero_gamma = (
        (call_wall * call_oi_at_wall + put_wall * put_oi_at_wall) / total_oi
        if total_oi > 0
        else (call_wall + put_wall) / 2
    )

    print(
        f"【計算結果】"
        f"Current: {current_price:.2f}, "
        f"CallWall: {call_wall:.2f}, "
        f"PutWall: {put_wall:.2f}, "
        f"ZeroGamma: {zero_gamma:.2f}"
    )

    return {
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "zero_gamma": round(zero_gamma, 2),
        "underlying_price": round(current_price, 2),
    }


def main():
    # コマンドライン引数で銘柄・出力ファイルを指定可能にする
    parser = argparse.ArgumentParser(description="GEX トラッカー")
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"ティッカーシンボル (デフォルト: {DEFAULT_TICKER})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_HISTORY_FILE,
        help=f"履歴JSONファイル名 (デフォルト: {DEFAULT_HISTORY_FILE})",
    )
    args = parser.parse_args()

    try:
        new_data = calculate_gex(args.ticker)

        # タイムスタンプ付きキーで同日複数実行でも上書きされない
        timestamp_str = datetime.now().strftime("%Y.%m.%d_%H%M")

        history = {}
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    print("  警告: 既存の履歴ファイルが破損していたため、新規作成します。")
                    history = {}

        history[timestamp_str] = new_data

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, ensure_ascii=False)

        print(f"✅ {timestamp_str} のデータを '{args.output}' に保存しました。")

    except Exception as e:
        print(f"❌ エラー: {e}")


if __name__ == "__main__":
    main()
