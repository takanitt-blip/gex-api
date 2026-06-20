#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_v2a_band_calibration.py  --  優位性検証 V-2a（バンド較正）

================================  事前登録（凍結）  ================================
研究設問:
  ④のEOD信号 signed_distance でセッションのボラ予報（=バンド幅）を条件づけると、
  trailing realized vol 単独で条件づけた場合より較正(calibration)が良いか。
  ＝ GEX が「ボラ・クラスタリングの上に」足す“増分の band 価値”を単離する
     （④ §1.8 の統制構造 log(RV) ~ trailing_log_rv (+ signed_distance) を継承）。

仮説（数値を見る前に確定）:
  H1: baseline(trailing単独) はネガγ日でボラを過小予報し、必要なときバンドが狭すぎる。
      ⇔ 標準化残差 r=log(RV/σ̂) を signed_distance に回帰した傾き γ_base が有意に負。
      treatment(trailing+signed_distance) では γ_gex が 0 近傍へ減衰
      （|γ_gex| < 0.5·|γ_base| かつ γ_gex の 95%CI が 0 を含む）。
  H0/null: γ_base ≈ 0（trailing 単独で既にレジーム較正済）→ GEX は band に無価値
           → V-2a は null（正直で許容される結果）。

標本:
  地図 = gex_history.json の data_quality=="ok"。primary = z_position=="inside"。
  価格 = US500 M15（MT5）。ET = broker − 7h（§3.6, DST 安定）。
         RTH = ET[09:30,16:00) = 26 本/日。<26 本（半日/不完全）は除外。
  期間ピン --start 2024-06-09 --end 2026-06-11（④と同一）。
  outcome セッション = 地図 key 当日（key = next_business_day(T) = EOD(T)地図が支配するセッション）。

予報・手法（walk-forward OOS = in-sample 直交の循環を回避）:
  signed_distance_T = (underlying_price − zero_gamma) / underlying_price  （EOD(T)、凍結）
  trailing_log_rv   = 直近 5 本の full-RTH 日 log(RV) 平均（④と同一、当該セッション直前まで）
  拡張窓 OLS（最小学習 60 セッション、時系列順）で各 OOS セッションに:
      baseline : log(RV) ~ 1 + trailing_log_rv
      treatment: log(RV) ~ 1 + trailing_log_rv + signed_distance
  σ̂ = sqrt(exp(ŷ))、標準化残差 r = log(RV / σ̂) = 0.5·(log(RV) − ŷ)。
  OOS の r を signed_distance に HAC(Newey-West) 回帰 → γ_base, γ_gex。

合否（凍結）:
  一次 = γ_base（有意に負か） vs γ_gex（0 へ減衰したか）の対比。
  二次 = q=0.90 片側バンド B = k·σ̂（k は学習窓のみで推定、test 非参照）の
         OOS 逸脱率（RV>B）と pinball(0.90) 損失。treatment < baseline なら H1 支持。

カーブフィッティング封じ:
  一次検定は乗数 k 不要。バンドの k も学習窓の経験分位点のみで test 成績では最適化しない。
  係数は過去データ OLS のみ（手置きしない）。walk-forward 厳守。

既知の限界（事前登録・辻褄合わせしない）:
  セッション単位の静的バンド。日中 Z 跨ぎ（二相）は RV_K に吸収（別途 V-2c）。
  RV(分散型) vs 実高安レンジは比例定数差 → 傾き検定は不変、絶対被覆 k は近似として記述のみ。
  US500≠SPY ベーシスは絶対水準のみに効き、対数リターン RV ≒ SPY RV（§3.6）。

RV 定義（再現性のため明示）:
  RTH 26 本の連続 close-to-close 対数リターン（25 本、オーバーナイト除外）の二乗和の平方根。
  ※ validate_zero_gamma_rv.py が「先頭 open→close を含む 26 本」を採るなら、当該行を 1 行直して整合。
     傾き/較正の結論は一定スケール下で不変だが、係数の絶対値再現には合わせること。

依存: numpy / pandas / scipy のみ。CI 非実行。出力 CSV は US500 由来 → .gitignore 推奨。
本番パイプライン（gex_engine/）には一切触れない（検証は tools/ に閉じる）。
=================================================================================
"""

import argparse
import sys
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

# ---- 定数（§3.6 / 事前登録）----------------------------------------------------
BROKER_MINUS_ET_HOURS = 7        # ET = broker - 7h
RTH_START = (9, 30)              # ET 09:30 (inclusive)
RTH_END = (16, 0)               # ET 16:00 (exclusive)
RTH_BARS_REQUIRED = 26          # full-RTH 本数
TRAILING_N = 5                  # trailing_log_rv の窓
MIN_TRAIN = 60                  # walk-forward 最小学習セッション数
COVERAGE_Q = 0.90               # 二次のバンド分位点
PIN_RANGE_DEFAULT = ("2024-06-09", "2026-06-11")


# ---- US500 価格ロード（ET 化 → RTH → RV）---------------------------------------
def load_us500_rv(price_csv: str) -> pd.DataFrame:
    """US500 M15 CSV から、ET セッション日ごとの RV（full-RTH のみ）を返す。
    返り値: DataFrame[index=ET date(datetime.date), columns=['rv','n_bars']]（full-RTH 26 本のみ）。"""
    df = pd.read_csv(
        price_csv, sep="\t", engine="python",
        names=["date", "time", "open", "high", "low", "close",
               "tickvol", "vol", "spread"],
        header=0,
    )
    # broker datetime
    bdt = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    et = bdt - pd.Timedelta(hours=BROKER_MINUS_ET_HOURS)
    df = df.assign(et=et, close=pd.to_numeric(df["close"], errors="coerce"))
    df = df.dropna(subset=["close"]).sort_values("et").reset_index(drop=True)

    # RTH マスク: ET 時刻 in [09:30, 16:00)
    et_minutes = df["et"].dt.hour * 60 + df["et"].dt.minute
    start_m = RTH_START[0] * 60 + RTH_START[1]
    end_m = RTH_END[0] * 60 + RTH_END[1]
    rth = df[(et_minutes >= start_m) & (et_minutes < end_m)].copy()
    rth["et_date"] = rth["et"].dt.date

    rows = []
    for d, g in rth.groupby("et_date", sort=True):
        g = g.sort_values("et")
        n = len(g)
        if n != RTH_BARS_REQUIRED:
            # full-RTH のみ採用（半日/不完全は除外、事前登録）
            rows.append((d, np.nan, n))
            continue
        logc = np.log(g["close"].to_numpy())
        rets = np.diff(logc)               # 連続 close-to-close（25 本、オーバーナイト除外）
        rv = float(np.sqrt(np.sum(rets ** 2)))
        rows.append((d, rv, n))
    out = pd.DataFrame(rows, columns=["et_date", "rv", "n_bars"]).set_index("et_date")
    return out


# ---- 地図ロード（EOD(T) 分類子）------------------------------------------------
def load_map(gex_json: str) -> pd.DataFrame:
    with open(gex_json, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for key, v in d.items():
        try:
            kd = datetime.strptime(key, "%Y.%m.%d").date()
        except ValueError:
            continue
        up = v.get("underlying_price")
        zg = v.get("zero_gamma")
        rows.append({
            "session_date": kd,                      # = key = 統治セッション K
            "data_quality": v.get("data_quality"),
            "z_position": v.get("z_position"),
            "underlying_price": up,
            "zero_gamma": zg,
            "as_of": v.get("as_of"),
        })
    m = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    # signed_distance（zero_gamma 欠損 = data_error 相当はあとで除外）
    m["signed_distance"] = (m["underlying_price"] - m["zero_gamma"]) / m["underlying_price"]
    return m


# ---- 自前 HAC(Newey-West) OLS --------------------------------------------------
def ols_hac(y, X, names=None):
    """y: (n,), X: (n,k) with intercept列込み。Newey-West HAC SE。
    lag = floor(4*(n/100)^(2/9))。返り値: dict(beta, se, t, p, n, lag, r2)。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    # S = sum_t e_t^2 x_t x_t' + sum_{l=1}^{L} w_l sum (x_t e_t e_{t-l} x_{t-l}' + transpose)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        Xe = X * resid[:, None]
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=n - k))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "beta": beta, "se": se, "t": tval, "p": pval,
        "n": n, "lag": L, "r2": r2, "names": names,
        "ci95": np.vstack([beta - 1.96 * se, beta + 1.96 * se]).T,
    }


def predict_sigma(train, test_row, use_signed):
    """train: DataFrame(logrv, trailing, signed)、test_row: Series。σ̂=sqrt(exp(ŷ)) と ŷ を返す。"""
    cols = ["const", "trailing"]
    if use_signed:
        cols.append("signed")
    Xtr = train[cols].to_numpy(float)
    ytr = train["logrv"].to_numpy(float)
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    xrow = np.array([1.0, test_row["trailing"]] + ([test_row["signed"]] if use_signed else []))
    yhat = float(xrow @ beta)
    return math.sqrt(math.exp(yhat)), yhat


def pinball(actual, forecast, q):
    d = actual - forecast
    return np.where(d >= 0, q * d, (q - 1) * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gex", default="gex_history.json")
    ap.add_argument("--price", default="US500_M15_202406030100_202606122100.csv")
    ap.add_argument("--start", default=PIN_RANGE_DEFAULT[0])
    ap.add_argument("--end", default=PIN_RANGE_DEFAULT[1])
    ap.add_argument("--all-zpos", action="store_true",
                    help="z_position!=inside も含める（robustness）。既定は inside のみ。")
    ap.add_argument("--out", default="v2a_band_calib_per_session.csv")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    rv = load_us500_rv(args.price)              # index=et_date, rv, n_bars
    m = load_map(args.gex)
    # マージキーを datetime64 に正規化（pandas 3.0 では object型の datetime.date 同士が
    # 値一致でも merge で突合しないため。両側を datetime64 に揃える＝根本修正）。
    m["session_date"] = pd.to_datetime(m["session_date"])

    # full-RTH の RV 系列（trailing 用）。地図と独立に時系列順で trailing を作る。
    full = rv.dropna(subset=["rv"]).copy()
    full = full[full["rv"] > 0].sort_index()
    full["logrv"] = np.log(full["rv"])
    # trailing = 直近 TRAILING_N 本の full-RTH 日 logrv の平均（当日は含めず、直前まで）
    full["trailing"] = full["logrv"].shift(1).rolling(TRAILING_N).mean()

    # 地図 key（= 統治セッション K）と full-RTH RV を結合
    m2 = m[(m["data_quality"] == "ok")].copy()
    if not args.all_zpos:
        m2 = m2[m2["z_position"] == "inside"]
    m2 = m2.dropna(subset=["signed_distance"])
    m2 = m2[(m2["session_date"] >= start) & (m2["session_date"] <= end)]

    full_reset = full.reset_index().rename(columns={"et_date": "session_date"})
    full_reset["session_date"] = pd.to_datetime(full_reset["session_date"])
    panel = m2.merge(full_reset[["session_date", "logrv", "trailing", "rv"]],
                     on="session_date", how="inner")
    panel = panel.dropna(subset=["logrv", "trailing", "signed_distance"])
    panel = panel.sort_values("session_date").reset_index(drop=True)
    panel = panel.rename(columns={"signed_distance": "signed"})
    panel["const"] = 1.0

    n = len(panel)
    print(f"[sample] merged sessions = {n}  (inside={'no(all)' if args.all_zpos else 'yes'}, "
          f"range {args.start}..{args.end})")
    if n < MIN_TRAIN + 30:
        print(f"[warn] サンプルが少ない（n={n}）。MIN_TRAIN={MIN_TRAIN} を見直すか期間を確認。")
    if n == 0:
        print("[FATAL] merged=0。考えられる原因:")
        if not m["z_position"].notna().any():
            print("  - gex_history.json に z_position が無い（z_position 移行“前”の古い版）。")
            print("    移行後の版（data_quality 全 ok・z_position あり）で再生成/差し替えを。")
        print("  - もしくは data_quality / 期間ピン / 価格カバレッジ。diag_v2a2.py で各段の件数を確認。")
        return

    # ---- walk-forward OOS ----
    recs = []
    for i in range(MIN_TRAIN, n):
        train = panel.iloc[:i]
        test = panel.iloc[i]
        sig_b, yh_b = predict_sigma(train, test, use_signed=False)
        sig_g, yh_g = predict_sigma(train, test, use_signed=True)
        rv_i = test["rv"]
        r_base = math.log(rv_i / sig_b)
        r_gex = math.log(rv_i / sig_g)
        recs.append({
            "session_date": test["session_date"],
            "signed": test["signed"],
            "rv": rv_i,
            "sigma_base": sig_b, "sigma_gex": sig_g,
            "r_base": r_base, "r_gex": r_gex,
        })
    oos = pd.DataFrame(recs)

    # ---- 一次: 標準化残差 r ~ signed の HAC 回帰 ----
    Xo = np.column_stack([np.ones(len(oos)), oos["signed"].to_numpy()])
    fit_base = ols_hac(oos["r_base"].to_numpy(), Xo, names=["const", "signed"])
    fit_gex = ols_hac(oos["r_gex"].to_numpy(), Xo, names=["const", "signed"])

    def fmt(fit, label):
        b = fit["beta"][1]; se = fit["se"][1]; t = fit["t"][1]; p = fit["p"][1]
        lo, hi = fit["ci95"][1]
        print(f"  {label}: slope(signed)={b:+.4f}  HAC_se={se:.4f}  t={t:+.2f}  "
              f"p(two)={p:.3g}  95%CI=[{lo:+.4f},{hi:+.4f}]  (n={fit['n']}, lag={fit['lag']})")
        return b, lo, hi, p

    print("\n=== 一次: 標準化残差 r=log(RV/σ̂) の signed_distance への傾き（OOS, HAC）===")
    gb, gb_lo, gb_hi, gb_p = fmt(fit_base, "baseline (trailing単独)")
    gg, gg_lo, gg_hi, gg_p = fmt(fit_gex, "treatment(trailing+signed)")

    # 凍結合否ロジック
    base_neg_sig = (gb < 0) and (gb_p / 2 < 0.05)        # 片側 p<0.05 で負
    attenuated = (abs(gg) < 0.5 * abs(gb)) and (gg_lo <= 0 <= gg_hi)
    print("\n--- 凍結合否（一次）---")
    print(f"  γ_base 有意に負?            : {base_neg_sig}")
    print(f"  γ_gex 0へ減衰(|γ_gex|<0.5|γ_base| & CI∋0)?: {attenuated}")
    if base_neg_sig and attenuated:
        verdict = "PASS（GEXがbandのレジーム較正に増分価値）"
    elif not base_neg_sig:
        verdict = "null（trailing単独で既に較正済 → GEXはbandに無価値。正直な許容結果）"
    else:
        verdict = "部分的/要検討（γ_base負だが γ_gex が十分減衰せず）"
    print(f"  >>> V-2a 一次判定: {verdict}")

    # ---- 二次: q=0.90 バンドの被覆 / pinball（k は全 OOS 学習窓側の経験分位点で固定）----
    # k_base / k_gex: OOS 期間に入る前の学習窓（最初の MIN_TRAIN 行）での RV/σ̂ の 0.90 分位点。
    # （test 非参照を厳守するため、最初の学習窓 1 つで k を固定し、以降不変。）
    warm = panel.iloc[:MIN_TRAIN]
    # warm 期間内 RV/σ̂（baseline/treatment）を leave-one-out 風でなく、warm 内 full-fit で近似
    Xw_b = warm[["const", "trailing"]].to_numpy(float)
    Xw_g = warm[["const", "trailing", "signed"]].to_numpy(float)
    yw = warm["logrv"].to_numpy(float)
    bw_b = np.linalg.lstsq(Xw_b, yw, rcond=None)[0]
    bw_g = np.linalg.lstsq(Xw_g, yw, rcond=None)[0]
    sig_w_b = np.sqrt(np.exp(Xw_b @ bw_b))
    sig_w_g = np.sqrt(np.exp(Xw_g @ bw_g))
    k_base = float(np.quantile(warm["rv"].to_numpy() / sig_w_b, COVERAGE_Q))
    k_gex = float(np.quantile(warm["rv"].to_numpy() / sig_w_g, COVERAGE_Q))

    B_base = k_base * oos["sigma_base"].to_numpy()
    B_gex = k_gex * oos["sigma_gex"].to_numpy()
    rv_o = oos["rv"].to_numpy()
    exc_base = float(np.mean(rv_o > B_base))
    exc_gex = float(np.mean(rv_o > B_gex))
    pin_base = float(np.mean(pinball(rv_o, B_base, COVERAGE_Q)))
    pin_gex = float(np.mean(pinball(rv_o, B_gex, COVERAGE_Q)))
    print(f"\n=== 二次: q={COVERAGE_Q} 片側バンド（k は学習窓固定: k_base={k_base:.3f}, k_gex={k_gex:.3f}）===")
    print(f"  逸脱率(RV>B) 名目{1-COVERAGE_Q:.0%}: baseline={exc_base:.3%}  treatment={exc_gex:.3%}")
    print(f"  pinball({COVERAGE_Q}) 損失      : baseline={pin_base:.5f}  treatment={pin_gex:.5f}  "
          f"(treatment<baseline なら H1 支持)")

    # signed_distance 五分位別の逸脱率（較正の透明性）
    oos = oos.assign(B_base=B_base, B_gex=B_gex)
    oos["q5"] = pd.qcut(oos["signed"], 5, labels=False)
    print("\n  五分位別 逸脱率(RV>B)  [signed昇順=ネガγ→ポジγ]")
    print("   q5 |   signed中央 | base_exc | gex_exc")
    for q in range(5):
        sub = oos[oos["q5"] == q]
        print(f"    {q}  | {sub['signed'].median():+.4f}  |  {np.mean(sub['rv']>sub['B_base']):.1%}   |  "
              f"{np.mean(sub['rv']>sub['B_gex']):.1%}")

    oos.to_csv(args.out, index=False)
    print(f"\n[out] per-session 出力 -> {args.out}（US500 由来 → .gitignore 推奨）")
    print("[note] 誤判断27: 本結果は実行環境依存。Windows/Python 3.14.4 実機で再実行して確定すること。")


if __name__ == "__main__":
    main()
