#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_v2a_prime_asymmetric_band.py  --  優位性検証 V-2a'（非対称バンド・マッピング）

================================  事前登録（凍結）  ================================
研究設問:
  非対称(ヒンジ@Z)のGEXバンド・マッピング ── ネガγ側で急な vol 予報 ＋ regime別バンド乗数 ──
  は、線形V-2aが達成できなかった「レジーム横断の較正」を達成するか。

動機の正直な開示（重要）:
  本検定の“非対称形”は V-2a で観察した非対称ミス較正に動機づけられている（後出しの危険）。
  カーブフィッティングを封じる固定:
    (i)  ヒンジ折れ点 = signed_distance=0 = ディーラーγ符号反転点（理論。データ探索しない）。
    (ii) 追加基底は neg_part=min(signed,0) の1本のみ。次数/knot探索なし（④「増幅は深さ比例」由来の最小形）。
    (iii)全係数・regime別バンド乗数 k は walk-forward で過去データのみから推定。
    (iv) 1回実行・凍結基準で判定・反復しない（null は null）。
    (v)  最強の確証は別銘柄/将来データの OOS（現データに無いため保留と明記）。

モデル（walk-forward OOS、拡張窓 最小60）:
  M0 trailing単独 : logRV ~ 1 + trailing                          （GEX無しの基準、γ0）
  M1 線形GEX(V-2a): logRV ~ 1 + trailing + signed                 （比較用、γ1）
  M2 非対称GEX(本) : logRV ~ 1 + trailing + signed + neg_part      （neg_part=min(signed,0)、γ2）
  σ̂ = sqrt(exp(ŷ))、標準化残差 r = log(RV/σ̂)。

一次（較正）: OOS の r を signed_distance に HAC回帰 → γ0, γ1, γ2。
  PASS: |γ2| < 0.5·|γ0| かつ γ2 の95%CIが0を含む
        （＝非対称形が、線形(M1)では消せなかったレジーム依存の平均ミス較正を除去）。
        ※ M1 は V-2a で |γ1|=9.66 > 0.5|γ0|=6.74 で不合格だった。同じバーで判定。

二次（本来のバンド目標）: regime別ローリング k。
  各OOS日: 学習窓(panel[:i])にM2をfit→学習窓内 RV/σ̂ を regime(signed符号)別に分け、
           各 regime の 0.90分位点を k_neg / k_pos とする（過去データのみ、test非参照）。
           バンド B = k_regime · σ̂(M2)。
  較正度: signed_distance 五分位別の逸脱率(RV>B)を名目10%と比較。
          指標 = max_q |逸脱率_q − 0.10|。
          PASS-二次（支持）: 全五分位が [0%,20%] 内（max偏差 ≤ 0.10）かつ V-2a(0.27)より明確改善。
  併せて全体逸脱率（ローリングkで~10%が期待）と pinball(0.90) を報告。

null: γ2 がなお強く負、または五分位がなお大きく外れる → ヒンジでは較正不足。
       「非対称ヒンジでは較正できず、より richer なモデルは“再事前登録”が必要」と記録（今回は項を足さない）。

標本・価格・RV 定義・期間ピンは V-2a（validate_v2a_band_calibration.py）と同一。
依存 numpy/pandas/scipy のみ。CI非実行。出力 CSV は US500由来 → .gitignore 推奨。
=================================================================================
"""

import argparse
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

BROKER_MINUS_ET_HOURS = 7
RTH_START = (9, 30)
RTH_END = (16, 0)
RTH_BARS_REQUIRED = 26
TRAILING_N = 5
MIN_TRAIN = 60
COVERAGE_Q = 0.90
PIN_RANGE_DEFAULT = ("2024-06-09", "2026-06-11")


def load_us500_rv(price_csv):
    df = pd.read_csv(price_csv, sep="\t", engine="python",
                     names=["date", "time", "open", "high", "low", "close",
                            "tickvol", "vol", "spread"], header=0)
    bdt = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    et = bdt - pd.Timedelta(hours=BROKER_MINUS_ET_HOURS)
    df = df.assign(et=et, close=pd.to_numeric(df["close"], errors="coerce"))
    df = df.dropna(subset=["close"]).sort_values("et").reset_index(drop=True)
    et_minutes = df["et"].dt.hour * 60 + df["et"].dt.minute
    start_m, end_m = RTH_START[0] * 60 + RTH_START[1], RTH_END[0] * 60 + RTH_END[1]
    rth = df[(et_minutes >= start_m) & (et_minutes < end_m)].copy()
    rth["et_date"] = rth["et"].dt.date
    rows = []
    for d, g in rth.groupby("et_date", sort=True):
        g = g.sort_values("et")
        if len(g) != RTH_BARS_REQUIRED:
            rows.append((d, np.nan, len(g))); continue
        logc = np.log(g["close"].to_numpy())
        rets = np.diff(logc)
        rows.append((d, float(np.sqrt(np.sum(rets ** 2))), len(g)))
    return pd.DataFrame(rows, columns=["et_date", "rv", "n_bars"]).set_index("et_date")


def load_map(gex_json):
    with open(gex_json, encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for key, v in d.items():
        try:
            kd = datetime.strptime(key, "%Y.%m.%d").date()
        except ValueError:
            continue
        rows.append({"session_date": kd, "data_quality": v.get("data_quality"),
                     "z_position": v.get("z_position"),
                     "underlying_price": v.get("underlying_price"),
                     "zero_gamma": v.get("zero_gamma")})
    m = pd.DataFrame(rows).sort_values("session_date").reset_index(drop=True)
    m["signed_distance"] = (m["underlying_price"] - m["zero_gamma"]) / m["underlying_price"]
    return m


def ols_hac(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=n - k))
    return {"beta": beta, "se": se, "t": tval, "p": pval, "n": n, "lag": L,
            "ci95": np.vstack([beta - 1.96 * se, beta + 1.96 * se]).T}


def fit_predict(train, test, cols):
    Xtr = train[cols].to_numpy(float)
    ytr = train["logrv"].to_numpy(float)
    beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    yhat = float(np.array([test[c] for c in cols]) @ beta)
    return math.sqrt(math.exp(yhat)), beta


def pinball(actual, forecast, q):
    d = actual - forecast
    return np.where(d >= 0, q * d, (q - 1) * d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gex", default="gex_history.json")
    ap.add_argument("--price", default="US500_M15_202406030100_202606122100.csv")
    ap.add_argument("--start", default=PIN_RANGE_DEFAULT[0])
    ap.add_argument("--end", default=PIN_RANGE_DEFAULT[1])
    ap.add_argument("--all-zpos", action="store_true")
    ap.add_argument("--out", default="v2a_prime_per_session.csv")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    rv = load_us500_rv(args.price)
    m = load_map(args.gex)
    # マージキーを datetime64 に正規化（pandas 3.0 では object型 datetime.date 同士が
    # 値一致でも merge で突合しないため。両側を datetime64 に揃える＝根本修正）。
    m["session_date"] = pd.to_datetime(m["session_date"])

    full = rv.dropna(subset=["rv"]).copy()
    full = full[full["rv"] > 0].sort_index()
    full["logrv"] = np.log(full["rv"])
    full["trailing"] = full["logrv"].shift(1).rolling(TRAILING_N).mean()

    m2 = m[m["data_quality"] == "ok"].copy()
    if not args.all_zpos:
        m2 = m2[m2["z_position"] == "inside"]
    m2 = m2.dropna(subset=["signed_distance"])
    m2 = m2[(m2["session_date"] >= start) & (m2["session_date"] <= end)]

    fr = full.reset_index().rename(columns={"et_date": "session_date"})
    fr["session_date"] = pd.to_datetime(fr["session_date"])
    panel = m2.merge(fr[["session_date", "logrv", "trailing", "rv"]],
                     on="session_date", how="inner").dropna(
        subset=["logrv", "trailing", "signed_distance"])
    panel = panel.sort_values("session_date").reset_index(drop=True)
    panel = panel.rename(columns={"signed_distance": "signed"})
    panel["const"] = 1.0
    panel["neg_part"] = np.minimum(panel["signed"], 0.0)   # ヒンジ@0（理論固定）

    n = len(panel)
    print(f"[sample] merged sessions = {n}  (inside={'no(all)' if args.all_zpos else 'yes'}, "
          f"{args.start}..{args.end})")
    if n == 0:
        print("[FATAL] merged=0。考えられる原因:")
        if not m["z_position"].notna().any():
            print("  - gex_history.json に z_position が無い（z_position 移行“前”の古い版）。")
            print("    移行後の版（data_quality 全 ok・z_position あり）で再生成/差し替えを。")
        print("  - もしくは data_quality / 期間ピン / 価格カバレッジ。diag_v2a2.py で各段の件数を確認。")
        return

    COLS = {
        "M0_trailing": ["const", "trailing"],
        "M1_linear":   ["const", "trailing", "signed"],
        "M2_asym":     ["const", "trailing", "signed", "neg_part"],
    }

    recs = []
    for i in range(MIN_TRAIN, n):
        train = panel.iloc[:i]
        test = panel.iloc[i]
        out = {"session_date": test["session_date"], "signed": test["signed"], "rv": test["rv"]}
        sigs = {}
        for name, cols in COLS.items():
            sig, _ = fit_predict(train, test, cols)
            sigs[name] = sig
            out[f"sigma_{name}"] = sig
            out[f"r_{name}"] = math.log(test["rv"] / sig)
        # 二次: M2 で regime別ローリング k（学習窓 in-window fit、過去データのみ）
        Xtr = train[COLS["M2_asym"]].to_numpy(float)
        ytr = train["logrv"].to_numpy(float)
        beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        sig_tr = np.sqrt(np.exp(Xtr @ beta))
        ratio_tr = train["rv"].to_numpy() / sig_tr
        neg_mask = train["signed"].to_numpy() < 0
        k_neg = float(np.quantile(ratio_tr[neg_mask], COVERAGE_Q)) if neg_mask.sum() >= 10 else np.nan
        k_pos = float(np.quantile(ratio_tr[~neg_mask], COVERAGE_Q)) if (~neg_mask).sum() >= 10 else np.nan
        k_use = k_neg if test["signed"] < 0 else k_pos
        out["k_use"] = k_use
        out["B_asym"] = k_use * sigs["M2_asym"]
        recs.append(out)
    oos = pd.DataFrame(recs)

    # ---- 一次: r ~ signed の HAC 傾き ----
    Xo = np.column_stack([np.ones(len(oos)), oos["signed"].to_numpy()])
    print("\n=== 一次: 標準化残差 r=log(RV/σ̂) の signed への傾き（OOS, HAC）===")
    gammas = {}
    for name in ["M0_trailing", "M1_linear", "M2_asym"]:
        fit = ols_hac(oos[f"r_{name}"].to_numpy(), Xo)
        b, se, t = fit["beta"][1], fit["se"][1], fit["t"][1]
        lo, hi = fit["ci95"][1]
        gammas[name] = (b, lo, hi)
        print(f"  {name:12s}: slope={b:+.3f}  se={se:.3f}  t={t:+.2f}  95%CI=[{lo:+.3f},{hi:+.3f}]")

    g0 = gammas["M0_trailing"][0]
    g2, g2lo, g2hi = gammas["M2_asym"]
    pass_primary = (abs(g2) < 0.5 * abs(g0)) and (g2lo <= 0 <= g2hi)
    print("\n--- 凍結合否（一次）---")
    print(f"  基準 |0.5·γ0| = {0.5*abs(g0):.3f}  /  |γ2(非対称)| = {abs(g2):.3f}  /  γ2 CI∋0? = {g2lo<=0<=g2hi}")
    print(f"  >>> 一次PASS（非対称が線形で消せなかったミス較正を除去）? : {pass_primary}")

    # ---- 二次: regime別ローリング k のバンド被覆 ----
    rv_o = oos["rv"].to_numpy()
    B = oos["B_asym"].to_numpy()
    valid = ~np.isnan(B)
    exc_all = float(np.mean(rv_o[valid] > B[valid]))
    pin = float(np.mean(pinball(rv_o[valid], B[valid], COVERAGE_Q)))
    print(f"\n=== 二次: regime別ローリング k のバンド（M2・k_neg/k_pos）===")
    print(f"  全体逸脱率(RV>B) 名目{1-COVERAGE_Q:.0%}: {exc_all:.3%}   pinball({COVERAGE_Q})={pin:.5f}")

    ov = oos[valid].copy()
    ov["q5"] = pd.qcut(ov["signed"], 5, labels=False)
    print("  五分位別 逸脱率(RV>B)  [signed昇順=ネガγ→ポジγ]")
    devs = []
    for q in range(5):
        sub = ov[ov["q5"] == q]
        e = float(np.mean(sub["rv"] > sub["B_asym"]))
        devs.append(abs(e - 0.10))
        print(f"    q{q} signed中央{sub['signed'].median():+.4f} : 逸脱 {e:.1%}")
    max_dev = max(devs)
    pass_secondary = (max_dev <= 0.10)
    print(f"\n--- 凍結合否（二次）---")
    print(f"  五分位の最大|逸脱−10%| = {max_dev:.1%}  (V-2aは0.27)  → 全分位[0,20%]内? : {pass_secondary}")

    print("\n=== 総合 ===")
    if pass_primary and pass_secondary:
        v = "PASS（非対称マッピングでGEXバンドが較正可能）"
    elif (not pass_primary) and max_dev > 0.10:
        v = "null（ヒンジでは較正不足 → richerは“再事前登録”。今回は項を足さない）"
    else:
        v = "部分的（一次/二次の片方のみ改善。判断は実機確認後）"
    print(f"  V-2a' 判定: {v}")

    oos.to_csv(args.out, index=False)
    print(f"\n[out] -> {args.out}（US500由来 → .gitignore 推奨）")
    print("[note] 誤判断27: 実行環境依存。Windows/Python 3.14.4 実機で再実行して確定。")
    print("[note] 後出し動機の検定。最強の確証は別銘柄/将来データの OOS（現データに無く保留）。")


if __name__ == "__main__":
    main()
