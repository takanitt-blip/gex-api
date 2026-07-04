#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_zero_gamma_rv.py  ─  統計検証フェーズ 最初の1検定（PC_VALIDATION §1.1 の1枠）

【事前登録された仮説（結果を見る前に確定）】
  ディーラーのγ符号則が SPY/US500 で機能するか、を土台レベルで直接検定する。
  地図(EOD T)が符号化する spot_T と Zero Gamma Z から regime を決め、
  その地図が統治する翌セッション K の実現ボラ(RV)を予測できるかを見る。

  H1: spot_T が Z を下回る(ネガγ)ほど、セッション K の実現ボラは高い。
      ⇔ 連続回帰 log(RV_K) ~ signed_distance + trailing_log_rv で
        signed_distance=(spot_T-Z)/spot_T の係数が【負】。
  H0: 係数 = 0（regime は RV を予測しない）。

  一次 = 連続回帰（near-Z ラベルジッタに頑健。閾値除外をしない＝カーブフィッティング回避）。
  伴走 = RV(neg) vs RV(pos) の Mann–Whitney（中央値、一方向 greater）。

【点時刻整合（obs.F 級の罠を踏まない）】
  分類子 spot_T, Z はいずれも EOD(T) 由来で、統治セッション K(=next_business_day(T))の
  開始前に既知。RV は K のセッション中に実現。同一セッションの終値で分類して同一
  セッションの RV を測る（同時性で因果汚染）ことは構造的にしない。

【データ規律】
  ・地図 = gex_history.json（v2+migration: data_quality=ok / z_position 実装済）。
    一次検定は z_position=="inside" のみ。非整序(above_call/below_put)は別掲・記述のみ。
    data_source!="rest_backfill_v2"（stale 窓端2日）は除外。
  ・価格 = US500 15分(MT5/OANDA export)。broker time = ET + 7h（実データで実証、DST安定）。
    RTH 09:30–16:00 ET の 26本(始値刻印)を満たす日のみ採用。<26本=早終了/不完全→除外。
    RV = RTH 15分対数リターンの二乗和（＝セッションの実現分散）。
  ・trailing_log_rv = 直近5本の full-RTH 日の log(RV) 平均（周囲のボラ水準を統制し、
    「高ボラ→spot下落→機械的Z割れ」の内生性を抑える）。
  ・留保: obs.C（0DTE の IV>500% 深OTM）は地図 levels 計算時点に焼き込まれており、
    分析段でフィルタ不可。levels への既知の留保として併記する。

  CI 非実行（tools/ 配下・手元実行）。出力 per-session CSV は US500 由来のため .gitignore 推奨。
  再現性: 既定窓 --start 2024-06-09 --end 2026-06-11（obs.J のスライド対策でピン留め）。
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 既定値（locked spec）
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAP = "gex_history.json"
DEFAULT_PRICE = "US500_M15_202406030100_202606122100.csv"
DEFAULT_START = "2024-06-09"
DEFAULT_END = "2026-06-11"
BROKER_MINUS_ET_HOURS = 7      # broker time = ET + 7h（実データで実証）
RTH_BARS = 26                  # 09:30..15:45 ET の 15分足(始値刻印)
TRAILING_N = 5                 # ambient vol 統制の窓
ANNUALIZE = 252                # 年率換算（表示用のみ）


# ─────────────────────────────────────────────────────────────────────────────
# 1. 地図の読み込み
# ─────────────────────────────────────────────────────────────────────────────
def load_map(path: str, start: str, end: str) -> pd.DataFrame:
    raw = json.loads(Path(path).read_text())
    rows = []
    for k, v in raw.items():
        rows.append(
            dict(
                date=pd.to_datetime(k, format="%Y.%m.%d"),
                data_quality=v.get("data_quality"),
                z_position=v.get("z_position"),
                data_source=v.get("data_source"),
                spot=v.get("underlying_price"),
                z=v.get("zero_gamma"),
            )
        )
    m = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    m = m[(m["date"] >= pd.Timestamp(start)) & (m["date"] <= pd.Timestamp(end))]
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 2. 価格の読み込み → ET化 → RTH窓 → 日次RV → trailing
# ─────────────────────────────────────────────────────────────────────────────
def _rth_slots() -> set[str]:
    # 09:30, 09:45, 10:00 ... 15:45 （始値刻印, 26本）
    out = []
    for h in range(9, 16):
        for mm in (0, 15, 30, 45):
            if h == 9 and mm < 30:
                continue
            out.append(f"{h:02d}:{mm:02d}")
    return set(out)


def load_price_rv(path: str, tz_offset: int, rth_bars: int, trailing: int) -> pd.DataFrame:
    p = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    p.columns = [c.strip("<>").lower() for c in p.columns]
    p["dt_broker"] = pd.to_datetime(p["date"] + " " + p["time"], format="%Y.%m.%d %H:%M:%S")
    p["dt_et"] = p["dt_broker"] - pd.Timedelta(hours=tz_offset)
    p["et_date"] = p["dt_et"].dt.normalize()
    p["et_hm"] = p["dt_et"].dt.strftime("%H:%M")

    rth = p[p["et_hm"].isin(_rth_slots())].copy().sort_values("dt_et")

    recs = []
    for d, g in rth.groupby("et_date"):
        g = g.sort_values("dt_et")
        n = len(g)
        # RTH 内 close-to-close 15分対数リターンの二乗和（= 実現分散）
        logret = np.log(g["close"].to_numpy())
        rv = float(np.sum(np.diff(logret) ** 2)) if n >= 2 else np.nan
        recs.append(dict(date=d, n_bars=n, rv=rv))
    px = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    px["full"] = px["n_bars"] == rth_bars

    # trailing は full-RTH 日のみで連続させる（半日RVは短窓で非比較→寄与させない）
    full = px[px["full"]].copy()
    full["log_rv"] = np.log(full["rv"])
    full["trailing_log_rv"] = full["log_rv"].rolling(trailing, min_periods=trailing).mean().shift(1)
    px = px.merge(full[["date", "log_rv", "trailing_log_rv"]], on="date", how="left")
    return px


# ─────────────────────────────────────────────────────────────────────────────
# 3. OLS + Newey-West HAC 標準誤差（statsmodels 非依存）
# ─────────────────────────────────────────────────────────────────────────────
def ols_hac(y: np.ndarray, X: np.ndarray, lag: int | None = None):
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta

    # classical SE
    sigma2 = float(e @ e) / (n - k)
    V_cls = sigma2 * XtX_inv
    se_cls = np.sqrt(np.diag(V_cls))

    # Newey-West HAC（Bartlett kernel）
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        Xe_t = (X[l:] * e[l:, None])
        Xe_tl = (X[:-l] * e[:-l, None])
        G = Xe_t.T @ Xe_tl
        S += w * (G + G.T)
    V_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diag(V_hac))

    # R^2
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return beta, se_cls, se_hac, r2, lag, n


def _z_p_two_sided(t: float) -> float:
    return 2.0 * stats.norm.sf(abs(t))


# ─────────────────────────────────────────────────────────────────────────────
# 4. レポート
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Zero-Gamma sign convention test (pre-registered).")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--price", default=DEFAULT_PRICE)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--tz-offset", type=int, default=BROKER_MINUS_ET_HOURS,
                    help="broker time minus ET, in hours (実証値=7)")
    ap.add_argument("--rth-bars", type=int, default=RTH_BARS)
    ap.add_argument("--trailing", type=int, default=TRAILING_N)
    ap.add_argument("--out", default="zg_rv_per_session.csv",
                    help="per-session 出力CSV（US500由来→.gitignore推奨）")
    args = ap.parse_args()

    m = load_map(args.map, args.start, args.end)
    px = load_price_rv(args.price, args.tz_offset, args.rth_bars, args.trailing)

    # ── 結合（inner: 地図 × 価格）。点時刻整合は date(=統治セッションK)で一致 ──
    df = m.merge(px, on="date", how="left")

    # 除外フラグ付与（理由をログに残す＝§1.6）
    def reason(r):
        if r["data_quality"] != "ok":
            return "data_error"
        if r["data_source"] != "rest_backfill_v2":
            return "stale_levels(pre-DTE-fix)"
        if pd.isna(r["rv"]):
            return "no_price"
        if not r["full"]:
            return "short_session(early_close/incomplete)"
        if pd.isna(r["trailing_log_rv"]):
            return "trailing_warmup"
        return ""  # 採用候補
    df["excl_reason"] = df.apply(reason, axis=1)

    df["signed_distance"] = (df["spot"] - df["z"]) / df["spot"] * 100.0  # %
    df["regime"] = np.where(df["spot"] < df["z"], "neg(Pr<Z)", "pos(Pr>=Z)")
    df["log_rv_y"] = df["log_rv"]
    df["ann_vol"] = np.sqrt(df["rv"] * ANNUALIZE)

    # ── 一次集合: inside かつ 除外理由なし ──
    primary = df[(df["z_position"] == "inside") & (df["excl_reason"] == "")].copy()
    noninside = df[(df["z_position"] != "inside") & (df["data_quality"] == "ok")].copy()

    print("=" * 78)
    print("Zero-Gamma sign convention — pre-registered test (PC_VALIDATION §1.1)")
    print("=" * 78)
    print(f"window           : {args.start} .. {args.end}")
    print(f"broker→ET offset : -{args.tz_offset}h   RTH bars required: {args.rth_bars}   trailing: {args.trailing}d")

    print("\n── 標本構成（除外理由つき, §1.6 ログ）─────────────────────────")
    counts = df["excl_reason"].replace("", "INCLUDED(candidate)").value_counts()
    for kk, vv in counts.items():
        print(f"  {kk:42s}: {vv}")
    print(f"  primary (inside & included)               : {len(primary)}")
    print(f"  non-inside (ok, 別掲・記述のみ)             : {len(noninside)}")
    print(f"  regime split (primary): {primary['regime'].value_counts().to_dict()}")

    # ── 一次: 連続回帰 log(RV) ~ signed_distance + trailing_log_rv ──
    print("\n── 一次検定: 連続回帰  log(RV_K) ~ signed_distance(%) + trailing_log_rv ──")
    y = primary["log_rv_y"].to_numpy()
    X = np.column_stack([
        np.ones(len(primary)),
        primary["signed_distance"].to_numpy(),
        primary["trailing_log_rv"].to_numpy(),
    ])
    beta, se_cls, se_hac, r2, lag, n = ols_hac(y, X)
    names = ["const", "signed_distance", "trailing_log_rv"]
    print(f"  n={n}   R^2={r2:.3f}   HAC lag(Newey-West)={lag}")
    print(f"  {'term':16s} {'coef':>11s} {'se_HAC':>10s} {'t_HAC':>8s} {'p_HAC(2s)':>11s} {'95%CI_HAC':>22s}")
    for i, nm in enumerate(names):
        t = beta[i] / se_hac[i]
        p = _z_p_two_sided(t)
        lo, hi = beta[i] - 1.96 * se_hac[i], beta[i] + 1.96 * se_hac[i]
        print(f"  {nm:16s} {beta[i]:11.4f} {se_hac[i]:10.4f} {t:8.2f} {p:11.2e}   [{lo:8.4f},{hi:8.4f}]")
    b_sd = beta[1]
    t_sd = beta[1] / se_hac[1]
    print(f"\n  >>> 仮説判定: signed_distance の係数 = {b_sd:+.4f}  (期待: 負)")
    print(f"      片側p(coef<0, HAC) = {stats.norm.cdf(t_sd):.3e}   "
          f"{'■ H1 を支持（spotがZより下ほど高RV）' if (b_sd < 0 and stats.norm.cdf(t_sd) < 0.05) else '□ 有意な負の効果は確認されず'}")

    # ── 伴走: Mann–Whitney（RV(neg) > RV(pos)）──
    print("\n── 伴走検定: Mann–Whitney  RV(neg) vs RV(pos), 一方向 greater ──")
    rv_neg = primary.loc[primary["regime"] == "neg(Pr<Z)", "rv"].to_numpy()
    rv_pos = primary.loc[primary["regime"] == "pos(Pr>=Z)", "rv"].to_numpy()
    U, pmw = stats.mannwhitneyu(rv_neg, rv_pos, alternative="greater")
    rb = 1.0 - 2.0 * U / (len(rv_neg) * len(rv_pos))  # rank-biserial（符号反転で neg>pos が正）
    rb = -rb
    print(f"  median ann.vol  neg={np.sqrt(np.median(rv_neg)*ANNUALIZE):.2%}  "
          f"pos={np.sqrt(np.median(rv_pos)*ANNUALIZE):.2%}")
    print(f"  U={U:.0f}  p(one-sided greater)={pmw:.3e}  rank-biserial(neg>pos)={rb:+.3f}")

    # ── 単調性の可視化: signed_distance 五分位ごとの RV ──
    print("\n── 単調性チェック: signed_distance 五分位 × 実現ボラ ──")
    q = pd.qcut(primary["signed_distance"], 5)
    tab = primary.groupby(q, observed=True).agg(
        n=("rv", "size"),
        sd_mid=("signed_distance", "median"),
        ann_vol_median=("ann_vol", "median"),
    ).reset_index(drop=True)
    print(f"  {'quintile(signed_dist %)':26s} {'n':>4s} {'sd_median':>10s} {'annvol_median':>14s}")
    for _, r in tab.iterrows():
        print(f"  {'':26s} {int(r['n']):4d} {r['sd_mid']:10.3f} {r['ann_vol_median']:14.2%}")

    # ── 非整序6日（別掲・記述のみ, プールしない）──
    print("\n── 非整序日（z_position!=inside, 別掲・記述のみ）──")
    if len(noninside):
        for _, r in noninside.sort_values("date").iterrows():
            av = f"{r['ann_vol']:.2%}" if pd.notna(r["ann_vol"]) else "no_price/short"
            print(f"  {r['date'].date()}  {r['z_position']:10s}  signed_dist={r['signed_distance']:+.3f}%  "
                  f"regime={r['regime']:10s}  annvol={av}")
    else:
        print("  （なし）")

    print("\n── 留保 ─────────────────────────────────────────────────────")
    print("  ・obs.C: 0DTE の IV>500% 深OTM が地図 levels 計算に焼き込まれている可能性")
    print("    （分析段で除去不可。spot_T/Z への影響は残留する既知の留保）。")
    print("  ・median signed_distance<0 はプットスキュー由来の構造（Zがspot上に座る）。")
    print("    二値分割は一部構造的→一次は連続回帰で解釈する。")

    # ── per-session 出力（.gitignore 推奨）──
    cols = ["date", "z_position", "data_source", "spot", "z", "signed_distance",
            "regime", "rv", "ann_vol", "log_rv_y", "trailing_log_rv", "n_bars",
            "full", "excl_reason"]
    df[cols].sort_values("date").to_csv(args.out, index=False)
    print(f"\n[written] {args.out}  （US500由来 → .gitignore 推奨）")


if __name__ == "__main__":
    sys.exit(main())
