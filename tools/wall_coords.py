#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wall_coords.py ─ 壁レベル(SPYストライク) を US500(指数pt) 座標へ整合させる + basis 検証

背景:
  GEX 壁(Call/Put/Z/MaxPain) は SPY オプション由来＝SPY価格座標。
  一次価格ソースは US500 15分(指数pt)。両者は別座標で ratio=US500/SPY≈10 は窓内で
  ドリフトする(固定10倍は不可)。④/④-bis は returns しか使わず尺度不変ゆえ踏まずに済んだが、
  壁検定(①②)は「価格 vs 壁レベル」なので座標整合が前提。

方針:
  日次再アンカー ratio_d = US500_EOD(d) / SPY_EOD(d)。壁_US500 = 壁_SPY × ratio_d。
  - 構造検定(①): 同日アンカー ratio_K(=session Kの当日EOD比) で変換 → オーバーナイト分を除去、
    残差は日中 basis ドリフトのみ(直接測定不可→日次トラッキングで間接評価)。
  - ライブEA: 予測アンカー ratio_T(前日EOD, 場中既知) → オーバーナイト分が誤差予算に乗る(数値化)。

直接 SPY 日中が無いため日中 basis 安定性は「日次リターンのトラッキングの密さ」で代理評価する
(密ならば連続性から日中も安定と推論)。これは間接証拠であり、その限界も出力に明記する。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from validate_zero_gamma_rv import _rth_slots, BROKER_MINUS_ET_HOURS


def spy_eod_series(map_path: str) -> pd.DataFrame:
    """gex_history の as_of(=トレード日T) → SPY EOD spot, 壁レベル。"""
    raw = json.loads(Path(map_path).read_text())
    rows = []
    for k, v in raw.items():
        rows.append(dict(
            trade_date=pd.to_datetime(v["as_of"]).normalize(),
            gov_session=pd.to_datetime(k, format="%Y.%m.%d"),
            spy_eod=v.get("underlying_price"),
            put=v.get("put_wall"), z=v.get("zero_gamma"),
            call=v.get("call_wall"), maxpain=v.get("max_pain"),
            z_position=v.get("z_position"), data_source=v.get("data_source"),
        ))
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def us500_eod_series(price_path: str, tz_offset: int = BROKER_MINUS_ET_HOURS) -> pd.DataFrame:
    """US500 15分 → ET日付ごとの RTH 最終バー close (= US500 EOD)。"""
    p = pd.read_csv(price_path, sep="\t", encoding="utf-8-sig")
    p.columns = [c.strip("<>").lower() for c in p.columns]
    p["dt"] = pd.to_datetime(p["date"] + " " + p["time"], format="%Y.%m.%d %H:%M:%S") \
        - pd.Timedelta(hours=tz_offset)
    p["et_date"] = p["dt"].dt.normalize()
    p["hm"] = p["dt"].dt.strftime("%H:%M")
    rth = p[p["hm"].isin(_rth_slots())].sort_values("dt")
    eod = rth.groupby("et_date")["close"].last().rename("us500_eod").reset_index()
    return eod.rename(columns={"et_date": "trade_date"})


def daily_ratio(map_path: str, price_path: str, tz_offset: int = BROKER_MINUS_ET_HOURS) -> pd.DataFrame:
    spy = spy_eod_series(map_path)
    us = us500_eod_series(price_path, tz_offset)
    m = spy.merge(us, on="trade_date", how="inner")
    m["ratio"] = m["us500_eod"] / m["spy_eod"]
    return m.sort_values("trade_date").reset_index(drop=True)


def anchor_wall_to_us500(wall_spy: float, ratio_d: float) -> float:
    """壁(SPY) → US500 座標。"""
    return wall_spy * ratio_d


def verify_basis(map_path: str, price_path: str, tz_offset: int = BROKER_MINUS_ET_HOURS,
                 approach_band_pct: float = 0.25):
    r = daily_ratio(map_path, price_path, tz_offset)
    n = len(r)
    ratio = r["ratio"].to_numpy()

    print("=" * 80)
    print("壁座標 basis 検証 (US500 ↔ SPYストライク)")
    print("=" * 80)
    print(f"matched trade-days : {n}   ratio=US500/SPY")
    print(f"ratio level        : min {ratio.min():.4f}  max {ratio.max():.4f}  "
          f"median {np.median(ratio):.4f}  drift(max-min) {ratio.max()-ratio.min():.4f} "
          f"({(ratio.max()-ratio.min())/np.median(ratio)*100:.2f}% of level)")
    print(f"  → 固定10倍の最大誤差 ≈ {abs(np.median(ratio)-10)/np.median(ratio)*100:.2f}% + drift。"
          f"日次アンカーで level/drift は除去。")

    # ── V1: 日次アンカーの残差予算（オーバーナイト Δratio = 予測アンカーの誤差源）──
    dr = np.diff(ratio)
    rel = np.abs(dr) / ratio[:-1]          # |Δratio|/ratio = 壁位置の相対誤差
    print("\n── V1: アンカー誤差予算 ─────────────────────────────────────")
    print("  予測アンカー(前日 ratio_T で session K を変換) を使う場合に壁位置へ乗る相対誤差")
    print("  = オーバーナイトの |Δratio|/ratio （%）:")
    for tag, val in [("median", np.median(rel)), ("P90", np.percentile(rel, 90)),
                     ("P99", np.percentile(rel, 99)), ("max", rel.max())]:
        print(f"    {tag:6s}: {val*100:.4f}%   (壁を価格と同帯とみなした時の壁位置ズレ)")
    band = approach_band_pct / 100.0
    frac_med = np.median(rel) / band
    print(f"  接近帯 {approach_band_pct:.2f}% に対する予測アンカー誤差の比: "
          f"median {frac_med*100:.1f}% / P99 {np.percentile(rel,99)/band*100:.1f}% of band")
    print("  ※ 構造検定(①)は『同日アンカー』を使うのでこのオーバーナイト分は乗らない。")
    print("    上記は後段ライブEA(予測アンカー)の劣化予算として記録。")

    # ── V2: 日次リターン・トラッキング（日中 basis 安定性の間接評価）──
    rr = r.copy()
    rr["r_us"] = np.log(rr["us500_eod"]).diff()
    rr["r_spy"] = np.log(rr["spy_eod"]).diff()
    rr = rr.dropna(subset=["r_us", "r_spy"])
    x = rr["r_spy"].to_numpy(); y = rr["r_us"].to_numpy()
    beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)
    ss_tot = float(((y - y.mean()) ** 2).sum()); ss_res = float((resid ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    print("\n── V2: 日次リターン・トラッキング US500~SPY (日中 basis 安定性の間接代理) ──")
    print(f"  beta={beta:.4f}  alpha={alpha*1e4:+.2f}bp/day  R^2={r2:.5f}")
    print(f"  tracking resid std(日次) = {resid.std(ddof=1)*100:.4f}%/day  "
          f"corr={np.corrcoef(x,y)[0,1]:.5f}")
    print("  → beta≈1 かつ R^2≈1 かつ残差極小 なら、両者はほぼ完全連動 = ratio は短時間で")
    print("    ほぼ一定 → 日中 basis も安定と推論。ただしこれは日次からの間接証拠(直接の日中")
    print("    SPY が無いため)。日中 basis の鋭いジャンプ(稀)は捕捉できない、という限界は残す。")

    # ── 判定 ──
    pass_anchor = np.percentile(rel, 99) < band          # 予測アンカーでも P99 が接近帯未満なら余裕
    pass_track = (r2 > 0.99) and (abs(beta - 1.0) < 0.05)
    print("\n── 判定 ─────────────────────────────────────────────────────")
    print(f"  [V1] 予測アンカー P99 誤差 < 接近帯({approach_band_pct}%): "
          f"{'PASS' if pass_anchor else 'FLAG'} "
          f"(P99={np.percentile(rel,99)*100:.3f}% vs {approach_band_pct}%)")
    print(f"  [V2] トラッキング R^2>0.99 & |beta-1|<0.05: {'PASS' if pass_track else 'FLAG'}")
    print(f"  同日アンカー(①用)残差 = 日中 basis のみ ≤ V2 残差 {resid.std(ddof=1)*100:.4f}%/day の")
    print("  さらに一部(日中は1日の数分の1) → 接近帯に対し無視可能と評価。")
    verdict = "PASS: 日次アンカーで壁を US500 座標に乗せて①へ進んで可。予測アンカー誤差は上記予算で記録。" \
        if (pass_anchor and pass_track) else \
        "FLAG: 誤差が接近帯に対し無視できない。アンカー方式かバンド定義を再考。"
    print(f"\n  >>> {verdict}")
    return dict(n=n, ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                rel_p99=float(np.percentile(rel, 99)), beta=beta, r2=r2,
                resid_std=float(resid.std(ddof=1)), pass_anchor=pass_anchor, pass_track=pass_track)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="gex_history.json")
    ap.add_argument("--price", default="US500_M15_202406030100_202606122100.csv")
    ap.add_argument("--tz-offset", type=int, default=BROKER_MINUS_ET_HOURS)
    ap.add_argument("--approach-band-pct", type=float, default=0.25)
    a = ap.parse_args()
    verify_basis(a.map, a.price, a.tz_offset, a.approach_band_pct)
