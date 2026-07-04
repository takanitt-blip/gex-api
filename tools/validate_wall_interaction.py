#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_wall_interaction.py ─ 優位性① 壁インタラクション（pre-registered, amended）

PRE-REGISTRATION（freeze before running, §1.1）

■ 目的 / 位置づけ
  レジーム・モデルの壁エッジを検定し EA Step 1B のエントリ規則へ写像。
  H-A primary / H-B・H-C secondary（多重性: この順で主従固定）。

■ 仮説
  H-A: narrow(inside & spot_T≥Z) で Call Wall は上限。近接ほど反落(フェード短)。突破幅は小。
  H-B: violent(inside & spot_T<Z) で Put 突破幅 > Call 突破幅（ボラ正規化後の非対称・真空）。
  H-C: Put break 後 reclaim の前方リターン>0（買い, exploratory, n小）。

■ 質問の型
  壁=予測的(EOD T)。接触/突破/回帰=session K 日中(Q2)。座標=同日アンカー ratio_K。
  look-ahead: 壁は EOD T 由来。ライブEA は ratio_T+ex-div（前段確定）。

■ データ
  壁=gex_history(EOD T) を ratio_K で US500 座標へ(wall_coords)。価格=US500 15分 RTH OHLC。
  標本=④/④-bis と同一 primary 488 = narrow223 / violent265。

■ ★H-A 主モデル（前半→後半 因果分割。バンドフリー。汎用MR統制）
  open=寄付き(09:30)。前半=bar[0:s], 正午=close[s-1], 後半=正午→EOD(と +4本)。
  D       = (Call_US500 − open)/open         （構造距離, 寄付きで既知）
  exc_am  = (max high[0:s] − open)/open       （朝の上昇幅＝汎用MR駆動因, 統制）
  prox_am = D − exc_am = (Call − 朝高値)/open  （朝の壁近接, <0=朝に突破）
  回帰: fwd_pm ~ γ1·exc_am + γ2·prox_am   (1日1obs, HAC)
  H-A判定: γ2>0 片側有意（壁近接で追加反落＝汎用MR統制後の壁固有効果）。
           γ1<0 期待(汎用MR)。 split s∈{10,13,16}・地平{4本,EOD} はグリッドで頑健性。
  ※ 事後 max は使わない（朝→午後で因果分離）。exc_am と D は独立変動で γ2 を汚染なく識別。

■ H-B（secondary, 突破幅 非対称・ボラ正規化）
  Call extent=(max close − Call)/Call (narrow break日)、Put extent=(Put − min close)/Put (violent break日)。
  正規化 extent_norm = extent / √rv（その日のボラ）で④のレジーム・ボラ差を除去。
  Mann–Whitney 片側(Put>Call) + 日ブートで median 差。H-B判定: Put extent_norm > Call。

■ H-C（secondary, exploratory, reclaim 買い）
  Put break→reclaim(終値が Put 内へ復帰)足から前方リターン(4本/EOD)。Wilcoxon 片側(>0)。
  violent レジームの baseline 同地平前方リターンとの差も併記。n=40 の検出力限界を明記。

■ 推論 / スコープ
  1日1壁1イベント＝日単位。方向は片側。HAC/日ブート。primary=H-A のみ (d)/Step1B 必須ゲート。
  コスト無し(後段)。妥当性の生命線=H-A の exc_am 統制 / H-B のボラ正規化。
  intraday US500・ex-div 留保。reclaim 頻度は標本サイジング用途に限定（H-B/H-C の証拠に使わない）。

■ 実装規約
  wall_coords + ④-bis 分類 + 新規 intraday-OHLC loader を import。CI 非実行。
  出力 per-event CSV は US500 由来→ .gitignore。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from validate_zero_gamma_rv import load_price_rv, _rth_slots, ols_hac, BROKER_MINUS_ET_HOURS
from wall_coords import daily_ratio

SPLITS = (10, 13, 16)
N_BOOT = 5000
BOOT_SEED = 42
EXPECTED = dict(n=488, narrow=223, violent=265)


def load_intraday_ohlc(price_path: str, tz_offset: int = BROKER_MINUS_ET_HOURS) -> pd.DataFrame:
    p = pd.read_csv(price_path, sep="\t", encoding="utf-8-sig")
    p.columns = [c.strip("<>").lower() for c in p.columns]
    p["dt"] = pd.to_datetime(p["date"] + " " + p["time"], format="%Y.%m.%d %H:%M:%S") \
        - pd.Timedelta(hours=tz_offset)
    p["et_date"] = p["dt"].dt.normalize()
    p["hm"] = p["dt"].dt.strftime("%H:%M")
    rth = p[p["hm"].isin(_rth_slots())].sort_values("dt")
    recs = []
    for d, g in rth.groupby("et_date"):
        g = g.sort_values("dt")
        recs.append(dict(date=d, n_bars=len(g), opens=g["open"].to_numpy(),
                         highs=g["high"].to_numpy(), lows=g["low"].to_numpy(),
                         closes=g["close"].to_numpy()))
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)


def _one_sided_p(t: float, direction: str) -> float:
    return float(stats.norm.sf(t)) if direction == "greater" else float(stats.norm.cdf(t))


def build_primary(map_path, price_path, start, end):
    raw = json.loads(Path(map_path).read_text())
    rows = [dict(date=pd.to_datetime(k, format="%Y.%m.%d"), spot=v["underlying_price"],
                 z=v["zero_gamma"], call=v["call_wall"], put=v["put_wall"],
                 z_position=v["z_position"], data_source=v["data_source"],
                 data_quality=v["data_quality"]) for k, v in raw.items()]
    m = pd.DataFrame(rows)
    m = m[(m.date >= pd.Timestamp(start)) & (m.date <= pd.Timestamp(end))]
    px = load_price_rv(price_path, BROKER_MINUS_ET_HOURS, 26, 5)
    ohlc = load_intraday_ohlc(price_path)
    rat = daily_ratio(map_path, price_path).rename(columns={"trade_date": "date"})[["date", "ratio"]]
    df = (m.merge(px[["date", "rv", "full", "trailing_log_rv"]], on="date", how="left")
            .merge(ohlc, on="date", how="left").merge(rat, on="date", how="left"))
    incl = ((df.data_quality == "ok") & (df.data_source == "rest_backfill_v2")
            & df.rv.notna() & df.full & df.trailing_log_rv.notna() & df.ratio.notna())
    prim = df[(df.z_position == "inside") & incl].copy().sort_values("date").reset_index(drop=True)
    prim["subregime"] = np.where(prim.spot < prim.z, "violent", "narrow")
    return prim


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="/mnt/project/gex_history.json")
    ap.add_argument("--price", default="/mnt/project/US500_M15_202406030100_202606122100.csv")
    ap.add_argument("--start", default="2024-06-09")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--out", default="wall_interaction_per_event.csv")
    ap.add_argument("--no-gate-assert", action="store_true")
    a = ap.parse_args()

    prim = build_primary(a.map, a.price, a.start, a.end)
    narrow = prim[prim["subregime"] == "narrow"].reset_index(drop=True)
    violent = prim[prim["subregime"] == "violent"].reset_index(drop=True)

    print("=" * 84)
    print("優位性① 壁インタラクション（pre-registered, amended）")
    print("=" * 84)
    print(f"primary {len(prim)} = narrow {len(narrow)} / violent {len(violent)}  (期待 488/223/265)")
    if not a.no_gate_assert:
        assert (len(prim), len(narrow), len(violent)) == (EXPECTED["n"], EXPECTED["narrow"], EXPECTED["violent"]), \
            f"[GATE] 標本不一致 {len(prim)}/{len(narrow)}/{len(violent)}"
        print("  [GATE ok] ④/④-bis と同一標本")

    # ════════════════ H-A primary ════════════════
    print("\n" + "─" * 84)
    print("H-A (primary): narrow で Call Wall 近接→反落  fwd_pm ~ γ1·exc_am + γ2·prox_am  (HAC)")
    print("─" * 84)

    def ha_rows(s):
        out = []
        for _, r in narrow.iterrows():
            cl, hi, op = r.closes, r.highs, r.opens[0]
            if len(cl) < 26:
                continue
            call_us = r.call * r.ratio
            D = (call_us - op) / op
            exc_am = (hi[:s].max() - op) / op
            prox_am = D - exc_am
            noon = cl[s - 1]
            fwd_eod = (cl[25] - noon) / noon
            fwd_4 = (cl[min(s - 1 + 4, 25)] - noon) / noon
            out.append((exc_am, prox_am, D, fwd_eod, fwd_4))
        return np.array(out)

    print(f"  {'split':>5s} {'horizon':>7s} {'γ1(exc,MR)':>12s} {'γ2(prox,壁)':>13s} "
          f"{'t(γ2)':>7s} {'片側p(γ2>0)':>12s} {'corr(exc,prox)':>15s} {'n':>4s}")
    ha_primary = None
    for s in SPLITS:
        M = ha_rows(s)
        exc, prox = M[:, 0], M[:, 1]
        cc = float(np.corrcoef(exc, prox)[0, 1])
        for hz, col in [("4bar", 4), ("EOD", 3)]:
            y = M[:, col]
            X = np.column_stack([np.ones_like(y), exc, prox])
            beta, se_cls, se_hac, r2, lag, n = ols_hac(y, X)
            t2 = beta[2] / se_hac[2]
            p2 = _one_sided_p(t2, "greater")
            flag = "■" if (beta[2] > 0 and p2 < 0.05) else "□"
            print(f"  {s:>5d} {hz:>7s} {beta[1]:>12.4f} {beta[2]:>13.4f} "
                  f"{t2:>7.2f} {p2:>11.3e}{flag} {cc:>15.3f} {n:>4d}")
            if s == 13 and hz == "EOD":
                ha_primary = (beta, se_hac, p2)

    bA, seA, pA = ha_primary
    print(f"\n  [primary spec: split=13(正午), horizon=EOD]")
    print(f"    γ1(exc_am, 汎用MR)  = {bA[1]:+.4f}  (t={bA[1]/seA[1]:.2f})   ← <0 期待")
    print(f"    γ2(prox_am, 壁固有) = {bA[2]:+.4f}  (t={bA[2]/seA[2]:.2f})   片側p(γ2>0)={pA:.3e}")
    ha_pass = (bA[2] > 0 and pA < 0.05)
    print(f"    → H-A {'支持: 壁近接で汎用MR超の追加反落あり' if ha_pass else '不支持: 汎用MR以上の壁効果なし'}")

    # ════════════════ H-B secondary ════════════════
    print("\n" + "─" * 84)
    print("H-B (secondary): 突破幅 非対称 (ボラ正規化 extent/√rv, Put>Call?)")
    print("─" * 84)

    def extent_call(r):
        cu = r.call * r.ratio
        mx = r.closes.max()
        return (mx - cu) / cu if mx > cu else np.nan

    def extent_put(r):
        pu = r.put * r.ratio
        mn = r.closes.min()
        return (pu - mn) / pu if mn < pu else np.nan

    ec = np.array([(extent_call(r), np.sqrt(r.rv)) for _, r in narrow.iterrows()])
    ep = np.array([(extent_put(r), np.sqrt(r.rv)) for _, r in violent.iterrows()])
    ec = ec[~np.isnan(ec[:, 0])]
    ep = ep[~np.isnan(ep[:, 0])]
    call_norm = ec[:, 0] / ec[:, 1]
    put_norm = ep[:, 0] / ep[:, 1]
    print(f"  Call break {len(ec)}日:  raw extent median {np.median(ec[:,0])*100:.3f}%  "
          f"norm median {np.median(call_norm):.3f}")
    print(f"  Put  break {len(ep)}日:  raw extent median {np.median(ep[:,0])*100:.3f}%  "
          f"norm median {np.median(put_norm):.3f}")
    u, pmw = stats.mannwhitneyu(put_norm, call_norm, alternative="greater")
    rng = np.random.default_rng(BOOT_SEED)
    diffs = [np.median(rng.choice(put_norm, put_norm.size)) - np.median(rng.choice(call_norm, call_norm.size))
             for _ in range(N_BOOT)]
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    hb_pass = pmw < 0.05 and ci[0] > 0
    print(f"  MW 片側(Put>Call) p={pmw:.3e}   median差(norm) {np.median(put_norm)-np.median(call_norm):+.3f} "
          f"[boot95% {ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  → H-B {'支持: Put 突破幅 > Call (ボラ調整後)' if hb_pass else '不支持: 非対称はボラ差で説明可/有意でない'}")

    # ════════════════ H-C secondary (exploratory) ════════════════
    print("\n" + "─" * 84)
    print("H-C (secondary, EXPLORATORY, n小): Put reclaim 後 前方リターン>0?")
    print("─" * 84)

    def reclaim_fwd(r, h):
        pu = r.put * r.ratio
        cl = r.closes
        below = cl < pu
        if not below.any():
            return None
        first = int(np.argmax(below))
        after = np.where(cl[first + 1:] > pu)[0]
        if after.size == 0:
            return None
        rb = first + 1 + int(after[0])           # reclaim bar index
        if rb >= 25:
            return None
        end = 25 if h == "EOD" else min(rb + 4, 25)
        return (cl[end] - cl[rb]) / cl[rb]

    def violent_baseline(h):
        vals = []
        for _, r in violent.iterrows():
            cl = r.closes
            step = 13 if h == "EOD" else 4
            for t in range(0, 26 - step):
                vals.append((cl[t + step] - cl[t]) / cl[t])
        return float(np.mean(vals))

    for h in ["4bar", "EOD"]:
        fwd = np.array([x for _, r in violent.iterrows() if (x := reclaim_fwd(r, h)) is not None])
        base = violent_baseline(h)
        try:
            w, pw = stats.wilcoxon(fwd, alternative="greater")
            pw = float(pw)
        except Exception:
            pw = np.nan
        print(f"  [{h:>4s}] reclaim n={len(fwd)}  median {np.median(fwd)*100:+.3f}%  mean {fwd.mean()*100:+.3f}%  "
              f"Wilcoxon片側p(>0)={pw:.3e}")
        print(f"         violent baseline 同地平 mean {base*100:+.3f}%   差 {(fwd.mean()-base)*100:+.3f}%")
    print("  ※ n=40級・交絡(violent は元々MR)のため exploratory。確証でなく方向の示唆に留める。")

    # ════════════════ ゲート → Step1B 写像 ════════════════
    print("\n" + "=" * 84)
    print("ゲート → EA Step 1B 写像")
    print("=" * 84)
    print(f"  H-A(primary) {'PASS' if ha_pass else 'FAIL'}: "
          f"{'narrow=Call Wall フェード短' if ha_pass else '壁は汎用MR以上でない→①の方向エッジ棄却'}")
    print(f"  H-B(secondary) {'支持' if hb_pass else '不支持'}: "
          f"{'Put で素直な逆張り買い禁止(真空)' if hb_pass else '突破幅の非対称は確認されず'}")
    print(f"  H-C(secondary/exploratory): reclaim 買いは方向示唆のみ(上記)。本採用は要追検証。")
    print("\n  留保: コスト無し。intraday US500・ex-div・obs.C。H-A の妥当性は exc_am 統制に依存。")

    # ── per-event CSV ──
    recs = []
    for _, r in narrow.iterrows():
        op = r.opens[0]; call_us = r.call * r.ratio
        D = (call_us - op) / op; exc = (r.highs[:13].max() - op) / op
        recs.append(dict(date=r.date.date(), sub="narrow", D=D, exc_am=exc, prox_am=D - exc,
                         fwd_eod=(r.closes[25] - r.closes[12]) / r.closes[12], rv=r.rv))
    pd.DataFrame(recs).to_csv(a.out, index=False)
    print(f"\n[written] {a.out}  （US500由来 → .gitignore 推奨）")


if __name__ == "__main__":
    main()
