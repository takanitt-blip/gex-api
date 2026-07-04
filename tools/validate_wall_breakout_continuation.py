#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_wall_breakout_continuation.py ─ 壁ブレイク後の継続性（pre-registered, B = 場中イベント）

================================================================================
PRE-REGISTRATION（freeze before running, PC_VALIDATION §1.1）
================================================================================

■ 目的 / 位置づけ
  四状態モデルの「未検証セル」を埋める。④-bis（§1.9）は壁の【内側】で方向の持続/反転を
  検定し VR<1（平均回帰, 荒いレンジも戻る）= null を得た。本検定は対称な問い ──
  壁の【外側】（Pr>Call / Pr<Put）でトレンド（方向の持続）が生じるか ── を直接叩く。

                    │ 壁の内側 (range)      │ 壁の外側 (squeeze/break)
  ─────────────────┼───────────────────────┼──────────────────────────
  方向の持続(VR)    │ ④-bis: VR<1 = 戻る     │  ← ★本検定（B）
                                                
  決定価値: 確定方針の戦術ゲート（PC_MT5 §1.3 役割a「ブレイク中はレンジ系逆張りを止める」）は
  「外側=順張り（継続）だから逆張りは死ぬ」を前提に乗る。本検定はその荷重前提を検証する。
  外側がトレンド → ゲートに根拠。外側も戻る → ゲートにエッジ無し（重大な発見）。

■ 仮説（結果を見る前に確定）
  H1: 壁ブレイク後の前向き15分リターンは「持続的」= VR(q) > 1、かつ内側 whole-session VR(<1)より上。
  H0: VR(q) = 1（ランダムウォーク, 継続も反転もしない）。
  ※ B は「ブレイクbar以降のみ」の純粋前向き窓。A（セッション全体VR）は採らない
    （前半の内側レンジ部分で希釈されるため。B が運用ゲートに忠実かつクリーン）。

■ なぜ「トレンド = 持続性(VR)」で測り、振幅では測らないか（誤判断34 の再発防止）
  壁の外側は④で確定済みの高ボラ regime。高ボラ＝上下どちらにも大きく動く。
  「ブレイク後こんなに動いた＝トレンド」は④（大きさ）を方向エッジと取り違える罠
  （①H-B が raw 2.4倍→√rv 正規化で消失したのと同型）。VR は比でボラ水準がキャンセル
  するため、振幅でなく【持続性】だけを測る。drift は伴走（符号付き follow-through）で別掲。

■ ブレイクイベントの定義（条件づけ。終値ベース＝EA の iClose(M15,1) に忠実, ヒゲ排除）
  壁座標 = 同日アンカー ratio_K（wall_coords / §3.7 構造検定アンカー）。call_us = call×ratio_K。
  Call break: closes が call_us を上抜けた最初の足。close[0] が既に外 = gap-out → 条件づけbar=0。
  Put  break: closes が put_us を下抜けた最初の足。同様に gap-out 対応。
  ・1セッション×1サイドにつき最初のクロスのみ（再クロス無視＝自己相関回避）。
  ・閾値 Y=0（壁そのもの。EAゲートはゼロ貫通で発火）。Y>0 は将来 exploratory のみ。
  ・持続フィルタ無し（即 reclaim も除外しない＝結果での条件づけ＝カーブフィッティング回避）。
    即 reclaim は前向き窓で自然に VR<1/負 drift として出る。

■ 前向き測定窓（先読み封じ, 誤判断33）
  条件づけbar t の【close 以降】= closes[t:] の対数リターン（当該セッションK内のみ）。
  ブレイク足自身の move（= ブレイクそのもの）は継続でないので forward に含めない。
  最小前向き長 ≥4本（= q=4 を成立させる最小）。t>21（残り<4本）の晩いブレイクは drop（件数記録）。
  壁が翌日 chase する問題は「K内のみ」測定で回避。

■ 指標
  主    : ブレイク群ごとに pooled forward VR(q=4)（イベント内 overlapping q集計→群内プール）。
          event-block bootstrap（B=5000, seed=42）で 95%CI と片側 p(VR≤1)。q=4 ≈1時間=機関の時計。
  伴走i : VR(q=2)。
  伴走ii: √rv 正規化した符号付き continuation = sign×(close[25]−close[t])/close[t] / √rv。
          mean>0 か（ブレイク方向への follow-through, 振幅交絡を√rvで除去）。bootstrap で評価。
  対照  : 内側 primary の whole-session pooled VR（全25本リターン）= ④-bis と整合する参照線(<1 期待)。

■ 判定（事前固定, 多重性）
  検定順: ① Call群 VR(q=4) → ② Put群 VR(q=4) → 伴走。
  「壁の外=トレンド」を支持と言うには【両群とも】VR(q=4)>1 が片側 bootstrap p<0.025
  （Bonferroni, 2群）。保守的に両群必須。
  検出力の正直さ: 実効 各~数十イベント。VR≳1.3 級の強い継続なら検出可。微弱(VR≈1.05)は
  検出不能 → null は「強い継続は無い」であって「継続ゼロの証明」ではない（Type II 留保）。
  ゲートは強い継続がある時だけ価値がある → この検出力プロファイルは意思決定に合う。

■ ②原設計からの amend（透明性, ①H-A と同じ作法）
  §1.4 ② は「ブレイク日の翌日リターン」だが EAゲートは場中(K内)発火 → 「K内のブレイクbar以降の
  前向き継続」へ amend。日跨ぎは V-2c 領域で別問題。

■ データ規律 / 実装規約
  標本 = ④/④-bis/① と同一 primary（build_primary を import, inside & rest_backfill_v2 & full-RTH,
  期待 488 = narrow223/violent265）。[GATE] で件数照合（誤判断35）。
  座標 look-ahead: 壁レベルは EOD(T) 由来（予測的）。ratio_K は構造検定アンカー（§3.7 容認）。
  ライブEA は ratio_T+ex-div（前段確定）。 留保: intraday US500・ex-div・obs.C。
  wall_coords + validate_wall_interaction(build_primary, load_intraday_ohlc) を import。CI 非実行。
  出力 per-event CSV は US500 由来 → .gitignore。
================================================================================
"""
from __future__ import annotations

import sys
# Windows cp932 で日本語/記号の stdout 不可を回避（誤判断35 の移植性ガード）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import numpy as np
import pandas as pd

from validate_wall_interaction import build_primary  # 同一標本（488 ゲート）を共有

# ── locked spec ───────────────────────────────────────────────────────────────
N_BOOT = 5000
BOOT_SEED = 42
MIN_FWD = 4                  # 最小前向き本数（= q=4 成立条件）
Q_PRIMARY = 4               # 主 VR
Q_SECOND = 2               # 伴走 VR
EXPECTED = dict(n=488, narrow=223, violent=265)


# ── ブレイク検出（終値ベース・gap-out 対応・最初のクロスのみ）────────────────────
def detect_break_bar(closes: np.ndarray, wall_us: float, side: str):
    """side='call': 上抜け / 'put': 下抜け。条件づけbar index を返す。無ければ None。"""
    if side == "call":
        outside = closes > wall_us
    else:
        outside = closes < wall_us
    if outside[0]:                       # close[0] が既に外 = gap-out（保持）
        return 0
    for t in range(1, len(closes)):
        if outside[t] and not outside[t - 1]:
            return t
    return None


# ── pooled VR（イベント内 overlapping q集計 → 群内プール, mean-adjusted）──────────
def _overlap_q(returns_1bar: np.ndarray, q: int) -> np.ndarray:
    """1本リターン列 → 重複あり q本和。長さ L-q+1（L<q なら空）。"""
    L = len(returns_1bar)
    if L < q:
        return np.empty(0)
    csum = np.cumsum(returns_1bar)
    head = np.concatenate([[0.0], csum[:-q]])
    return csum[q - 1:] - head           # 長さ L-q+1


def pooled_vr(events_1bar: list[np.ndarray], q: int) -> float:
    """events_1bar: 各イベントの前向き1本リターン列（raw, 符号整列しない＝VRは分散比で符号不問）。"""
    used = [e for e in events_1bar if len(e) >= q]   # 主標本は >=MIN_FWD>=q を満たす
    if not used:
        return np.nan
    all1 = np.concatenate(used)
    var1 = all1.var(ddof=1)
    allq = np.concatenate([_overlap_q(e, q) for e in used])
    varq = allq.var(ddof=1)
    if var1 <= 0:
        return np.nan
    return (varq / q) / var1


def boot_vr(events_1bar: list[np.ndarray], q: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(events_1bar)
    point = pooled_vr(events_1bar, q)
    dist = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        samp = [events_1bar[i] for i in rng.choice(idx, n, replace=True)]
        dist[b] = pooled_vr(samp, q)
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    p_le1 = float(np.nanmean(dist <= 1.0))          # 片側 bootstrap p(VR<=1)
    return point, float(lo), float(hi), p_le1


def boot_mean(vals: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(vals)
    point = float(np.mean(vals))
    dist = np.array([np.mean(rng.choice(vals, n, replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p_le0 = float(np.mean(dist <= 0.0))             # 片側 p(mean<=0)
    return point, float(lo), float(hi), p_le0


# ── イベント収集 ────────────────────────────────────────────────────────────────
def collect_events(prim: pd.DataFrame, side: str):
    """戻り: (events_1bar[list], drift_norm[np.ndarray], rec[list of dict])"""
    wallcol = "call" if side == "call" else "put"
    sign = +1.0 if side == "call" else -1.0
    ev_1bar, drift, rec = [], [], []
    n_gap = n_late = n_break = 0
    for _, r in prim.iterrows():
        cl = r.closes
        if len(cl) < 26:                 # full-RTH のみ（build_primary で担保済だが安全弁）
            continue
        wall_us = r[wallcol] * r.ratio   # 同日アンカー ratio_K
        t = detect_break_bar(cl, wall_us, side)
        if t is None:
            continue
        n_break += 1
        gap = (t == 0)
        n_gap += int(gap)
        L = 25 - t                       # 前向き本数
        if L < MIN_FWD:                  # 晩いブレイク → drop（事前登録）
            n_late += 1
            continue
        fwd = np.diff(np.log(cl[t:]))    # close[t] 以降の前向き1本リターン（長さ L）
        ev_1bar.append(fwd)
        dnorm = sign * (cl[25] - cl[t]) / cl[t] / np.sqrt(r.rv)   # √rv 正規化 符号付き continuation
        drift.append(dnorm)
        rec.append(dict(date=r.date.date(), side=side, subregime=r.subregime,
                        break_bar=t, gap_out=gap, fwd_bars=L,
                        wall_us=wall_us, ratio=r.ratio, rv=r.rv,
                        drift_norm=dnorm, cum_fwd_pct=(cl[25] - cl[t]) / cl[t] * 100.0))
    return ev_1bar, np.array(drift), rec, dict(n_break=n_break, n_gap=n_gap, n_late=n_late)


def whole_session_vr(prim: pd.DataFrame, q: int):
    """内側 primary 全日の whole-session pooled VR（参照線, ④-bis と整合 <1 期待）。"""
    evs = []
    for _, r in prim.iterrows():
        cl = r.closes
        if len(cl) >= 26:
            evs.append(np.diff(np.log(cl)))   # 25本
    return pooled_vr(evs, q), len(evs)


def report_group(name, ev, drift, meta, q1, q2):
    print("\n" + "─" * 84)
    print(f"{name}  ブレイク検出 {meta['n_break']}（gap-out {meta['n_gap']} / 晩late-drop {meta['n_late']}）"
          f" → 主標本 {len(ev)}")
    print("─" * 84)
    if len(ev) < 8:
        print("  [SKIP] 主標本 < 8 イベント。検出力不足のため検定保留。")
        return None
    p4, lo4, hi4, ple4 = boot_vr(ev, q1, N_BOOT, BOOT_SEED)
    p2, lo2, hi2, ple2 = boot_vr(ev, q2, N_BOOT, BOOT_SEED)
    dpt, dlo, dhi, dpl = boot_mean(drift, N_BOOT, BOOT_SEED)
    passq4 = (p4 > 1.0) and (ple4 < 0.025)
    print(f"  主 VR(q={q1}) = {p4:.3f}  [boot95% {lo4:.3f},{hi4:.3f}]  片側p(VR<=1)={ple4:.3e}  "
          f"{'■ >1 有意' if passq4 else '□ >1 とは言えない'}")
    print(f"  伴 VR(q={q2}) = {p2:.3f}  [boot95% {lo2:.3f},{hi2:.3f}]  片側p(VR<=1)={ple2:.3e}")
    print(f"  伴 drift(√rv正規化, 方向follow-through) mean = {dpt:+.4f}  "
          f"[boot95% {dlo:+.4f},{dhi:+.4f}]  片側p(mean<=0)={dpl:.3e}")
    return dict(vr4=p4, vr4_lo=lo4, vr4_p=ple4, passq4=passq4, vr2=p2, drift=dpt, drift_p=dpl, n=len(ev))


def main():
    ap = argparse.ArgumentParser(description="Wall breakout continuation test (pre-registered, B).")
    ap.add_argument("--map", default="/mnt/project/gex_history.json")
    ap.add_argument("--price", default="/mnt/project/US500_M15_202406030100_202606122100.csv")
    ap.add_argument("--start", default="2024-06-09")
    ap.add_argument("--end", default="2026-06-11")
    ap.add_argument("--out", default="wall_breakout_per_event.csv")
    ap.add_argument("--no-gate-assert", action="store_true")
    a = ap.parse_args()

    prim = build_primary(a.map, a.price, a.start, a.end)
    narrow = (prim.subregime == "narrow").sum()
    violent = (prim.subregime == "violent").sum()

    print("=" * 84)
    print("壁ブレイク後の継続性 — pre-registered test (B: 場中イベント)")
    print("=" * 84)
    print(f"window {a.start} .. {a.end}")
    print(f"primary {len(prim)} = narrow {narrow} / violent {violent}  (期待 488/223/265)")
    if not a.no_gate_assert:
        assert (len(prim), narrow, violent) == (EXPECTED["n"], EXPECTED["narrow"], EXPECTED["violent"]), \
            f"[GATE] 標本不一致 {len(prim)}/{narrow}/{violent} ≠ 488/223/265（入力 gex/価格を確認＝誤判断35）"
        print("  [GATE ok] ④/④-bis/① と同一標本（誤判断35 ガード）")

    # ── 対照: 内側 whole-session VR（④-bis 整合 <1 期待）──
    ws4, nws = whole_session_vr(prim, Q_PRIMARY)
    ws2, _ = whole_session_vr(prim, Q_SECOND)
    print(f"\n[対照] 内側 whole-session pooled VR  (n={nws}日, 全25本):  "
          f"VR(q={Q_PRIMARY})={ws4:.3f}   VR(q={Q_SECOND})={ws2:.3f}   ← ④-bis と同様 <1 を期待")

    # ── Call break / Put break ──
    ev_c, dr_c, rec_c, meta_c = collect_events(prim, "call")
    ev_p, dr_p, rec_p, meta_p = collect_events(prim, "put")
    res_c = report_group("[Call break] Pr>Call（上抜け→上に継続？）", ev_c, dr_c, meta_c, Q_PRIMARY, Q_SECOND)
    res_p = report_group("[Put  break] Pr<Put（下抜け→下に継続？）", ev_p, dr_p, meta_p, Q_PRIMARY, Q_SECOND)

    # ── 判定 ──
    print("\n" + "=" * 84)
    print("判定（事前固定: 両群とも VR(q=4)>1 片側 bootstrap p<0.025 = Bonferroni 2群）")
    print("=" * 84)
    cok = bool(res_c and res_c["passq4"])
    pok = bool(res_p and res_p["passq4"])
    both = cok and pok
    print(f"  Call break VR(q=4)>1 : {'PASS' if cok else 'FAIL'}")
    print(f"  Put  break VR(q=4)>1 : {'PASS' if pok else 'FAIL'}")
    print(f"  >>> 壁の外=トレンド（持続）: {'支持' if both else '不支持/不十分'}")
    if both:
        print("      → 確定方針の戦術ゲート（外側でレンジ系逆張りを止める）に実証的根拠。")
    else:
        print("      → 外側も内側同様に持続性を欠く可能性。ゲートの『外側=順張り』前提は")
        print("        本標本・本地平では支持されない（検出力 Type II 留保つき, 上記 n 参照）。")
        print("        内側 whole-session VR との大小も併読（外側>内側 なら微弱な持続の示唆）。")
    print("\n  留保: コスト無し（後段）。intraday US500・ex-div・obs.C。VR は振幅でなく持続性のみを測る。")

    # ── per-event CSV ──
    allrec = rec_c + rec_p
    if allrec:
        pd.DataFrame(allrec).sort_values(["side", "date"]).to_csv(a.out, index=False)
        print(f"\n[written] {a.out}  （US500由来 → .gitignore 推奨, n={len(allrec)}）")


if __name__ == "__main__":
    main()
