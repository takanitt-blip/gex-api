#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/probe_thetadata_history.py

Phase 2 の事実確認用「実 API ダンプ」スクリプト（使い捨てにせず tools/ に残す）。
PC_GOVERNANCE 6.7「方針転換時は実 API の生レスポンスをダンプ調査で確認」に従う。

このスクリプトは backfill 本体ではない。データ層の「可用性・形・時間」だけを
生レスポンスで観測し、報告書の未確定点（遡及境界 / req速度 / 所要時間）を実測値で
置き換えるための de-risk 用 probe。

閉じる監査ゴール:
  (1) 遡及境界 …… SPY で OI / greeks-eod が 200 を返す最古日を一次証拠で確定（§3.5 ①）。
                   ※ greeks/eod は原資産 SPY 終値に依存するため、OI 境界(2016説)と
                     greeks 境界(2017/2020説)が割れうる。真の床は「遅い方」で決まる。
  (2) 時間/サイズ … 1 リクエストの実 wall-clock・行数・バイト数を実測
  (3) スロットリング 連続 N 日を sleep なしで叩き 429/570 の有無を観測
  (4) date=T 規約 … 過去日で date=T の戻り timestamp がその日の EOD であることを目視（§3.5 ②/誤判断24）

設計上の約束:
  - standalone（rest.py を import しない）。rest.py は 472→空DF・バックオフ・エラー分類で
    生レスポンスを抽象化するため、probe では生の status / body を直接記録する。
  - probe 日は on_date で自己検証する（「取引日だと決めつけない」）。weekend/full_close なら
    最寄り開場日へ歩いて寄せる。
  - GEX 計算（calculate_all）は通さない。gex_history.json には一切書かない。

前提: Theta Terminal (v3) が起動していること。SPY のみ。Standard プラン。
実行: python tools/probe_thetadata_history.py [--throttle-days 5]
出力: 標準出力にサマリ + tools/probe_results_<UTCstamp>.json に生証拠を保存
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx が必要です: pip install httpx")

# ---- 設定（doc で裏取り済みの値） -------------------------------------------
BASE_URL = "http://127.0.0.1:25503/v3"   # 全エンドポイント共通の base（各 doc サンプルより）
SYMBOL = "SPY"
TIMEOUT = 120.0                          # 大チェーンの転送を考慮し doc サンプルの 60 より長め
TRADING_DAY_TYPES = {"open", "early_close"}   # on_date の type のうち取引日扱い
MAX_WALK_STEPS = 10                      # 開場日へ歩く際の上限（無限ループ防止）

# probe ごとに返すレコードのキー（生証拠）
#   status, theta_code, elapsed_s, bytes, rows, ts_min, ts_max, header, samples, error


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


# ---- 低レベル: 生 GET（status を潰さない） ----------------------------------
def raw_get(client: httpx.Client, path: str, params: dict) -> dict:
    """raise_for_status を呼ばず、生の status・経過時間・バイト数・本文先頭を記録する。"""
    url = BASE_URL + path
    t0 = time.perf_counter()
    try:
        resp = client.get(url, params=params)
    except httpx.ConnectError as e:
        # Terminal 未起動だと WinError 10061 等。probe 全体が無意味なので呼び出し側で止める。
        return {"status": None, "elapsed_s": round(time.perf_counter() - t0, 3),
                "error": f"connection refused (Theta Terminal 未起動?): {e}"}
    except httpx.HTTPError as e:
        return {"status": None, "elapsed_s": round(time.perf_counter() - t0, 3),
                "error": f"{type(e).__name__}: {e}"}
    elapsed = round(time.perf_counter() - t0, 3)
    body = resp.content or b""
    return {
        "status": resp.status_code,
        "elapsed_s": elapsed,
        "bytes": len(body),
        "text": resp.text,        # 呼び出し側で CSV パース or エラー本文抽出に使う
    }


def parse_csv_record(r: dict, ts_col: str | None) -> dict:
    """raw_get の戻りを CSV として要約。data 行数・ヘッダ・サンプル・timestamp の日付範囲。"""
    out = {"status": r.get("status"), "elapsed_s": r.get("elapsed_s"),
           "bytes": r.get("bytes")}
    if r.get("error"):
        out["error"] = r["error"]
        return out
    if r["status"] != 200:
        # 非 200 は Theta コード（HTTP コード自体）と本文先頭だけ残す（PC_PIPELINE §5.6 の分類）
        out["theta_code"] = r["status"]
        out["text_head"] = r["text"][:300]
        return out

    lines = [ln for ln in r["text"].splitlines() if ln != ""]
    if not lines:
        out["rows"] = 0
        out["note"] = "200 だが本文空"
        return out
    reader = list(csv.reader(io.StringIO(r["text"])))
    header = reader[0] if reader else []
    data = reader[1:]
    out["rows"] = len(data)
    out["header_cols"] = len(header)
    out["header"] = ",".join(header)
    out["samples"] = [",".join(row) for row in (data[:2] + data[-1:])] if data else []

    # date=T 規約の目視: timestamp 列の日付部分の min/max を出す（§3.5 ②/誤判断24）
    if ts_col and ts_col in header and data:
        idx = header.index(ts_col)
        dates = set()
        for row in data:
            if idx < len(row) and row[idx]:
                dates.add(row[idx].split("T")[0])
        if dates:
            out["ts_min"] = min(dates)
            out["ts_max"] = max(dates)
            out["ts_distinct"] = len(dates)
    return out


# ---- カレンダー: on_date で開場判定し、開場日へ寄せる ------------------------
def on_date_type(client: httpx.Client, d: date) -> dict:
    """on_date の type（open/early_close/full_close/weekend）を取得。"""
    r = raw_get(client, "/calendar/on_date", {"date": ymd(d)})
    rec = {"date": d.isoformat(), "status": r.get("status"),
           "elapsed_s": r.get("elapsed_s")}
    if r.get("error"):
        rec["error"] = r["error"]
        return rec
    if r["status"] != 200:
        rec["theta_code"] = r["status"]
        rec["text_head"] = r["text"][:200]
        return rec
    rows = list(csv.reader(io.StringIO(r["text"])))
    if len(rows) >= 2 and rows[0]:
        header = rows[0]
        first = rows[1]
        if "type" in header:
            rec["type"] = first[header.index("type")]
    return rec


def resolve_open_day(client: httpx.Client, seed: date, direction: int) -> dict:
    """seed から direction(+1/-1)方向へ歩き、最初に open/early_close になる日を返す。
    各ステップで on_date を叩くので「取引日だと決めつけない」自己検証になる。"""
    walk = []
    d = seed
    for _ in range(MAX_WALK_STEPS):
        t = on_date_type(client, d)
        walk.append(t)
        if t.get("error"):
            return {"resolved": None, "walk": walk, "error": t["error"]}
        if t.get("type") in TRADING_DAY_TYPES:
            return {"resolved": d.isoformat(), "type": t.get("type"), "walk": walk}
        d = d + timedelta(days=direction)
    return {"resolved": None, "walk": walk, "error": f"{MAX_WALK_STEPS} 歩で開場日に届かず"}


# ---- 各データエンドポイントの probe -----------------------------------------
def probe_oi(client: httpx.Client, d: date) -> dict:
    r = raw_get(client, "/option/history/open_interest",
                {"symbol": SYMBOL, "expiration": "*", "date": ymd(d)})
    return parse_csv_record(r, ts_col="timestamp")


def probe_greeks(client: httpx.Client, d: date) -> dict:
    r = raw_get(client, "/option/history/greeks/eod",
                {"symbol": SYMBOL, "expiration": "*",
                 "start_date": ymd(d), "end_date": ymd(d)})
    return parse_csv_record(r, ts_col="timestamp")


def probe_year_holidays(client: httpx.Client, year: int) -> dict:
    r = raw_get(client, "/calendar/year_holidays", {"year": str(year)})
    rec = {"year": year, "status": r.get("status"), "elapsed_s": r.get("elapsed_s"),
           "bytes": r.get("bytes")}
    if r.get("error"):
        rec["error"] = r["error"]
        return rec
    if r["status"] != 200:
        rec["theta_code"] = r["status"]
        rec["text_head"] = r["text"][:300]
        return rec
    rows = list(csv.reader(io.StringIO(r["text"])))
    rec["header"] = ",".join(rows[0]) if rows else ""
    rec["rows"] = max(0, len(rows) - 1)
    rec["samples"] = [",".join(x) for x in rows[1:4]]   # 先頭 3 件だけ（スキーマ目視用）
    return rec


# ---- セクション実行 ----------------------------------------------------------
def section_calendar_sanity(client: httpx.Client) -> dict:
    """year_holidays が動くこと + 戻りスキーマを確認（backfill の営業日リスト設計の布石）。"""
    this_year = date.today().year
    return {str(this_year): probe_year_holidays(client, this_year),
            str(this_year - 1): probe_year_holidays(client, this_year - 1)}


def section_retention(client: httpx.Client) -> list[dict]:
    """遡及境界探索。各 intent の seed を on_date で開場日に寄せてから OI/greeks を叩く。"""
    # (説明, seed 日, 歩く方向)  ─ seed は「取引日かどうか不問」。on_date が判定する。
    # ラダーは資料間で割れた 3 候補を挟む:
    #   2016-01-01 = Options 表 STANDARD First Access（オプションデータ下限）
    #   2017-01-01 = stocks ページ CTA データ開始（SPY 原資産の候補）
    #   2020-01-01 = Subscription 本文「CTA-only 銘柄(SPY含む)は 2020 まで」（原資産の悲観候補）
    # OI と greeks/eod を別々に叩くので、OI 境界と greeks 境界が割れるか（原資産依存）を観る。
    intents = [
        ("recent anchor (today-7d 付近, 既知良好の基準点)", date.today() - timedelta(days=7), -1),
        ("first trading day 2020 (CTA-only 株 制限説 2020-01-01)", date(2020, 1, 1), +1),
        ("first trading day 2019 (2020 説の1つ下)", date(2019, 1, 1), +1),
        ("first trading day 2017 (CTA pricing 説 2017-01-01)", date(2017, 1, 1), +1),
        ("first trading day 2016 (Options 表 STANDARD 説 2016-01-01)", date(2016, 1, 1), +1),
        ("last trading day 2015 (2016 説の1つ下)", date(2015, 12, 31), -1),
        ("first trading day 2013 (深い床, PRO=2012-06)", date(2013, 1, 1), +1),
    ]
    results = []
    for label, seed, direction in intents:
        resolved = resolve_open_day(client, seed, direction)
        rec = {"intent": label, "seed": seed.isoformat(), "resolve": resolved}
        if resolved.get("resolved"):
            d = date.fromisoformat(resolved["resolved"])
            rec["on_date_type"] = resolved.get("type")
            rec["oi"] = probe_oi(client, d)
            rec["greeks_eod"] = probe_greeks(client, d)
        results.append(rec)
    return results


def section_throttle(client: httpx.Client, n_days: int) -> dict:
    """連続 N 営業日を sleep なしで OI+greeks 連続取得し、429/570 と per-request 時間を観測。
    500 日は叩かない（それは backfill 本体）。小 N で詰まるかだけを正直に観る。"""
    # 直近の開場日から過去へ N 営業日を集める
    days: list[date] = []
    d = date.today() - timedelta(days=2)   # current-day 回避（PC_GOVERNANCE 6.9）
    steps = 0
    while len(days) < n_days and steps < n_days * 4:
        t = on_date_type(client, d)
        if t.get("type") in TRADING_DAY_TYPES:
            days.append(d)
        d -= timedelta(days=1)
        steps += 1

    calls = []
    flagged = False
    t_start = time.perf_counter()
    for dd in days:
        oi = probe_oi(client, dd)
        gr = probe_greeks(client, dd)
        for ep, rec in (("open_interest", oi), ("greeks_eod", gr)):
            st = rec.get("status")
            if st in (429, 570):
                flagged = True
            calls.append({"date": dd.isoformat(), "endpoint": ep,
                          "status": st, "elapsed_s": rec.get("elapsed_s"),
                          "rows": rec.get("rows"), "bytes": rec.get("bytes")})
    return {"requested_days": [x.isoformat() for x in days],
            "total_elapsed_s": round(time.perf_counter() - t_start, 3),
            "any_429_or_570": flagged,
            "calls": calls}


# ---- サマリ印字 --------------------------------------------------------------
def print_summary(report: dict) -> None:
    def fmt(rec: dict) -> str:
        if not rec:
            return "  (skip)"
        if rec.get("error"):
            return f"  ERROR: {rec['error']}"
        st = rec.get("status")
        if st != 200:
            return f"  status={st} theta={rec.get('theta_code')} head={rec.get('text_head','')[:80]!r}"
        ts = f" ts=[{rec.get('ts_min','?')}..{rec.get('ts_max','?')}]" if "ts_min" in rec else ""
        return (f"  status=200 rows={rec.get('rows')} cols={rec.get('header_cols','?')} "
                f"bytes={rec.get('bytes')} {rec.get('elapsed_s')}s{ts}")

    print("=" * 72)
    print(f"ThetaData history probe  symbol={SYMBOL}  run_at={report['meta']['run_at_utc']}")
    print("=" * 72)

    print("\n[A] calendar/year_holidays sanity")
    for yr, rec in report["calendar_sanity"].items():
        if rec.get("error"):
            print(f"  {yr}: ERROR {rec['error']}")
        else:
            print(f"  {yr}: status={rec.get('status')} rows={rec.get('rows')} "
                  f"header={rec.get('header')!r}")

    print("\n[B] retention boundary probe  (最古の status=200 が遡及境界)")
    for r in report["retention_probes"]:
        print(f"\n  - {r['intent']}")
        res = r.get("resolve", {})
        print(f"    resolved open day: {res.get('resolved')} (type={r.get('on_date_type')})")
        if res.get("error"):
            print(f"    resolve ERROR: {res['error']}")
            continue
        print(f"    OI    :{fmt(r.get('oi'))}")
        print(f"    greeks:{fmt(r.get('greeks_eod'))}")

    print("\n[C] throttle probe  (連続営業日 / sleep なし)")
    tp = report["throttle_probe"]
    print(f"  days={tp['requested_days']}")
    print(f"  total_elapsed={tp['total_elapsed_s']}s  any_429_or_570={tp['any_429_or_570']}")
    for c in tp["calls"]:
        print(f"    {c['date']} {c['endpoint']:<13} status={c['status']} "
              f"{c['elapsed_s']}s rows={c.get('rows')} bytes={c.get('bytes')}")
    print("\n" + "=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="ThetaData history 事実確認 probe（SPY/Standard）")
    ap.add_argument("--throttle-days", type=int, default=5,
                    help="スロットリング観測の連続営業日数（既定 5、500 は叩かない）")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent
    report: dict = {
        "meta": {
            "symbol": SYMBOL, "base_url": BASE_URL,
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "throttle_days": args.throttle_days,
            "note": "standalone probe; rest.py 非経由; gex 計算なし; gex_history.json 不書込",
        }
    }

    with httpx.Client(timeout=TIMEOUT) as client:
        # 接続疎通を最初に確認（Terminal 未起動なら即停止）
        ping = on_date_type(client, date.today() - timedelta(days=3))
        if ping.get("error"):
            print(f"[FATAL] Theta Terminal に接続できません: {ping['error']}", file=sys.stderr)
            print("        Theta Terminal (v3) を起動してから再実行してください。", file=sys.stderr)
            return 1

        print("[*] calendar sanity ...", file=sys.stderr)
        report["calendar_sanity"] = section_calendar_sanity(client)

        print("[*] retention boundary ...", file=sys.stderr)
        report["retention_probes"] = section_retention(client)

        print(f"[*] throttle probe ({args.throttle_days} days) ...", file=sys.stderr)
        report["throttle_probe"] = section_throttle(client, args.throttle_days)

    out_path = out_dir / f"probe_results_{now_stamp()}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(report)
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
