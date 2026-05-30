"""5/19〜5/22 実データを使った rest.py + schema 動作検証（誤判断25 検証）。

目的:
    γ-1/γ-2/γ-3 パッチ適用後のリポジトリで、本物の ThetaData history
    レスポンス CSV を使って:
      1. rest.py が trade_date 列を正しく出すか
      2. schema.validate() が REQUIRED 10 列を正しく検証するか
      3. 4 日分の trade_date が想定どおりの日付になるか
      4. 5/19〜5/22 の chain として妥当な OI 規模が出るか
    を end-to-end で確認する。

    HTTP 層は respx でモックして、rest.py の本物の経路（calendar 問い合わせ
    → greeks/eod → open_interest → merge → trade_date 列追加 → coerce）を
    通す。

実行:
    cd <repo_root> && python -m gex_engine.scripts.verify_obs_f_fix

合格基準:
    - 4 日分すべて完走
    - 各日 schema.validate(df).is_valid == True
    - 各日 trade_date 列が想定日付（5/19, 5/20, 5/21, 5/22）と一致
    - 各日 underlying_price が現実的（spot 700〜800 程度の範囲）
    - 行数が妥当（数千〜1万オーダー）

依存:
    本スクリプトは gex_engine リポジトリ直下から実行することを想定。
    spy_2026MMDD_greeks_eod.csv / spy_2026MMDD_open_interest.csv の
    4 セットが gex_engine/tests/fixtures/ または リポジトリ直下に
    存在すること。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import httpx
import respx

from gex_engine import schema
from gex_engine.adapters.rest import ThetaRestAdapter


# ──────────────────────────────────────────────────────────
# 検証設定
# ──────────────────────────────────────────────────────────

# 検証する 4 日分: (as_of, expected_trade_date)
# rest.py の _resolve_trade_date は as_of の直近過去営業日を返すので、
# 各 as_of に対する expected_trade_date は:
#   as_of=5/20(水) → T=5/19(火)
#   as_of=5/21(木) → T=5/20(水)
#   as_of=5/22(金) → T=5/21(木)
#   as_of=5/23(土) → T=5/22(金)（土日遡及）
#
# CSV ファイル名は「取引日 T」基準なので、各 as_of に対して
# spy_<T>_*.csv を読む。
TEST_DAYS = [
    (date(2026, 5, 20), date(2026, 5, 19), "20260519"),
    (date(2026, 5, 21), date(2026, 5, 20), "20260520"),
    (date(2026, 5, 22), date(2026, 5, 21), "20260521"),
    (date(2026, 5, 23), date(2026, 5, 22), "20260522"),  # 土曜の cron 想定
]

BASE_URL = "http://127.0.0.1:25503/v3"
ON_DATE_URL = f"{BASE_URL}/calendar/on_date"
GREEKS_EOD_URL = f"{BASE_URL}/option/history/greeks/eod"
OI_HISTORY_URL = f"{BASE_URL}/option/history/open_interest"

CSV_OPEN = 'type,open,close\n"open","09:30:00","16:00:00"\n\n'
CSV_WEEKEND = 'type,open,close\n"weekend",,\n\n'


# ──────────────────────────────────────────────────────────
# 出力
# ──────────────────────────────────────────────────────────

class C:
    G = "\033[92m"
    R = "\033[91m"
    Y = "\033[93m"
    B = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def section(t):
    print(f"\n{C.BOLD}{'═'*70}\n{t}\n{'═'*70}{C.END}")


def ok(m):
    print(f"  {C.G}✓{C.END} {m}")


def ng(m):
    print(f"  {C.R}✗{C.END} {m}")


def info(m):
    print(f"    {m}")


# ──────────────────────────────────────────────────────────
# フィクスチャ探索
# ──────────────────────────────────────────────────────────

def find_csv(filename: str) -> Path:
    """spy_*_greeks_eod.csv / spy_*_open_interest.csv を探す。

    探索順:
      1. gex_engine/tests/fixtures/  (テスト用配置)
      2. リポジトリ直下                (検証用に置いた場所)
      3. カレントディレクトリ
    """
    candidates = [
        Path("gex_engine/tests/fixtures") / filename,
        Path(filename),
        Path.cwd() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename!r}. Searched: {[str(c) for c in candidates]}"
    )


# ──────────────────────────────────────────────────────────
# calendar モックルーター
# ──────────────────────────────────────────────────────────

def calendar_router(request: httpx.Request) -> httpx.Response:
    """calendar/on_date モック。

    2026-05-19〜22 を平日扱い、土日 (5/23, 5/24) を weekend 扱い。
    rest.py の _resolve_trade_date は週末を遡及する。
    """
    d = request.url.params["date"]
    # 5/23, 5/24 は土日
    weekends = {"20260523", "20260524"}
    body = CSV_WEEKEND if d in weekends else CSV_OPEN
    return httpx.Response(200, text=body)


# ──────────────────────────────────────────────────────────
# 1 日分の検証
# ──────────────────────────────────────────────────────────

def verify_one_day(
    adapter: ThetaRestAdapter,
    as_of: date,
    expected_t: date,
    csv_date_str: str,
) -> bool:
    """1 日分の検証。"""
    section(f"検証日: as_of={as_of} (expected T={expected_t})")

    # フィクスチャ読み込み
    try:
        greeks_path = find_csv(f"spy_{csv_date_str}_greeks_eod.csv")
        oi_path = find_csv(f"spy_{csv_date_str}_open_interest.csv")
    except FileNotFoundError as e:
        ng(f"フィクスチャが見つからない: {e}")
        return False

    info(f"greeks: {greeks_path} ({greeks_path.stat().st_size:,} bytes)")
    info(f"oi:     {oi_path} ({oi_path.stat().st_size:,} bytes)")

    greeks_csv = greeks_path.read_text(encoding="utf-8")
    oi_csv = oi_path.read_text(encoding="utf-8")

    # respx でモック
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=greeks_csv)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=oi_csv)
    )

    # 取得
    try:
        df = adapter.get_option_chain("SPY", as_of)
    except Exception as e:
        ng(f"get_option_chain 例外: {type(e).__name__}: {e}")
        return False

    info(f"DataFrame shape: {df.shape}")
    info(f"columns ({len(df.columns)}): {list(df.columns)}")

    # 検証 1: 必須 10 列がそろっている
    expected_cols = set(schema.REQUIRED_DTYPES.keys())
    actual_cols = set(df.columns)
    if actual_cols == expected_cols:
        ok(f"columns == REQUIRED_DTYPES ({len(expected_cols)} 列)")
    else:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        ng(f"columns 不一致: missing={missing}, extra={extra}")
        return False

    # 検証 2: trade_date 列が一意かつ expected_t と一致
    import pandas as pd  # noqa
    if df["trade_date"].nunique() != 1:
        ng(f"trade_date が一意でない: nunique={df['trade_date'].nunique()}")
        return False
    actual_t = df["trade_date"].iloc[0]
    expected_ts = pd.Timestamp(expected_t)
    if actual_t == expected_ts:
        ok(f"trade_date == {expected_t} (Adapter の T 解決が正しい)")
    else:
        ng(f"trade_date 不一致: actual={actual_t}, expected={expected_ts}")
        return False

    # 検証 3: schema.validate() で is_valid
    result = schema.validate(df)
    if result.is_valid:
        ok(f"schema.validate(): is_valid (warnings={len(result.warnings)})")
        for w in result.warnings[:3]:
            info(f"  warning: {w}")
        if len(result.warnings) > 3:
            info(f"  ... and {len(result.warnings) - 3} more")
    else:
        ng(f"schema.validate(): {len(result.errors)} errors")
        for e in result.errors[:5]:
            info(f"  error: {e}")
        return False

    # 検証 4: underlying_price が一意かつ現実的
    if df["underlying_price"].nunique() != 1:
        info(
            f"underlying_price が一意でない: "
            f"nunique={df['underlying_price'].nunique()} "
            f"(同一 chain 内では timestamp 差で複数あり得る)"
        )
    spot = df["underlying_price"].iloc[0]
    if 600 <= spot <= 900:
        ok(f"underlying_price = {spot} (現実的な SPY 価格帯)")
    else:
        ng(f"underlying_price = {spot} (想定外: 600〜900 範囲外)")
        return False

    # 検証 5: 行数が妥当
    if 1000 <= len(df) <= 30000:
        ok(f"行数 = {len(df):,} (妥当な chain サイズ)")
    else:
        ng(f"行数 = {len(df):,} (想定外: 1000〜30000 範囲外)")
        return False

    # 補足情報
    info(f"strike 範囲: {df['strike'].min():.2f} 〜 {df['strike'].max():.2f}")
    info(f"expiration 数: {df['expiration'].nunique()}")
    info(f"OI 合計: {df['open_interest'].sum():,}")
    info(
        f"call/put: "
        f"{(df['right']=='call').sum()} / {(df['right']=='put').sum()}"
    )

    return True


# ──────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────

@respx.mock
def main() -> int:
    section("誤判断25 検証: 5/19〜5/22 実データ × rest.py + schema")
    info("γ-1/γ-2/γ-3 パッチ適用済みのリポジトリで実行してください")

    # calendar はすべてのリクエストで共通ルーター
    respx.get(ON_DATE_URL).mock(side_effect=calendar_router)

    adapter = ThetaRestAdapter(max_retries=0, retry_backoff_base=0.0)
    try:
        results = []
        for as_of, expected_t, csv_date in TEST_DAYS:
            ok_flag = verify_one_day(adapter, as_of, expected_t, csv_date)
            results.append((as_of, ok_flag))
    finally:
        adapter.close()

    section("結果サマリー")
    n_pass = sum(1 for _, ok_flag in results if ok_flag)
    n_total = len(results)
    for as_of, ok_flag in results:
        marker = f"{C.G}✓{C.END}" if ok_flag else f"{C.R}✗{C.END}"
        print(f"  {marker} as_of={as_of}")

    if n_pass == n_total:
        print(f"\n{C.G}{C.BOLD}全 {n_total} 日合格{C.END}")
        return 0
    else:
        print(
            f"\n{C.R}{C.BOLD}失敗 {n_total - n_pass} / 合計 {n_total}{C.END}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
