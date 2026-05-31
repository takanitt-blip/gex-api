"""段階4 用 end-to-end スモークテスト（REST Adapter → Core → I/O）。

段階3.5 のスモークテストは Mock Adapter で行ったが、こちらは
REST Adapter（respx で httpx をモック）で同じ一気通貫を確認する。

実 ThetaData API は契約していないため、respx で公式仕様通りの
レスポンスを返すモックを立てて、Adapter 以降の挙動を検証する。

history 移行（DESIGN_history_rest_adapter.md）に伴い、モックすべき
エンドポイントは以下の 3 つ（snapshot 時代の 2 つから変わった、obs.H）:
    1. /v3/calendar/on_date            … _resolve_trade_date が取引日 T を解決
    2. /v3/option/history/open_interest … OI（date=T）
    3. /v3/option/history/greeks/eod    … IV/bid/ask/spot（start=end=T）
       ★ greeks/implied_volatility ではない（rest.py の警告参照）。

実行:
    cd /home/claude && python -m gex_engine.scripts.smoke_test_rest
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import respx

from gex_engine.adapters.rest import ThetaRestAdapter
from gex_engine.core.gex import calculate_all
from gex_engine.io_layer import save_gex_result
from gex_engine.market_calendar import next_business_day

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"


# 既存スモークテストの色付けユーティリティを使い回す
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def section(title: str) -> None:
    print(f"\n{Colors.BOLD}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.RESET}")


def step(msg: str) -> None:
    print(f"{Colors.BLUE}▶{Colors.RESET} {msg}")


def passed(msg: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def failed(msg: str) -> None:
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


# calendar/on_date が全営業日「open」を返すモック本文。
# rest.py の実機ダンプ仕様: ヘッダ + 1 データ行（値はダブルクオート囲み）。
#   type,open,close
#   "open","09:30:00","16:00:00"
# どの date パラメータでも常に open を返すので、_resolve_trade_date は
# as_of の前日（直近過去営業日）を即 T として確定する。
_CALENDAR_OPEN_CSV = 'type,open,close\n"open","09:30:00","16:00:00"\n'


@respx.mock
def run_smoke_test() -> bool:
    """REST Adapter（モック化）→ Core → I/O の一気通貫を検証。"""
    section("end-to-end スモークテスト（段階4: REST Adapter / history）")

    BASE = "http://127.0.0.1:25503/v3"
    CAL_URL = f"{BASE}/calendar/on_date"
    OI_URL = f"{BASE}/option/history/open_interest"
    GREEKS_URL = f"{BASE}/option/history/greeks/eod"

    oi_csv = (FIXTURES / "oi_normal.csv").read_text(encoding="utf-8")
    greeks_csv = (FIXTURES / "greeks_eod_normal.csv").read_text(encoding="utf-8")

    # calendar は date パラメータに依らず常に open（respx は query を無視して
    # path で一致させるため、ヘルパー1 つで全候補日をカバーできる）。
    respx.get(CAL_URL).mock(return_value=httpx.Response(200, text=_CALENDAR_OPEN_CSV))
    respx.get(OI_URL).mock(return_value=httpx.Response(200, text=oi_csv))
    respx.get(GREEKS_URL).mock(return_value=httpx.Response(200, text=greeks_csv))

    # ── ステップ 1: REST Adapter からデータ取得 ──
    section("ステップ 1: REST Adapter → DataFrame")
    step("ThetaRestAdapter（respx でモック）を生成")

    # as_of=5/13 → _resolve_trade_date が前日 5/12 を取引日 T として確定。
    # フィクスチャの underlying_timestamp が 2026-05-12 EOD なので、
    # T=5/12 で計算すれば DTE がフィクスチャの実態と整合する。
    as_of = date(2026, 5, 13)
    with ThetaRestAdapter(max_retries=0, retry_backoff_base=0.0) as fetcher:
        step(f"get_option_chain('SPY', {as_of})")
        df = fetcher.get_option_chain("SPY", as_of)
        # a7-A: 取引日 T は df 由来。session_date は fetcher が開いている間に算出。
        trade_date = df["trade_date"].iloc[0].date()
        session_date = next_business_day(trade_date, fetcher.schedule_type_on)

    info(f"DataFrame shape: {df.shape}")
    info(f"列: {list(df.columns)}")
    info(f"underlying_price (unique): {df['underlying_price'].unique().tolist()}")
    info(f"strike 範囲: {df['strike'].min():.2f} 〜 {df['strike'].max():.2f}")

    info(f"trade_date (df 由来, a7-A): {trade_date}")
    info(f"session_date (JSON key): {session_date}")

    # ── ステップ 2: Core Logic で計算 ──
    section("ステップ 2: calculate_all → GEXResult")
    result = calculate_all(df, as_of=trade_date, data_source=fetcher.source_name)

    info(f"symbol: {result.symbol}")
    info(f"underlying_price: {result.underlying_price}")
    info(f"call_wall: {result.call_wall}")
    info(f"put_wall: {result.put_wall}")
    info(f"zero_gamma: {result.zero_gamma}")
    info(f"max_pain: {result.max_pain}")
    info(f"total_gex (素): {result.total_gex:.2f}")
    info(f"data_source: {result.data_source}")

    # ── ステップ 3: I/O 層で JSON 書き出し ──
    section("ステップ 3: save_gex_result → JSON")
    tmpdir = tempfile.mkdtemp()
    json_path = os.path.join(tmpdir, "gex_history.json")
    fixed_utc = datetime(2026, 5, 13, 22, 30, 0, tzinfo=timezone.utc)
    save_gex_result(result, path=json_path, session_date=session_date, now_utc=fixed_utc)

    with open(json_path, encoding="utf-8") as f:
        history = json.load(f)
    print(json.dumps(history, indent=2, ensure_ascii=False))

    # ── ステップ 4: 合格基準検証 ──
    section("ステップ 4: 合格基準")

    fail_count = 0
    pass_count = 0

    def check(label: str, predicate, detail: str = "") -> None:
        nonlocal fail_count, pass_count
        try:
            ok = predicate()
        except Exception as ex:
            ok = False
            detail = f"{detail} (例外: {type(ex).__name__}: {ex})"
        if ok:
            passed(f"{label} {detail}".rstrip())
            pass_count += 1
        else:
            failed(f"{label} {detail}".rstrip())
            fail_count += 1

    e = history[next(iter(history.keys()))]

    # フィクスチャ greeks_eod_normal.csv の underlying_price は 738.18
    # （history 時代の SPY 現実データ。snapshot/Mock 時代の 450.25 ではない）。
    FIXTURE_SPOT = 738.18
    # フィクスチャに存在する strike の範囲（max_pain はこの中の 1 つ）。
    STRIKE_MIN, STRIKE_MAX = 335.0, 990.0

    check(
        "data_source == 'rest'",
        lambda: e["data_source"] == "rest",
        f"(実測: {e['data_source']!r})",
    )
    check(
        f"underlying_price がフィクスチャ値 {FIXTURE_SPOT} と一致",
        lambda: e["underlying_price"] == FIXTURE_SPOT,
        f"(実測: {e['underlying_price']})",
    )
    check(
        "symbol == 'SPY'",
        lambda: e["symbol"] == "SPY",
    )
    check(
        "call_wall >= underlying_price",
        lambda: e["call_wall"] >= e["underlying_price"],
        f"(spot={e['underlying_price']}, CW={e['call_wall']})",
    )
    check(
        "put_wall <= underlying_price",
        lambda: e["put_wall"] <= e["underlying_price"],
        f"(spot={e['underlying_price']}, PW={e['put_wall']})",
    )
    check(
        f"max_pain がフィクスチャ strike 範囲 [{STRIKE_MIN}, {STRIKE_MAX}] 内",
        lambda: STRIKE_MIN <= e["max_pain"] <= STRIKE_MAX,
        f"(MP={e['max_pain']})",
    )
    check(
        "total_gex が int 型",
        lambda: isinstance(e["total_gex"], int) and not isinstance(e["total_gex"], bool),
    )
    check(
        "regime ∈ {'range', 'trend'}",
        lambda: e["regime"] in {"range", "trend"},
        f"(実測: {e['regime']!r})",
    )

    # 後片付け
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    section("結果サマリー")
    total = pass_count + fail_count
    if fail_count == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}全 {total} 件合格{Colors.RESET}")
        return True
    print(f"{Colors.RED}{Colors.BOLD}失敗 {fail_count} / 合計 {total}{Colors.RESET}")
    return False


def main() -> int:
    try:
        return 0 if run_smoke_test() else 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
