"""段階4 用 end-to-end スモークテスト（REST Adapter → Core → I/O）。

段階3.5 のスモークテストは Mock Adapter で行ったが、こちらは
REST Adapter（respx で httpx をモック）で同じ一気通貫を確認する。

実 ThetaData API は契約していないため、respx で公式仕様通りの
レスポンスを返すモックを立てて、Adapter 以降の挙動を検証する。

実行:
    cd /home/claude && python -m gex_engine.scripts.smoke_test_rest
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import respx

from gex_engine.adapters.rest import ThetaRestAdapter
from gex_engine.core.gex import calculate_all
from gex_engine.io_layer import save_gex_result

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


@respx.mock
def run_smoke_test() -> bool:
    """REST Adapter（モック化）→ Core → I/O の一気通貫を検証。"""
    section("end-to-end スモークテスト（段階4: REST Adapter）")

    OI_URL = "http://127.0.0.1:25503/v3/option/snapshot/open_interest"
    IV_URL = "http://127.0.0.1:25503/v3/option/snapshot/greeks/implied_volatility"

    oi_csv = (FIXTURES / "oi_normal.csv").read_text(encoding="utf-8")
    iv_csv = (FIXTURES / "iv_normal.csv").read_text(encoding="utf-8")

    respx.get(OI_URL).mock(return_value=httpx.Response(200, text=oi_csv))
    respx.get(IV_URL).mock(return_value=httpx.Response(200, text=iv_csv))

    # ── ステップ 1: REST Adapter からデータ取得 ──
    section("ステップ 1: REST Adapter → DataFrame")
    step("ThetaRestAdapter（respx でモック）を生成")

    with ThetaRestAdapter(max_retries=0, retry_backoff_base=0.0) as fetcher:
        step("get_option_chain('SPY', today)")
        df = fetcher.get_option_chain("SPY", date(2026, 5, 9))

    info(f"DataFrame shape: {df.shape}")
    info(f"列: {list(df.columns)}")
    info(f"underlying_price (unique): {df['underlying_price'].unique().tolist()}")
    info(f"strike 範囲: {df['strike'].min():.2f} 〜 {df['strike'].max():.2f}")

    # ── ステップ 2: Core Logic で計算 ──
    section("ステップ 2: calculate_all → GEXResult")
    result = calculate_all(df, as_of=date(2026, 5, 9), data_source=fetcher.source_name)

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
    fixed_utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
    save_gex_result(result, path=json_path, now_utc=fixed_utc)

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
        except Exception as e:
            ok = False
            detail = f"{detail} (例外: {type(e).__name__}: {e})"
        if ok:
            passed(f"{label} {detail}".rstrip())
            pass_count += 1
        else:
            failed(f"{label} {detail}".rstrip())
            fail_count += 1

    e = history[next(iter(history.keys()))]

    check(
        "data_source == 'rest'",
        lambda: e["data_source"] == "rest",
        f"(実測: {e['data_source']!r})",
    )
    check(
        "underlying_price がフィクスチャ値 450.25 と一致",
        lambda: e["underlying_price"] == 450.25,
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
        "max_pain がフィクスチャ範囲 [440, 460] 内",
        lambda: 440 <= e["max_pain"] <= 460,
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
