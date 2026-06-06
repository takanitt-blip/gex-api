"""
end-to-end スモークテスト（段階3.5）

目的:
    Mock Adapter → calculate_all → save_gex_result の一気通貫で動作確認。
    ユニットテストでは捕まえられない、実コンポーネント同士を
    繋いだときの統合的な不具合を検出する。

ユニットテストとの違い:
    - ユニットテスト: 各層を独立してテスト（FakeGEXResult 等を使う）
    - スモークテスト: 実 GEXResult を使い、JSON 出力までの一貫性を検証

合格基準（先に決めた、後出しで増やさない）:
    1. 例外なく完走する
    2. 出力 JSON のキー数 = 12 個（v17: regime/regime_text 廃止、data_quality 追加。
       正常時は anomaly_detail を出さないので 12）
    3. call_wall >= spot
    4. put_wall <= spot
    5. put_wall <= zero_gamma <= call_wall
    6. max_pain がストライク範囲（360〜540）内
    7. total_gex が int 型、|total_gex| が 1e6 〜 1e8 のオーダー
    8. data_quality == "ok"（正常な Mock 地図。Z が見つかり C>=Z>=P が成立）
    9. symbol == "SPY"
    10. data_source == "mock"
    11. timestamp が "Z" で終わる ISO 8601
    12. 日付キーが "YYYY.MM.DD" 形式

実行:
    cd /home/claude && python -m gex_engine.scripts.smoke_test
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timezone
from typing import Any, Callable

from gex_engine.adapters.mock import MockDataFetcher
from gex_engine.core.gex import calculate_all
from gex_engine.io_layer import save_gex_result
from gex_engine.market_calendar import next_business_day


# ──────────────────────────────────────────────────────────
# 出力ヘルパー
# ──────────────────────────────────────────────────────────

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
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


# ──────────────────────────────────────────────────────────
# 検証ロジック
# ──────────────────────────────────────────────────────────

class CheckRunner:
    """合格基準を集めて run() で一斉検証。失敗しても全件回す。"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def check(self, label: str, predicate: Callable[[], bool], detail: str = "") -> None:
        try:
            ok = predicate()
        except Exception as e:
            ok = False
            detail = f"{detail} (例外: {type(e).__name__}: {e})"

        if ok:
            passed(f"{label} {detail}".rstrip())
            self.passed += 1
        else:
            failed(f"{label} {detail}".rstrip())
            self.failed += 1
            self.failures.append(label)

    def summary(self) -> bool:
        total = self.passed + self.failed
        print()
        if self.failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}全 {total} 件合格{Colors.RESET}")
            return True
        else:
            print(
                f"{Colors.RED}{Colors.BOLD}失敗 {self.failed} / 合計 {total}{Colors.RESET}"
            )
            for label in self.failures:
                print(f"  {Colors.RED}・{label}{Colors.RESET}")
            return False


# ──────────────────────────────────────────────────────────
# メインフロー
# ──────────────────────────────────────────────────────────

def run_smoke_test() -> bool:
    """Mock → Core → I/O の一気通貫を検証。

    Returns:
        全合格基準を満たしたら True、ひとつでも失敗したら False。
    """
    section("end-to-end スモークテスト（段階3.5）")

    # ── ステップ 1: Mock Adapter からデータ取得 ──
    section("ステップ 1: Mock Adapter → DataFrame")
    step("MockDataFetcher を生成（spot=450.0, seed=42）")
    fetcher = MockDataFetcher(spot_price=450.0, seed=42)

    step("get_option_chain('SPY', today) を呼ぶ")
    df = fetcher.get_option_chain("SPY", date.today())

    info(f"DataFrame shape: {df.shape}")
    info(f"列: {list(df.columns)}")
    info(f"行数: {len(df)}, ストライク数: {df['strike'].nunique()}")
    info(f"call/put 比: {(df['right']=='call').sum()} / {(df['right']=='put').sum()}")
    info(
        f"strike 範囲: {df['strike'].min():.0f} 〜 {df['strike'].max():.0f}"
    )

    # Adapter が解釈した取引日 T を df から抽出 (obs.F 修正、誤判断25)
    # 詳細は run_daily.py 同箇所のコメント参照。
    assert "trade_date" in df.columns, (
        "Adapter must emit trade_date column (誤判断25)"
    )
    assert df["trade_date"].nunique() == 1, (
        "trade_date must be unique per get_option_chain call (誤判断25)"
    )
    trade_date = df["trade_date"].iloc[0].date()
    info(f"Adapter resolved trade_date: {trade_date}")
    session_date = next_business_day(trade_date, fetcher.schedule_type_on)
    info(f"session_date (JSON key): {session_date}")

    # ── ステップ 2: Core Logic で計算 ──
    section("ステップ 2: calculate_all → GEXResult")
    step(f"calculate_all(df, as_of={trade_date}, data_source='mock') を呼ぶ")
    result = calculate_all(df, as_of=trade_date, data_source=fetcher.source_name)

    info(f"型: {type(result).__name__}")
    info(f"symbol: {result.symbol}")
    info(f"underlying_price: {result.underlying_price}")
    info(f"call_wall: {result.call_wall}")
    info(f"put_wall: {result.put_wall}")
    info(f"zero_gamma: {result.zero_gamma}")
    info(f"max_pain: {result.max_pain}")
    info(f"total_gex (素): {result.total_gex:.2f}")
    info(f"n_contracts_used: {result.n_contracts_used}")
    info(f"data_source: {result.data_source}")
    info(f"data_quality: {result.data_quality}")
    info(f"anomaly_detail: {result.anomaly_detail}")

    # ── ステップ 3: I/O 層で JSON 書き出し ──
    section("ステップ 3: save_gex_result → JSON ファイル")

    tmpdir = tempfile.mkdtemp()
    json_path = os.path.join(tmpdir, "gex_history.json")
    step(f"save_gex_result(result, path={json_path})")

    fixed_utc = datetime(2026, 5, 9, 22, 30, 0, tzinfo=timezone.utc)
    entry = save_gex_result(
        result,
        path=json_path,
        session_date=session_date,
        now_utc=fixed_utc,  # timestamp 用（再現性のため固定）
    )

    info(f"返り値の型: {type(entry).__name__}")
    info(f"ファイル存在: {os.path.exists(json_path)}")

    # ── ステップ 4: 出力 JSON を読み返して目視 ──
    section("ステップ 4: 出力 JSON（目視確認）")
    with open(json_path, encoding="utf-8") as f:
        history = json.load(f)

    print(json.dumps(history, indent=2, ensure_ascii=False))

    # ── ステップ 5: 合格基準検証 ──
    section("ステップ 5: 合格基準の自動検証")

    runner = CheckRunner()

    # 履歴は date_key → entry のフラット辞書なはず
    runner.check(
        "履歴のトップレベルが dict",
        lambda: isinstance(history, dict),
    )

    runner.check(
        "履歴のキーがちょうど 1 件",
        lambda: len(history) == 1,
        detail=f"(実測: {len(history)})",
    )

    date_key = next(iter(history.keys()))
    runner.check(
        "日付キーが 'YYYY.MM.DD' 形式",
        lambda: bool(re.match(r"^\d{4}\.\d{2}\.\d{2}$", date_key)),
        detail=f"(実測: {date_key!r})",
    )

    runner.check(
        "日付キーが session_date と一致",
        lambda: date_key == session_date.strftime("%Y.%m.%d"),
        detail=f"(実測: {date_key!r}, 期待: {session_date.strftime('%Y.%m.%d')!r})",
    )

    e = history[date_key]

    # 出力フィールド数（v17: regime/regime_text 廃止、data_quality 追加。
    # 正常時は anomaly_detail を出さないので 12 個）
    expected_keys = {
        "data_quality",
        "call_wall", "put_wall", "zero_gamma", "max_pain", "underlying_price",
        "total_gex", "timestamp", "data_source",
        "symbol", "as_of", "n_contracts_used",
    }
    runner.check(
        "出力フィールド数 = 12",
        lambda: len(e) == 12,
        detail=f"(実測: {len(e)})",
    )
    runner.check(
        "出力フィールド名が一致",
        lambda: set(e.keys()) == expected_keys,
        detail=f"(差分: 余分={set(e.keys())-expected_keys}, 不足={expected_keys-set(e.keys())})",
    )

    # 数値関係（v11 セクション8-4 の例: spot=450, CW=465, PW=435）
    spot = e["underlying_price"]
    cw = e["call_wall"]
    pw = e["put_wall"]
    zg = e["zero_gamma"]
    mp = e["max_pain"]

    runner.check(
        "call_wall >= spot",
        lambda: cw >= spot,
        detail=f"(spot={spot}, CW={cw})",
    )
    runner.check(
        "put_wall <= spot",
        lambda: pw <= spot,
        detail=f"(spot={spot}, PW={pw})",
    )
    runner.check(
        "put_wall <= zero_gamma <= call_wall",
        lambda: pw <= zg <= cw,
        detail=f"(PW={pw}, ZG={zg}, CW={cw})",
    )
    runner.check(
        "max_pain がストライク範囲 (360〜540) 内",
        lambda: 360.0 <= mp <= 540.0,
        detail=f"(MP={mp})",
    )

    # total_gex のスケールと型
    tg = e["total_gex"]
    runner.check(
        "total_gex が int 型",
        lambda: isinstance(tg, int) and not isinstance(tg, bool),
        detail=f"(型: {type(tg).__name__})",
    )
    runner.check(
        "|total_gex| が 1e6 〜 1e8 のオーダー",
        lambda: 1e6 <= abs(tg) <= 1e8,
        detail=f"(実測: {tg:,})",
    )

    # メタフィールド（v17: regime → data_quality）
    runner.check(
        "data_quality == 'ok' (正常な Mock 地図: Z 検出 & C>=Z>=P)",
        lambda: e["data_quality"] == "ok",
        detail=f"(実測: {e['data_quality']!r})",
    )
    runner.check(
        "正常時は anomaly_detail を出力しない",
        lambda: "anomaly_detail" not in e,
        detail=f"(キー有無: {'anomaly_detail' in e})",
    )
    runner.check(
        "symbol == 'SPY'",
        lambda: e["symbol"] == "SPY",
        detail=f"(実測: {e['symbol']!r})",
    )
    runner.check(
        "data_source == 'mock'",
        lambda: e["data_source"] == "mock",
        detail=f"(実測: {e['data_source']!r})",
    )
    runner.check(
        "timestamp が 'Z' で終わる ISO 8601",
        lambda: bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", e["timestamp"])),
        detail=f"(実測: {e['timestamp']!r})",
    )

    # GEXResult の伝搬確認（誤判断13 の再発防止）
    runner.check(
        "as_of が GEXResult から伝搬",
        lambda: e["as_of"] is not None and isinstance(e["as_of"], str),
        detail=f"(実測: {e['as_of']!r})",
    )
    runner.check(
        "n_contracts_used が GEXResult から伝搬",
        lambda: isinstance(e["n_contracts_used"], int) and e["n_contracts_used"] > 0,
        detail=f"(実測: {e['n_contracts_used']})",
    )

    # 既存 update_gex.py との桁整合（v12 セクション8-4 例: 13_002_525）
    runner.check(
        "既存 update_gex.py との桁整合（× S² × 0.01 スケール）",
        # raw=6421, S=450 → 6421 × 202500 × 0.01 = 13_002_525
        # Mock + Brent 法 + 暦日 T + r=0.04 等の差で完全一致はしないが、
        # 同一桁（数百万〜数千万 USD）に収まることだけ確認
        lambda: 5e6 <= abs(tg) <= 5e7,
        detail=f"(実測: {tg:,}, 既存例: 13,002,525)",
    )

    # ── 後片付け ──
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # ── サマリー ──
    section("結果サマリー")
    return runner.summary()


def main() -> int:
    """exit code を返す（CI から呼べるように）。"""
    try:
        ok = run_smoke_test()
        return 0 if ok else 1
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}スモークテストが例外で失敗:{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
