"""get_option_chain のエンドツーエンドテスト（Step 5）。

calendar → greeks/eod → open_interest → merge → coerce の全経路を
respx でモックして検証する。フィクスチャは実 API ダンプ由来。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from gex_engine import schema
from gex_engine.adapters.rest import ThetaRestAdapter

BASE_URL = "http://127.0.0.1:25503/v3"
ON_DATE_URL = f"{BASE_URL}/calendar/on_date"
GREEKS_EOD_URL = f"{BASE_URL}/option/history/greeks/eod"
OI_HISTORY_URL = f"{BASE_URL}/option/history/open_interest"

FIXTURES = Path(__file__).parent / "fixtures"
GREEKS_EOD_NORMAL = (FIXTURES / "greeks_eod_normal.csv").read_text()
GREEKS_EOD_PARTIAL = (FIXTURES / "greeks_eod_partial.csv").read_text()
OI_NORMAL = (FIXTURES / "oi_normal.csv").read_text()

CSV_OPEN = 'type,open,close\n"open","09:30:00","16:00:00"\n\n'


@pytest.fixture
def adapter() -> ThetaRestAdapter:
    return ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)


def _mock_all_open() -> None:
    """全 calendar/on_date 問い合わせを open で返す（平日のみのシナリオ）。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_OPEN)
    )


# ── 正常系：エンドツーエンド ──

@respx.mock
def test_get_option_chain_happy_path(adapter: ThetaRestAdapter) -> None:
    """全経路が通り、統一スキーマの DataFrame が返る。

    greeks_eod_normal(12行→IVフィルタで8行) と oi_normal(12行) を
    4列キーで join。両側揃うのは normal 8 ストライク。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    # 両側揃う 8 行（normal）。iv_zero/iv_err100 は IV 側に無く left_only。
    assert len(df) == 8
    # 統一スキーマの必須9列
    assert set(df.columns) == set(schema.REQUIRED_DTYPES.keys())


@respx.mock
def test_get_option_chain_passes_schema_validation(
    adapter: ThetaRestAdapter,
) -> None:
    """出力が schema.validate() を通る（dtype・値レベル）。"""
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    df = adapter.get_option_chain("SPY", date(2026, 5, 14))
    result = schema.validate(df)
    assert result.is_valid, f"errors: {result.errors}"


@respx.mock
def test_get_option_chain_dtype_coercion(
    adapter: ThetaRestAdapter,
) -> None:
    """coerce_to_schema により必須列の dtype が変換される。"""
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    df = adapter.get_option_chain("SPY", date(2026, 5, 14))
    for col, dtype in schema.REQUIRED_DTYPES.items():
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype}"


# ── 日付の解決（両エンドポイントとも T を使う）──

@respx.mock
def test_get_option_chain_uses_T_for_both_endpoints(
    adapter: ThetaRestAdapter,
) -> None:
    """greeks/eod と open_interest の両方に同じ取引日 T を渡す。

    as_of=5/14(木) → T=5/13(水)。両エンドポイントとも date=20260513。

    2026-05-19 訂正履歴:
        旧実装は open_interest の date に「T の翌営業日」を渡しており、
        旧テスト test_get_option_chain_date_asymmetry は oi date を
        "20260514"(=T+1) でアサートしていた。これは DESIGN セクション
        2.1 / 3.4 が公式ドキュメントの一文 "The reported open interest
        represents the open interest at the end of the previous trading
        day." を OPRA 報告タイミングの説明として読まず、API パラメータ
        仕様と取り違えた誤読に基づくものだった。
        正しい挙動は「date に渡した日 = 欲しい取引日そのもの」
        （greeks/eod と同じ規約）。2026-05-19 実行で date=今日 を
        渡したところ 400 "Cannot fetch current-day data" を観測し、
        誤読が一次証拠で確定して是正。
    """
    _mock_all_open()
    iv_route = respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    oi_route = respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    adapter.get_option_chain("SPY", date(2026, 5, 14))

    iv_params = iv_route.calls.last.request.url.params
    oi_params = oi_route.calls.last.request.url.params
    # 両エンドポイントとも取引日 T = 5/13 を使う
    assert iv_params["start_date"] == "20260513"
    assert iv_params["end_date"] == "20260513"
    assert oi_params["date"] == "20260513"


# ── 空レスポンス ──

@respx.mock
def test_get_option_chain_empty_both_returns_empty(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """greeks/eod・open_interest が両方空（symmetric）なら空の統一スキーマ

    DataFrame を返し、ERROR で EMPTY_BOTH と明示する（監査20）。
    trade_date はカレンダー検証済みのため「休場日」という語は出ない。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text="")
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text="")
    )

    with caplog.at_level(logging.ERROR):
        df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    assert df.empty
    assert set(df.columns) == set(schema.REQUIRED_DTYPES.keys())
    error_logs = [r for r in caplog.records if "EMPTY_BOTH" in r.message]
    assert len(error_logs) == 1
    assert error_logs[0].levelno == logging.ERROR
    assert "likely market holiday" not in error_logs[0].message.lower()


@respx.mock
def test_get_option_chain_empty_iv_only_fetches_oi_anyway(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """greeks/eod だけが空（asymmetric, iv_only）でも open_interest は必ず

    取得され、ERROR で EMPTY_ASYMMETRIC (iv_only) と非空側の行数が
    明示される（監査20）。旧実装は iv_df.empty で即 return し、
    open_interest を一切叩かなかったため、このケースを構造的に
    検出できなかった。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text="")
    )
    oi_route = respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    with caplog.at_level(logging.ERROR):
        df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    # 監査20 の核心: OI は実際に叩かれている（旧実装は叩かなかった）
    assert oi_route.called
    assert df.empty
    error_logs = [
        r for r in caplog.records
        if "EMPTY_ASYMMETRIC (iv_only)" in r.message
    ]
    assert len(error_logs) == 1
    assert error_logs[0].levelno == logging.ERROR
    # OI 側の行数（12 行）が明示されている
    assert "12 row" in error_logs[0].message
    assert "likely market holiday" not in error_logs[0].message.lower()


@respx.mock
def test_get_option_chain_empty_oi_only_returns_empty(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """open_interest だけが空（asymmetric, oi_only）なら空の統一スキーマ

    DataFrame を返し、ERROR で EMPTY_ASYMMETRIC (oi_only) と IV 側の
    行数が明示される（監査20）。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text="")
    )

    with caplog.at_level(logging.ERROR):
        df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    assert df.empty
    error_logs = [
        r for r in caplog.records
        if "EMPTY_ASYMMETRIC (oi_only)" in r.message
    ]
    assert len(error_logs) == 1
    assert error_logs[0].levelno == logging.ERROR
    # IV 側の行数（greeks_eod_normal は 12 行だが、IV 健全性フィルタで
    # iv_zero 2 + iv_err100 2 = 4 行が落ちるため _fetch_greeks_eod の
    # 戻り値は 8 行。happy_path テストの前提と同じ）
    assert "8 row" in error_logs[0].message
    assert "likely market holiday" not in error_logs[0].message.lower()


# ── Step 3b: サイレント Wall 欠落の事実ログ ──

@respx.mock
def test_get_option_chain_logs_silent_wall_loss(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """left_only（OI あり・IV なし）の OI 規模が INFO ログに出る。

    iv_zero K=550(OI=12325) / K=335(OI=9) と iv_err100 K=985,990 は
    IV フィルタで落ち、OI 側には残るため left_only になる。
    特に K=550 は OI が大きく、Wall 欠落リスクの観察対象。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    with caplog.at_level(logging.INFO):
        adapter.get_option_chain("SPY", date(2026, 5, 14))

    wall_logs = [
        r for r in caplog.records
        if "OI but no usable IV" in r.message
    ]
    assert len(wall_logs) == 1
    rec = wall_logs[0]
    assert rec.levelno == logging.INFO
    # IV フィルタで落ちた 4 ストライク（335,550,985,990 CALL）
    assert "4 strike(s)" in rec.message
    # lost OI = 9 + 12325 + 3300 + 3453 = 19087
    assert "19087" in rec.message


@respx.mock
def test_get_option_chain_partial_merge(
    adapter: ThetaRestAdapter,
) -> None:
    """greeks_eod_partial（K=718 欠落）でも両側揃う行のみ返る。

    greeks_eod_partial は K=718 CALL/PUT を抜いた 10 行（→IVフィルタ後6行）。
    oi_normal は 12 行（K=718 含む）。両側揃うのは normal 6 ストライク。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_PARTIAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    df = adapter.get_option_chain("SPY", date(2026, 5, 14))
    # normal 8 - K=718 の 2 = 6 行
    assert len(df) == 6
    strikes = set(df["strike"].astype(float))
    assert 718.0 not in strikes


# ── 営業日の遡及が get_option_chain 経由でも効く ──

@respx.mock
def test_get_option_chain_resolves_through_weekend(
    adapter: ThetaRestAdapter,
) -> None:
    """as_of が月曜なら T は前金曜（土日を遡及）。

    as_of=2026-05-18(月): T 解決で 5/17(日)・5/16(土)を飛ばし 5/15(金)。
    両エンドポイントとも date=20260515 を使う（2026-05-19 訂正以降）。
    """
    def on_date_router(request: httpx.Request) -> httpx.Response:
        d = request.url.params["date"]
        weekend = {"20260516", "20260517"}
        t = "weekend" if d in weekend else "open"
        body = (
            f'type,open,close\n"{t}",,\n\n' if t == "weekend"
            else CSV_OPEN
        )
        return httpx.Response(200, text=body)

    respx.get(ON_DATE_URL).mock(side_effect=on_date_router)
    iv_route = respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    oi_route = respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    adapter.get_option_chain("SPY", date(2026, 5, 18))

    # 両エンドポイントとも T = 5/15(金)
    assert iv_route.calls.last.request.url.params["start_date"] == "20260515"
    assert oi_route.calls.last.request.url.params["date"] == "20260515"


# ── 監査15: symbol 不一致の検出 ──

@respx.mock
def test_get_option_chain_symbol_mismatch_raises(
    adapter: ThetaRestAdapter,
) -> None:
    """引数 symbol と取得データの symbol が食い違うと ThetaFatalError。

    "QQQ" を要求したが、上流（greeks/eod・open_interest）が
    両方 SPY フィクスチャを返すシナリオ。両側 SPY で join 成立し、
    merged の symbol は SPY。引数 QQQ と不一致 → FATAL。
    監査15 が想定した「銘柄取り違え」の検出。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    from gex_engine.adapters.rest import ThetaFatalError
    with pytest.raises(ThetaFatalError, match="symbol mismatch"):
        adapter.get_option_chain("QQQ", date(2026, 5, 14))


# ── 監査14: マッチゼロの検出 ──

@respx.mock
def test_get_option_chain_zero_match_logs_and_returns_empty(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """IV/OI 両方非空だがキーが 1 つも一致しないと ERROR + 空 df。

    OI 側の expiration を別の満期に改変し、greeks/eod 側
    （2026-06-18）と一致しないようにする。both 行ゼロ。
    監査20 に合わせてログレベルを WARNING → ERROR に格上げ（trade_date は
    カレンダー検証済みで空が休場日ではあり得ない以上、この異常も同じ
    強度で顕在化させる）。
    """
    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    # OI の expiration を 2026-06-18 → 2026-07-17 に置換（満期不一致）
    oi_mismatched = OI_NORMAL.replace("2026-06-18", "2026-07-17")
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=oi_mismatched)
    )

    with caplog.at_level(logging.ERROR):
        df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    assert df.empty
    assert set(df.columns) == set(schema.REQUIRED_DTYPES.keys())
    zero_match_logs = [
        r for r in caplog.records
        if "MERGE_MISMATCH" in r.message
    ]
    assert len(zero_match_logs) == 1
    assert zero_match_logs[0].levelno == logging.ERROR


# ── 誤判断25: trade_date 列の契約 ──

@respx.mock
def test_get_option_chain_emits_trade_date_column(
    adapter: ThetaRestAdapter,
) -> None:
    """rest が出す trade_date 列は _resolve_trade_date の結果と一致する。

    obs.F (run_daily.py の as_of=today バグ) の再発を構造的に防ぐ
    レグレッションテスト。Adapter が「自分が解釈した取引日 T」を
    必ず外部に公開する契約 (誤判断25, 2026-05-24)。

    as_of=2026-05-14(木) → T=2026-05-13(水)。
    全行で trade_date 列が pd.Timestamp("2026-05-13") を持つ。
    """
    # pandas は他のテストでは未使用のため関数ローカル import
    import pandas as pd

    _mock_all_open()
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )

    df = adapter.get_option_chain("SPY", date(2026, 5, 14))

    # trade_date 列が必須スキーマに含まれる
    assert "trade_date" in df.columns
    # 全行同じ値（1 つの get_option_chain 呼び出し = 1 取引日）
    assert df["trade_date"].nunique() == 1
    # 値は _resolve_trade_date の結果 (T = 5/13)
    expected = pd.Timestamp("2026-05-13")
    assert df["trade_date"].iloc[0] == expected
