"""_fetch_greeks_eod と IV 健全性フィルタのテスト（Step 3a）。

フィクスチャ greeks_eod_normal.csv は実 API ダンプ
（4_option_history_greeks_eod, 2026-05-12 SPY）から切り出した実データ。
normal 8 行 + iv_zero 2 行 + iv_err100 2 行を含む。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from gex_engine.adapters.rest import ThetaFatalError, ThetaRestAdapter

BASE_URL = "http://127.0.0.1:25503/v3"
GREEKS_EOD_URL = f"{BASE_URL}/option/history/greeks/eod"

FIXTURES = Path(__file__).parent / "fixtures"
GREEKS_EOD_NORMAL = (FIXTURES / "greeks_eod_normal.csv").read_text()


@pytest.fixture
def adapter() -> ThetaRestAdapter:
    return ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)


# ── 正常系: フィルタ後の行構成 ──

@respx.mock
def test_greeks_eod_returns_only_healthy_rows(
    adapter: ThetaRestAdapter,
) -> None:
    """12 行のうち健全な 8 行（normal）のみ返る。

    iv_zero 2 行（implied_vol==0）と iv_err100 2 行（iv_error==100）は
    IV 健全性フィルタで除外される。
    """
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    assert len(df) == 8


@respx.mock
def test_greeks_eod_output_columns(adapter: ThetaRestAdapter) -> None:
    """出力列が _merge_oi_iv の契約 8 列ちょうど（iv_error は落とす）。"""
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    expected = {
        "symbol", "expiration", "strike", "right",
        "bid", "ask", "implied_volatility", "underlying_price",
    }
    assert set(df.columns) == expected
    assert "iv_error" not in df.columns


@respx.mock
def test_greeks_eod_right_normalized_lowercase(
    adapter: ThetaRestAdapter,
) -> None:
    """right が "CALL"/"PUT" → "call"/"put" に正規化される。

    フィクスチャ（実ダンプ）は大文字。出力は小文字でなければならない。
    """
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    assert set(df["right"].unique()) <= {"call", "put"}


@respx.mock
def test_greeks_eod_iv_zero_excluded(adapter: ThetaRestAdapter) -> None:
    """implied_vol==0 の行（K=335, K=550 CALL）が結果に残らない。"""
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    strikes = set(df["strike"].astype(float))
    assert 335.0 not in strikes
    assert 550.0 not in strikes


@respx.mock
def test_greeks_eod_iv_error_sentinel_excluded(
    adapter: ThetaRestAdapter,
) -> None:
    """iv_error==100.0 の行（K=985, K=990 CALL）が結果に残らない。

    これらは implied_vol==0.1250 でシード値が残るため、
    implied_vol<=0 だけでは捕捉できない。番兵値判定が効くことの検証。
    """
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    strikes = set(df["strike"].astype(float))
    assert 985.0 not in strikes
    assert 990.0 not in strikes


@respx.mock
def test_greeks_eod_all_kept_rows_have_positive_iv(
    adapter: ThetaRestAdapter,
) -> None:
    """残った全行の implied_volatility が正（gamma 計算可能）。"""
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    iv = df["implied_volatility"].astype(float)
    assert (iv > 0).all()


# ── パラメータ ──

@respx.mock
def test_greeks_eod_sends_correct_params(
    adapter: ThetaRestAdapter,
) -> None:
    """start_date == end_date == T、expiration=*、format=csv。"""
    route = respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    params = route.calls.last.request.url.params
    assert params["symbol"] == "SPY"
    assert params["expiration"] == "*"
    assert params["start_date"] == "20260512"
    assert params["end_date"] == "20260512"
    assert params["format"] == "csv"


# ── 空レスポンス ──

@respx.mock
def test_greeks_eod_empty_returns_empty_df(
    adapter: ThetaRestAdapter,
) -> None:
    """空レスポンス（休場日等）は空 DataFrame を返す。"""
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text="")
    )
    df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))
    assert df.empty


# ── 必須列欠落 ──

@respx.mock
def test_greeks_eod_missing_column_raises(
    adapter: ThetaRestAdapter,
) -> None:
    """iv_error 列が無いレスポンスは ThetaFatalError。

    iv_error は健全性フィルタに必須なので、欠落は致命的。
    （他の必須列は揃え、iv_error だけを欠落させて意図を明確化）
    """
    bad_csv = (
        "symbol,expiration,strike,right,bid,ask,implied_vol,"
        "underlying_price\n"
        "SPY,2026-06-18,740.0,CALL,14.15,14.21,0.1466,738.18\n"
    )
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=bad_csv)
    )
    with pytest.raises(ThetaFatalError, match="missing expected columns"):
        adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))


# ── ログ ──

@respx.mock
def test_greeks_eod_logs_exclusion_summary(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """除外がサマリー INFO ログ 1 行で出る（1 行ごと WARNING ではない）。

    内訳: implied_vol<=0 が 2、iv_error==100 が 2。
    """
    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=GREEKS_EOD_NORMAL)
    )
    with caplog.at_level(logging.INFO):
        adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))

    filter_logs = [
        r for r in caplog.records if "IV health filter" in r.message
    ]
    # サマリーは 1 行のみ（1 行ごとログではない）
    assert len(filter_logs) == 1
    rec = filter_logs[0]
    assert rec.levelno == logging.INFO
    assert "implied_vol<=0: 2" in rec.message
    assert "iv_error==100.0: 2" in rec.message
    assert "kept 8 / 12" in rec.message


@respx.mock
def test_greeks_eod_no_log_when_all_healthy(
    adapter: ThetaRestAdapter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """除外ゼロのときは IV health filter ログを出さない。

    greeks_eod_partial.csv の K=718 のみ抜いた版ではなく、
    ここでは正常 8 行だけのレスポンスを使う。
    """
    healthy_only = (FIXTURES / "greeks_eod_normal.csv").read_text()
    lines = healthy_only.splitlines()
    # ヘッダ + normal 8 行（先頭 9 行）のみ
    healthy_csv = "\n".join(lines[:9]) + "\n"

    respx.get(GREEKS_EOD_URL).mock(
        return_value=httpx.Response(200, text=healthy_csv)
    )
    with caplog.at_level(logging.INFO):
        df = adapter._fetch_greeks_eod("SPY", date(2026, 5, 12))

    assert len(df) == 8
    filter_logs = [
        r for r in caplog.records if "IV health filter" in r.message
    ]
    assert len(filter_logs) == 0
