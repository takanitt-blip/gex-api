"""_fetch_open_interest（history 版）のテスト（Step 4）。

フィクスチャ oi_normal.csv は実 API ダンプ
（1_option_history_open_interest, 2026-05-13 SPY, expiration=2026-06-18）
から切り出した実データ。right は実 API 通り "CALL"/"PUT" 大文字。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from gex_engine.adapters.rest import ThetaFatalError, ThetaRestAdapter

BASE_URL = "http://127.0.0.1:25503/v3"
OI_HISTORY_URL = f"{BASE_URL}/option/history/open_interest"

FIXTURES = Path(__file__).parent / "fixtures"
OI_NORMAL = (FIXTURES / "oi_normal.csv").read_text()


@pytest.fixture
def adapter() -> ThetaRestAdapter:
    return ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)


# ── 正常系 ──

@respx.mock
def test_oi_returns_all_rows(adapter: ThetaRestAdapter) -> None:
    """フィクスチャ 12 行が全て返る（OI にフィルタは無い）。"""
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )
    df = adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    assert len(df) == 12


@respx.mock
def test_oi_output_columns(adapter: ThetaRestAdapter) -> None:
    """出力列が必須 5 列ちょうど（timestamp は落とす）。"""
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )
    df = adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    assert set(df.columns) == {
        "symbol", "expiration", "strike", "right", "open_interest",
    }


@respx.mock
def test_oi_right_normalized_lowercase(adapter: ThetaRestAdapter) -> None:
    """right が "CALL"/"PUT" → "call"/"put" に正規化される。

    フィクスチャ（実ダンプ）は大文字。正規化処理が実際に効いている
    ことの検証（旧フィクスチャは小文字で素通りしていた）。
    """
    # 前提確認: フィクスチャは大文字を含んでいる
    assert "CALL" in OI_NORMAL and "PUT" in OI_NORMAL

    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )
    df = adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    assert set(df["right"].unique()) <= {"call", "put"}
    assert "CALL" not in set(df["right"])
    assert "PUT" not in set(df["right"])


# ── パラメータ（日付の非対称性）──

@respx.mock
def test_oi_sends_date_param_as_given(adapter: ThetaRestAdapter) -> None:
    """渡された oi_date がそのまま date パラメータになる。

    日付の解釈（T → T翌営業日）はこの関数の責務ではない。
    呼び出し側が算出した日付をそのまま date に渡すだけ。
    """
    route = respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )
    adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    params = route.calls.last.request.url.params
    assert params["symbol"] == "SPY"
    assert params["expiration"] == "*"
    assert params["date"] == "20260514"
    assert params["format"] == "csv"


@respx.mock
def test_oi_uses_history_endpoint_not_snapshot(
    adapter: ThetaRestAdapter,
) -> None:
    """history エンドポイントを叩く（snapshot ではない）。"""
    route = respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=OI_NORMAL)
    )
    adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    assert route.called
    assert "/option/history/open_interest" in str(
        route.calls.last.request.url
    )


# ── 空レスポンス ──

@respx.mock
def test_oi_empty_returns_empty_df(adapter: ThetaRestAdapter) -> None:
    """空レスポンス（休場日等）は空 DataFrame を返す。"""
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text="")
    )
    df = adapter._fetch_open_interest("SPY", date(2026, 5, 14))
    assert df.empty


# ── 必須列欠落 ──

@respx.mock
def test_oi_missing_column_raises(adapter: ThetaRestAdapter) -> None:
    """open_interest 列が無いレスポンスは ThetaFatalError。"""
    bad_csv = (
        "symbol,expiration,strike,right,timestamp\n"
        "SPY,2026-06-18,740.0,CALL,2026-05-13T06:30:00.000\n"
    )
    respx.get(OI_HISTORY_URL).mock(
        return_value=httpx.Response(200, text=bad_csv)
    )
    with pytest.raises(ThetaFatalError, match="missing expected columns"):
        adapter._fetch_open_interest("SPY", date(2026, 5, 14))
