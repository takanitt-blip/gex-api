"""_request_csv のエラー処理機構・分類・Protocol 準拠のテスト。

DESIGN 5.4 の test_rest_adapter.py 全面改修に伴い、旧 snapshot 前提の
テストのうち「経路非依存で有効」な検証をここに集約した。

旧 test_rest_adapter.py からの移行:
    TestClassifyStatus      → そのまま移植（純粋関数、HTTP/経路非依存）
    TestProtocolCompliance  → そのまま移植（経路非依存）
    TestFatalErrors         → _request_csv 直接テストに書き直し
    TestRetryable           → _request_csv 直接テストに書き直し

書き直しの方針:
    旧テストは get_option_chain 経由で snapshot URL を叩いていた。
    history 化で get_option_chain は calendar + greeks/eod + OI の
    3 経路になったため、リトライ・FATAL 機構を get_option_chain 経由で
    試すと経路の都合に汚染される。リトライ・エラー分類は _request_csv
    1 関数の責務なので、_request_csv を直接呼んで純粋に検証する。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from gex_engine.adapters.base import DataFetcher
from gex_engine.adapters.rest import (
    ThetaErrorCategory,
    ThetaFatalError,
    ThetaPermissionError,
    ThetaRestAdapter,
    ThetaRetryExhaustedError,
    classify_status,
)

# _request_csv を直接叩くための任意のテスト用エンドポイント。
# calendar/fetch のいずれでもない中立な path を使い、
# エラー処理機構だけを純粋に検証する。
TEST_PATH = "/test/endpoint"
TEST_URL = f"http://127.0.0.1:25503/v3{TEST_PATH}"


@pytest.fixture
def adapter() -> ThetaRestAdapter:
    """max_retries=2, backoff=0 で高速化。"""
    a = ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)
    yield a
    a.close()


# ──────────────────────────────────────────────────────────
# エラー分類のロジック単体テスト（旧 TestClassifyStatus、無修正移植）
# ──────────────────────────────────────────────────────────

class TestClassifyStatus:
    """公式エラーコードを 4 区分に分類する純粋関数のテスト。"""

    def test_200_is_success(self) -> None:
        assert classify_status(200) == ThetaErrorCategory.SUCCESS

    def test_472_is_no_data(self) -> None:
        assert classify_status(472) == ThetaErrorCategory.NO_DATA

    @pytest.mark.parametrize("code", [429, 470, 474, 570, 571])
    def test_retryable_codes(self, code: int) -> None:
        assert classify_status(code) == ThetaErrorCategory.RETRYABLE

    @pytest.mark.parametrize(
        "code", [404, 471, 473, 475, 476, 477, 478, 572]
    )
    def test_fatal_codes(self, code: int) -> None:
        assert classify_status(code) == ThetaErrorCategory.FATAL

    def test_unknown_code_is_fatal(self) -> None:
        """未知のステータスは FATAL 扱い（保守的な選択）。"""
        assert classify_status(999) == ThetaErrorCategory.FATAL


# ──────────────────────────────────────────────────────────
# _request_csv: 成功と NO_DATA
# ──────────────────────────────────────────────────────────

class TestRequestCsvSuccess:
    """SUCCESS / NO_DATA の基本動作。"""

    @respx.mock
    def test_200_returns_body(self, adapter: ThetaRestAdapter) -> None:
        """200 はレスポンス本文をそのまま返す。"""
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(200, text="col1,col2\n1,2\n")
        )
        result = adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert result == "col1,col2\n1,2\n"

    @respx.mock
    def test_472_returns_empty_string(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """472 NO_DATA は空文字列を返す（例外は投げない）。"""
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(472, text="no data")
        )
        result = adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert result == ""

    @respx.mock
    def test_format_csv_is_auto_appended(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """format=csv は _request_csv が自動付与する。"""
        route = respx.get(TEST_URL).mock(
            return_value=httpx.Response(200, text="ok")
        )
        adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        params = route.calls.last.request.url.params
        assert params["format"] == "csv"
        assert params["symbol"] == "SPY"


# ──────────────────────────────────────────────────────────
# _request_csv: FATAL（旧 TestFatalErrors を書き直し）
# ──────────────────────────────────────────────────────────

class TestRequestCsvFatal:
    """FATAL は即 raise、リトライしない。"""

    @respx.mock
    def test_471_raises_permission_error(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """471 は ThetaPermissionError（Free プランで Option 叩いた等）。"""
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(471, text="permission denied")
        )
        with pytest.raises(ThetaPermissionError) as exc_info:
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert exc_info.value.status_code == 471

    @pytest.mark.parametrize(
        "code", [404, 473, 475, 476, 477, 478, 572]
    )
    @respx.mock
    def test_other_fatal_codes_raise_fatal_error(
        self, adapter: ThetaRestAdapter, code: int
    ) -> None:
        """471 以外の FATAL は ThetaFatalError。"""
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(code, text=f"error {code}")
        )
        with pytest.raises(ThetaFatalError) as exc_info:
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert exc_info.value.status_code == code

    @respx.mock
    def test_fatal_does_not_retry(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """FATAL は即 raise、リトライ呼び出しは発生しない（1 回のみ）。"""
        route = respx.get(TEST_URL).mock(
            return_value=httpx.Response(473, text="invalid params")
        )
        with pytest.raises(ThetaFatalError):
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert route.call_count == 1


# ──────────────────────────────────────────────────────────
# _request_csv: RETRYABLE（旧 TestRetryable を書き直し）
# ──────────────────────────────────────────────────────────

class TestRequestCsvRetryable:
    """RETRYABLE は指数バックオフで最大 max_retries 回リトライ。"""

    @respx.mock
    def test_retryable_recovers_on_second_attempt(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """1 回目 429 → 2 回目 200 → 成功。"""
        respx.get(TEST_URL).mock(side_effect=[
            httpx.Response(429, text="os limit"),
            httpx.Response(200, text="recovered"),
        ])
        result = adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert result == "recovered"

    @respx.mock
    def test_retryable_exhausts(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """max_retries=2 → 計 3 回（初回 + リトライ 2 回）全て 429
        → ThetaRetryExhaustedError。"""
        route = respx.get(TEST_URL).mock(
            return_value=httpx.Response(429, text="os limit")
        )
        with pytest.raises(ThetaRetryExhaustedError) as exc_info:
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        # max_retries=2 → 試行回数は 3（初回 + 2 リトライ）
        assert route.call_count == 3
        assert exc_info.value.status_code == 429

    @pytest.mark.parametrize("code", [429, 470, 474, 570, 571])
    @respx.mock
    def test_all_retryable_codes_retry(
        self, adapter: ThetaRestAdapter, code: int
    ) -> None:
        """RETRYABLE の全コードがリトライ対象（初回 + 2 リトライ）。"""
        route = respx.get(TEST_URL).mock(
            return_value=httpx.Response(code, text=f"retryable {code}")
        )
        with pytest.raises(ThetaRetryExhaustedError):
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert route.call_count == 3

    @respx.mock
    def test_network_error_is_retryable(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """httpx.ConnectError 等の通信エラーもリトライ扱い。"""
        respx.get(TEST_URL).mock(side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, text="recovered"),
        ])
        result = adapter._request_csv(TEST_PATH, {"symbol": "SPY"})
        assert result == "recovered"

    @respx.mock
    def test_network_error_exhausts(
        self, adapter: ThetaRestAdapter
    ) -> None:
        """通信エラーが回復しないとリトライ枯渇で ThetaRetryExhaustedError。"""
        respx.get(TEST_URL).mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(ThetaRetryExhaustedError):
            adapter._request_csv(TEST_PATH, {"symbol": "SPY"})


# ──────────────────────────────────────────────────────────
# Protocol 準拠（旧 TestProtocolCompliance、無修正移植）
# ──────────────────────────────────────────────────────────

class TestProtocolCompliance:
    """ThetaRestAdapter が DataFetcher Protocol を満たすこと。"""

    def test_has_source_name(self) -> None:
        adapter = ThetaRestAdapter()
        try:
            assert adapter.source_name == "rest"
        finally:
            adapter.close()

    def test_satisfies_datafetcher_protocol(self) -> None:
        adapter = ThetaRestAdapter()
        try:
            assert isinstance(adapter, DataFetcher)
        finally:
            adapter.close()
