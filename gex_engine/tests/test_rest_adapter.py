"""REST Adapter のユニットテスト（respx で httpx をモック）。

カバー範囲:
    - 正常系: 2 エンドポイントから取得 → outer join → 統一スキーマ
    - エラー分類: 14 エラーコードを 4 区分にマップする分岐
    - リトライ: RETRYABLE は指数バックオフで最大 max_retries 回
    - 片側欠落: 警告ログ + 両側揃ったレコードのみ残す
    - 空レスポンス: NO_DATA / 空 CSV → 空 DataFrame
    - 列の不整合: 想定列が無い場合 ThetaFatalError
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from gex_engine.adapters.rest import (
    ThetaDataError,
    ThetaErrorCategory,
    ThetaFatalError,
    ThetaPermissionError,
    ThetaRestAdapter,
    ThetaRetryExhaustedError,
    classify_status,
)
from gex_engine.schema import REQUIRED_DTYPES, validate

FIXTURES = Path(__file__).parent / "fixtures"

OI_URL = "http://127.0.0.1:25503/v3/option/snapshot/open_interest"
IV_URL = "http://127.0.0.1:25503/v3/option/snapshot/greeks/implied_volatility"


# ──────────────────────────────────────────────────────────
# 共通ヘルパー
# ──────────────────────────────────────────────────────────

def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def adapter():
    """テスト用に max_retries=2, backoff=0 で高速化。"""
    a = ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)
    yield a
    a.close()


# ──────────────────────────────────────────────────────────
# エラー分類のロジック単体テスト
# ──────────────────────────────────────────────────────────

class TestClassifyStatus:
    """公式エラーコードを 4 区分に分類する純粋関数のテスト。"""

    def test_200_is_success(self):
        assert classify_status(200) == ThetaErrorCategory.SUCCESS

    def test_472_is_no_data(self):
        assert classify_status(472) == ThetaErrorCategory.NO_DATA

    @pytest.mark.parametrize("code", [429, 470, 474, 570, 571])
    def test_retryable_codes(self, code):
        assert classify_status(code) == ThetaErrorCategory.RETRYABLE

    @pytest.mark.parametrize(
        "code", [404, 471, 473, 475, 476, 477, 478, 572]
    )
    def test_fatal_codes(self, code):
        assert classify_status(code) == ThetaErrorCategory.FATAL

    def test_unknown_code_is_fatal(self):
        """未知のステータスは FATAL 扱い（保守的な選択）。"""
        assert classify_status(999) == ThetaErrorCategory.FATAL


# ──────────────────────────────────────────────────────────
# 正常系: 完全な OI + IV → 結合 → スキーマ準拠
# ──────────────────────────────────────────────────────────

class TestNormalFlow:
    """両エンドポイントが正常に応答する典型シナリオ。"""

    @respx.mock
    def test_returns_schema_compliant_dataframe(self, adapter):
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))

        # 必須列がすべて存在
        for col in REQUIRED_DTYPES.keys():
            assert col in df.columns, f"missing column: {col}"

        # validate() を通る
        result = validate(df)
        assert result.is_valid, f"errors: {result.errors}"

    @respx.mock
    def test_row_count_equals_intersection(self, adapter):
        """完全一致（10 行 vs 10 行）→ 結合後 10 行。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 10

    @respx.mock
    def test_underlying_price_propagated(self, adapter):
        """IV 側の underlying_price が DataFrame に乗ってくる。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        # フィクスチャ全行で underlying_price=450.25
        assert (df["underlying_price"] == 450.25).all()

    @respx.mock
    def test_implied_vol_renamed_to_implied_volatility(self, adapter):
        """ThetaData は 'implied_vol' を返す → 'implied_volatility' にリネームされる。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert "implied_volatility" in df.columns
        assert "implied_vol" not in df.columns

    @respx.mock
    def test_format_csv_param_is_sent(self, adapter):
        """format=csv が必ず送られる（json 等を勝手に使わない）。"""
        oi_route = respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        adapter.get_option_chain("SPY", date(2026, 5, 9))

        request = oi_route.calls[0].request
        assert request.url.params["format"] == "csv"
        assert request.url.params["expiration"] == "*"
        assert request.url.params["symbol"] == "SPY"


# ──────────────────────────────────────────────────────────
# 片側欠落（outer join + drop）
# ──────────────────────────────────────────────────────────

class TestPartialMerge:
    """OI / IV の片側欠落の振る舞い（論点3a/3b）。"""

    @respx.mock
    def test_partial_iv_drops_oi_only_records(self, adapter, caplog):
        """OI 10 行 vs IV 6 行 → 結合後 6 行（IV 欠落の 4 行は除外）。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_partial.csv"))
        )

        with caplog.at_level(logging.WARNING):
            df = adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert len(df) == 6

        # 警告ログに件数が出る
        merge_logs = [r for r in caplog.records if "OI/IV merge" in r.message]
        assert len(merge_logs) == 1
        assert "oi_only=4" in merge_logs[0].message

    @respx.mock
    def test_no_overlap_returns_empty(self, adapter):
        """両側にキー一致なし → 結合後 0 行。"""
        oi_text = (
            "timestamp,symbol,expiration,strike,right,open_interest\n"
            "2026-05-09T16:30:00.000,SPY,2026-05-16,500.00,call,100\n"
        )
        iv_text = _read_fixture("iv_normal.csv")  # strike 440-460 のみ

        respx.get(OI_URL).mock(return_value=httpx.Response(200, text=oi_text))
        respx.get(IV_URL).mock(return_value=httpx.Response(200, text=iv_text))

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 0


# ──────────────────────────────────────────────────────────
# NO_DATA / 空レスポンス
# ──────────────────────────────────────────────────────────

class TestNoData:
    """休場日や該当データなしのケース。"""

    @respx.mock
    def test_472_oi_returns_empty_dataframe(self, adapter):
        """OI が 472 NO_DATA → 空 DataFrame、IV は呼ばれない。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(472, text="no data for the request")
        )
        iv_route = respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert len(df) == 0
        # IV エンドポイントは OI が空ならスキップされる（早期リターン）
        assert iv_route.call_count == 0

    @respx.mock
    def test_472_iv_returns_empty_dataframe(self, adapter):
        """IV が 472 NO_DATA → 空 DataFrame。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        respx.get(IV_URL).mock(
            return_value=httpx.Response(472, text="no data")
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 0

    @respx.mock
    def test_empty_csv_body_returns_empty(self, adapter):
        """200 だが本文が空 → 空 DataFrame（実 API でヘッダのみのケース対応）。"""
        respx.get(OI_URL).mock(return_value=httpx.Response(200, text=""))

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 0


# ──────────────────────────────────────────────────────────
# FATAL カテゴリ（raise）
# ──────────────────────────────────────────────────────────

class TestFatalErrors:
    """FATAL は即 raise、リトライしない。"""

    @respx.mock
    def test_471_raises_permission_error(self, adapter):
        """Free プランで Option を叩いた時の代表的な事故。"""
        respx.get(OI_URL).mock(
            return_value=httpx.Response(471, text="permission denied")
        )

        with pytest.raises(ThetaPermissionError) as exc_info:
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert exc_info.value.status_code == 471

    @pytest.mark.parametrize(
        "code", [404, 473, 475, 476, 477, 478, 572]
    )
    @respx.mock
    def test_other_fatal_codes_raise_fatal_error(self, adapter, code):
        respx.get(OI_URL).mock(
            return_value=httpx.Response(code, text=f"error {code}")
        )

        with pytest.raises(ThetaFatalError) as exc_info:
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert exc_info.value.status_code == code

    @respx.mock
    def test_fatal_does_not_retry(self, adapter):
        """FATAL は即 raise、リトライ呼び出しは発生しない。"""
        route = respx.get(OI_URL).mock(
            return_value=httpx.Response(473, text="invalid params")
        )

        with pytest.raises(ThetaFatalError):
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert route.call_count == 1


# ──────────────────────────────────────────────────────────
# RETRYABLE カテゴリ（リトライ）
# ──────────────────────────────────────────────────────────

class TestRetryable:
    """RETRYABLE は最大 max_retries 回リトライ。"""

    @respx.mock
    def test_retryable_recovers_on_second_attempt(self, adapter):
        """1 回目 429 → 2 回目 200 → 成功。"""
        # respx の side_effect でレスポンスを順番に切り替える
        responses = [
            httpx.Response(429, text="os limit"),
            httpx.Response(200, text=_read_fixture("oi_normal.csv")),
        ]
        respx.get(OI_URL).mock(side_effect=responses)
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 10

    @respx.mock
    def test_retryable_exhausts(self, adapter):
        """max_retries=2 → 計 3 回（初回 + リトライ 2 回）試みて全て 429
        → ThetaRetryExhaustedError。"""
        route = respx.get(OI_URL).mock(
            return_value=httpx.Response(429, text="os limit")
        )

        with pytest.raises(ThetaRetryExhaustedError) as exc_info:
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        # max_retries=2 → 試行回数は 3（初回 + 2 リトライ）
        assert route.call_count == 3
        assert exc_info.value.status_code == 429

    @respx.mock
    def test_network_error_is_retryable(self, adapter):
        """httpx.ConnectError 等の通信エラーもリトライ扱い。"""
        responses = [
            httpx.ConnectError("connection refused"),
            httpx.Response(200, text=_read_fixture("oi_normal.csv")),
        ]
        respx.get(OI_URL).mock(side_effect=responses)
        respx.get(IV_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("iv_normal.csv"))
        )

        df = adapter.get_option_chain("SPY", date(2026, 5, 9))
        assert len(df) == 10


# ──────────────────────────────────────────────────────────
# レスポンス列の不整合（防御的プログラミング）
# ──────────────────────────────────────────────────────────

class TestSchemaMismatch:
    """API 仕様変更や不正レスポンスへの対応。"""

    @respx.mock
    def test_oi_missing_open_interest_column_raises_fatal(self, adapter):
        """open_interest 列が欠落していれば FATAL。"""
        bad_csv = (
            "timestamp,symbol,expiration,strike,right\n"  # open_interest 欠落
            "2026-05-09T16:30:00.000,SPY,2026-05-16,450.00,call\n"
        )
        respx.get(OI_URL).mock(return_value=httpx.Response(200, text=bad_csv))

        with pytest.raises(ThetaFatalError) as exc_info:
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert "open_interest" in str(exc_info.value)

    @respx.mock
    def test_iv_missing_underlying_price_raises_fatal(self, adapter):
        respx.get(OI_URL).mock(
            return_value=httpx.Response(200, text=_read_fixture("oi_normal.csv"))
        )
        bad_csv = (
            "symbol,expiration,strike,right,timestamp,bid,ask,implied_vol\n"
            "SPY,2026-05-16,450.00,call,2026-05-09T16:30:00.000,4.20,4.35,0.148\n"
        )
        respx.get(IV_URL).mock(return_value=httpx.Response(200, text=bad_csv))

        with pytest.raises(ThetaFatalError) as exc_info:
            adapter.get_option_chain("SPY", date(2026, 5, 9))

        assert "underlying_price" in str(exc_info.value)


# ──────────────────────────────────────────────────────────
# DataFetcher Protocol 適合
# ──────────────────────────────────────────────────────────

class TestProtocolCompliance:
    """Mock と同じ DataFetcher Protocol を満たしているか確認。"""

    def test_has_source_name(self):
        a = ThetaRestAdapter()
        assert a.source_name == "rest"
        a.close()

    def test_satisfies_datafetcher_protocol(self):
        from gex_engine.adapters.base import DataFetcher
        a = ThetaRestAdapter()
        assert isinstance(a, DataFetcher)
        a.close()
