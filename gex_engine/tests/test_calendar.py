"""calendar 系メソッドのテスト（history 再設計フェーズ）。

Step 2-1: _fetch_calendar_on_date のみ。
respx で calendar/on_date をモックし、実 API ダンプ
（2026-05-16 取得）と同一のレスポンス形式で検証する。
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest
import respx

from gex_engine.adapters.rest import (
    ThetaFatalError,
    ThetaRestAdapter,
)

BASE_URL = "http://127.0.0.1:25503/v3"
ON_DATE_URL = f"{BASE_URL}/calendar/on_date"

# 実 API ダンプと同一形式（ヘッダ行・データ1行・末尾空行・クオート囲み）
CSV_OPEN = 'type,open,close\n"open","09:30:00","16:00:00"\n\n'
CSV_FULL_CLOSE = 'type,open,close\n"full_close",,\n\n'
CSV_EARLY_CLOSE = 'type,open,close\n"early_close","09:30:00","13:00:00"\n\n'
CSV_WEEKEND = 'type,open,close\n"weekend",,\n\n'


@pytest.fixture
def adapter() -> ThetaRestAdapter:
    """リトライ待ちゼロの Adapter（テスト高速化）。"""
    return ThetaRestAdapter(max_retries=2, retry_backoff_base=0.0)


# ── 正常系: 4 種類の type ──

@respx.mock
def test_on_date_open(adapter: ThetaRestAdapter) -> None:
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_OPEN)
    )
    assert adapter._fetch_calendar_on_date(date(2026, 5, 15)) == "open"


@respx.mock
def test_on_date_full_close(adapter: ThetaRestAdapter) -> None:
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_FULL_CLOSE)
    )
    assert adapter._fetch_calendar_on_date(date(2025, 12, 25)) == "full_close"


@respx.mock
def test_on_date_early_close(adapter: ThetaRestAdapter) -> None:
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_EARLY_CLOSE)
    )
    assert adapter._fetch_calendar_on_date(date(2025, 11, 28)) == "early_close"


@respx.mock
def test_on_date_weekend(adapter: ThetaRestAdapter) -> None:
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_WEEKEND)
    )
    assert adapter._fetch_calendar_on_date(date(2026, 5, 16)) == "weekend"


# ── 日付パラメータの形式 ──

@respx.mock
def test_on_date_sends_yyyymmdd(adapter: ThetaRestAdapter) -> None:
    """date パラメータが YYYYMMDD 形式で送られることを確認。"""
    route = respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=CSV_OPEN)
    )
    adapter._fetch_calendar_on_date(date(2026, 5, 15))
    request = route.calls.last.request
    assert request.url.params["date"] == "20260515"
    assert request.url.params["format"] == "csv"


# ── 異常系 ──

@respx.mock
def test_on_date_empty_response_raises(adapter: ThetaRestAdapter) -> None:
    """空レスポンス（calendar が引けない）は ThetaFatalError。

    OI の 472=休場日とは意味が異なる。遡及ループに weekend と
    誤認させてはいけないため、即停止が正しい。
    """
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text="")
    )
    with pytest.raises(ThetaFatalError, match="empty"):
        adapter._fetch_calendar_on_date(date(2026, 5, 15))


@respx.mock
def test_on_date_unknown_type_raises(adapter: ThetaRestAdapter) -> None:
    """未知の type は ThetaFatalError（黙って通すと遡及が誤動作）。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(
            200, text='type,open,close\n"holiday_xyz",,\n\n'
        )
    )
    with pytest.raises(ThetaFatalError, match="unknown schedule type"):
        adapter._fetch_calendar_on_date(date(2026, 5, 15))


@respx.mock
def test_on_date_header_only_raises(adapter: ThetaRestAdapter) -> None:
    """ヘッダのみでデータ行が無い場合は ThetaFatalError。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text="type,open,close\n\n")
    )
    with pytest.raises(ThetaFatalError, match="no data row"):
        adapter._fetch_calendar_on_date(date(2026, 5, 15))


@respx.mock
def test_on_date_missing_type_column_raises(adapter: ThetaRestAdapter) -> None:
    """type 列が無いレスポンスは ThetaFatalError。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(
            200, text='foo,bar\n"open","09:30:00"\n\n'
        )
    )
    with pytest.raises(ThetaFatalError, match="'type' column not found"):
        adapter._fetch_calendar_on_date(date(2026, 5, 15))


@respx.mock
def test_on_date_fatal_status_propagates(adapter: ThetaRestAdapter) -> None:
    """473 INVALID_PARAMS 等の FATAL は _request_csv から伝播。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(473, text="bad params")
    )
    with pytest.raises(ThetaFatalError):
        adapter._fetch_calendar_on_date(date(2026, 5, 15))


# ──────────────────────────────────────────────────────────
# Step 2-2: _resolve_trade_date
# ──────────────────────────────────────────────────────────

def _csv_for(type_value: str) -> str:
    """指定 type の on_date レスポンス CSV を生成（実ダンプと同一形式）。"""
    if type_value in ("open", "early_close"):
        close = "16:00:00" if type_value == "open" else "13:00:00"
        return f'type,open,close\n"{type_value}","09:30:00","{close}"\n\n'
    return f'type,open,close\n"{type_value}",,\n\n'


def _mock_calendar(schedule: dict[str, str]) -> None:
    """日付（YYYYMMDD 文字列）→ type の対応で on_date をモックする。

    respx は params マッチで日付ごとに別レスポンスを割り当てる。
    監査7 のとおり on_date レスポンスに date 列が無いため、
    呼び出し順依存ではなく date パラメータで対応づける。
    """
    for date_str, type_value in schedule.items():
        respx.get(ON_DATE_URL, params={"date": date_str}).mock(
            return_value=httpx.Response(200, text=_csv_for(type_value))
        )


@respx.mock
def test_resolve_trade_date_previous_weekday(
    adapter: ThetaRestAdapter,
) -> None:
    """as_of が平日 → 前営業日（前日）を返す。

    2026-05-15(金) を as_of とすると、前日 5/14(木) が取引日。
    """
    _mock_calendar({"20260514": "open"})
    result = adapter._resolve_trade_date(date(2026, 5, 15))
    assert result == date(2026, 5, 14)


@respx.mock
def test_resolve_trade_date_skips_weekend(
    adapter: ThetaRestAdapter,
) -> None:
    """as_of が月曜 → 土日を飛ばして前金曜を返す。

    2026-05-18(月) の前日は 5/17(日)→5/16(土)→5/15(金/open)。
    """
    _mock_calendar({
        "20260517": "weekend",
        "20260516": "weekend",
        "20260515": "open",
    })
    result = adapter._resolve_trade_date(date(2026, 5, 18))
    assert result == date(2026, 5, 15)


@respx.mock
def test_resolve_trade_date_skips_holiday_long_weekend(
    adapter: ThetaRestAdapter,
) -> None:
    """祝日 + 土日の連続休場を飛ばす。

    例: 火曜が as_of、前日 月曜が祝日(full_close)、その前が日・土、
    さらに前の金曜が取引日。計 4 日遡及。
    """
    _mock_calendar({
        "20260526": "full_close",  # 月（祝日想定）
        "20260525": "weekend",     # 日
        "20260524": "weekend",     # 土
        "20260523": "open",        # 金
    })
    result = adapter._resolve_trade_date(date(2026, 5, 27))
    assert result == date(2026, 5, 23)


@respx.mock
def test_resolve_trade_date_early_close_is_trading_day(
    adapter: ThetaRestAdapter,
) -> None:
    """early_close（短縮営業日）は取引日として返される。

    DESIGN 3.3 step4: early_close でも EOD レポートは生成される。
    """
    _mock_calendar({"20251127": "early_close"})
    result = adapter._resolve_trade_date(date(2025, 11, 28))
    assert result == date(2025, 11, 27)


@respx.mock
def test_resolve_trade_date_exhausts_lookback(
    adapter: ThetaRestAdapter,
) -> None:
    """上限日数を超えて取引日が見つからない場合は ThetaFatalError。

    全日 weekend を返すモックで上限到達を強制する。
    """
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=_csv_for("weekend"))
    )
    with pytest.raises(ThetaFatalError, match="no trading day found"):
        adapter._resolve_trade_date(date(2026, 5, 15))


@respx.mock
def test_resolve_trade_date_does_not_use_as_of_itself(
    adapter: ThetaRestAdapter,
) -> None:
    """as_of 当日は問い合わせず、必ず前日から遡る（DESIGN 3.2）。

    as_of 当日が open でも、それは返さない。前日を見る。
    """
    route_today = respx.get(
        ON_DATE_URL, params={"date": "20260515"}
    ).mock(return_value=httpx.Response(200, text=_csv_for("open")))
    _mock_calendar({"20260514": "open"})

    result = adapter._resolve_trade_date(date(2026, 5, 15))
    assert result == date(2026, 5, 14)
    # as_of 当日(5/15)は一度も問い合わせていないこと
    assert not route_today.called


@respx.mock
def test_resolve_trade_date_lookback_boundary(
    adapter: ThetaRestAdapter,
) -> None:
    """上限ぎりぎり（10 日目）で取引日が見つかれば成功する。

    9 日連続 weekend、10 日目に open。境界を踏む。
    """
    schedule = {}
    cursor = date(2026, 5, 14)  # as_of=5/15 の前日から
    for i in range(9):
        schedule[(cursor - timedelta(days=i)).strftime("%Y%m%d")] = "weekend"
    schedule[(cursor - timedelta(days=9)).strftime("%Y%m%d")] = "open"
    _mock_calendar(schedule)

    result = adapter._resolve_trade_date(date(2026, 5, 15))
    assert result == cursor - timedelta(days=9)


# ──────────────────────────────────────────────────────────
# Step 2-3: _next_business_day
# ──────────────────────────────────────────────────────────

@respx.mock
def test_next_business_day_simple(adapter: ThetaRestAdapter) -> None:
    """target が平日 → 翌日（平日）を返す。

    2026-05-14(木) の翌営業日は 5/15(金)。
    """
    _mock_calendar({"20260515": "open"})
    result = adapter._next_business_day(date(2026, 5, 14))
    assert result == date(2026, 5, 15)


@respx.mock
def test_next_business_day_skips_weekend(adapter: ThetaRestAdapter) -> None:
    """target が金曜 → 土日を飛ばして翌月曜を返す。

    2026-05-15(金) の翌は 5/16(土)→5/17(日)→5/18(月/open)。
    """
    _mock_calendar({
        "20260516": "weekend",
        "20260517": "weekend",
        "20260518": "open",
    })
    result = adapter._next_business_day(date(2026, 5, 15))
    assert result == date(2026, 5, 18)


@respx.mock
def test_next_business_day_skips_holiday_long_weekend(
    adapter: ThetaRestAdapter,
) -> None:
    """祝日 + 土日の連続休場を飛ばして翌取引日を返す。

    金曜が target、翌が土・日、月曜が祝日(full_close)、火曜が取引日。
    """
    _mock_calendar({
        "20260524": "weekend",     # 土
        "20260525": "weekend",     # 日
        "20260526": "full_close",  # 月（祝日想定）
        "20260527": "open",        # 火
    })
    result = adapter._next_business_day(date(2026, 5, 23))
    assert result == date(2026, 5, 27)


@respx.mock
def test_next_business_day_early_close_is_trading_day(
    adapter: ThetaRestAdapter,
) -> None:
    """early_close（短縮営業日）は翌営業日として返される。"""
    _mock_calendar({"20251127": "early_close"})
    result = adapter._next_business_day(date(2025, 11, 26))
    assert result == date(2025, 11, 27)


@respx.mock
def test_next_business_day_does_not_use_target_itself(
    adapter: ThetaRestAdapter,
) -> None:
    """target 当日は問い合わせず、必ず翌日から探索する。

    target 当日が open でも、それは返さない。翌日を見る。
    """
    route_target = respx.get(
        ON_DATE_URL, params={"date": "20260514"}
    ).mock(return_value=httpx.Response(200, text=_csv_for("open")))
    _mock_calendar({"20260515": "open"})

    result = adapter._next_business_day(date(2026, 5, 14))
    assert result == date(2026, 5, 15)
    assert not route_target.called


@respx.mock
def test_next_business_day_exhausts_scan(
    adapter: ThetaRestAdapter,
) -> None:
    """上限日数を超えて取引日が見つからない場合は ThetaFatalError。"""
    respx.get(ON_DATE_URL).mock(
        return_value=httpx.Response(200, text=_csv_for("weekend"))
    )
    with pytest.raises(ThetaFatalError, match="no trading day found"):
        adapter._next_business_day(date(2026, 5, 15))


@respx.mock
def test_resolve_and_next_are_asymmetric(
    adapter: ThetaRestAdapter,
) -> None:
    """DESIGN 3.5 の日付非対称性の確認。

    同じ取引日 T に対し、resolve は過去、next は未来へ進む。
    T=2026-05-15(金) を中心に:
      _resolve_trade_date(5/18(月)) → 5/15  (過去へ)
      _next_business_day(5/15)      → 5/18  (未来へ)
    両者は逆方向であり、結果が一致しないことを明示的に確認。
    """
    _mock_calendar({
        "20260517": "weekend",
        "20260516": "weekend",
        "20260515": "open",
        "20260518": "open",
    })
    t = adapter._resolve_trade_date(date(2026, 5, 18))
    nxt = adapter._next_business_day(t)
    assert t == date(2026, 5, 15)
    assert nxt == date(2026, 5, 18)
    assert t != nxt
