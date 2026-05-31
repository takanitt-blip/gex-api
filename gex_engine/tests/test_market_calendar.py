"""market_calendar.next_business_day のユニットテスト（純粋・ネットワーク不要）。

schedule_lookup を fake で注入し、未来向き走査ロジックのみを検証する。
実カレンダー API（calendar/on_date）の forward 挙動はここでは検証しない
（実機で別途確認する。テストは論理のみ担保）。
"""

from __future__ import annotations

from datetime import date

import pytest

from gex_engine.market_calendar import TRADING_DAY_TYPES, next_business_day


def _make_lookup(schedule):
    """date -> type の辞書から schedule_lookup を作る。未登録日は KeyError。"""

    def lookup(d: date) -> str:
        return schedule[d]

    return lookup


class TestNextBusinessDay:
    def test_weekday_to_next_weekday(self):
        # 月(open) 起点 → 火(open)
        schedule = {date(2026, 5, 19): "open", date(2026, 5, 20): "open"}
        assert next_business_day(
            date(2026, 5, 19), _make_lookup(schedule)
        ) == date(2026, 5, 20)

    def test_friday_skips_weekend_to_monday(self):
        # 金(open) → 土(weekend) → 日(weekend) → 月(open)
        schedule = {
            date(2026, 5, 22): "open",     # 金（起点。lookup は start+1 から）
            date(2026, 5, 23): "weekend",  # 土
            date(2026, 5, 24): "weekend",  # 日
            date(2026, 5, 25): "open",     # 月
        }
        assert next_business_day(
            date(2026, 5, 22), _make_lookup(schedule)
        ) == date(2026, 5, 25)

    def test_skips_full_close_holiday(self):
        # 金 → 土/日 → 月(full_close=祝日) → 火(open)
        schedule = {
            date(2026, 5, 22): "open",
            date(2026, 5, 23): "weekend",
            date(2026, 5, 24): "weekend",
            date(2026, 5, 25): "full_close",  # 例: 祝日で終日休場
            date(2026, 5, 26): "open",
        }
        assert next_business_day(
            date(2026, 5, 22), _make_lookup(schedule)
        ) == date(2026, 5, 26)

    def test_early_close_is_a_trading_day(self):
        # early_close（短縮営業日）も取引日として返す
        schedule = {date(2026, 5, 22): "open", date(2026, 5, 23): "early_close"}
        assert next_business_day(
            date(2026, 5, 22), _make_lookup(schedule)
        ) == date(2026, 5, 23)

    def test_does_not_include_start_day(self):
        # 起点当日は走査対象外（start+1 から）。起点が open でも翌取引日を返す。
        schedule = {date(2026, 5, 19): "open", date(2026, 5, 20): "open"}
        result = next_business_day(date(2026, 5, 19), _make_lookup(schedule))
        assert result != date(2026, 5, 19)

    def test_raises_when_no_trading_day_within_scan(self):
        # 全日 full_close なら上限到達で ValueError
        def always_closed(d):
            return "full_close"

        with pytest.raises(ValueError, match="no trading day found"):
            next_business_day(date(2026, 5, 22), always_closed, max_scan_days=3)

    def test_lookup_call_count_bounded(self):
        # 走査回数が max_scan_days を超えない（サーキットブレーカーの実証）
        calls = []

        def counting(d):
            calls.append(d)
            return "full_close"

        with pytest.raises(ValueError):
            next_business_day(date(2026, 5, 22), counting, max_scan_days=3)
        assert len(calls) == 3

    def test_unknown_type_is_treated_as_non_trading(self):
        # 想定外 type は「取引日でない」として走査継続（type 検証は lookup の責務）。
        schedule = {
            date(2026, 5, 22): "open",
            date(2026, 5, 23): "garbage",  # 未知 type
            date(2026, 5, 24): "open",
        }
        assert next_business_day(
            date(2026, 5, 22), _make_lookup(schedule)
        ) == date(2026, 5, 24)

    def test_trading_day_types_matches_rest(self):
        # rest.py の _TRADING_DAY_TYPES と一致していること（共有定数の責務）
        assert TRADING_DAY_TYPES == frozenset({"open", "early_close"})
