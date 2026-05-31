"""市場カレンダー: 取引日の前後解決（純粋・ネットワーク非依存）。

責務:
  - next_business_day(start, schedule_lookup): start より後の直近取引日を返す。
  - TRADING_DAY_TYPES: 「取引日とみなす calendar type」の共有定数。

設計方針:
  - httpx 等のネットワーク依存を一切持たない。1 日分の calendar type の取得は
    呼び出し側が schedule_lookup として注入する
    （rest は実 API、mock は素朴版、テストは fake）。
  - rest.py の _resolve_trade_date（過去向き走査）の完全な未来向き対称版。
    「open / early_close を取引日とする」分類を両者で一致させるため、
    TRADING_DAY_TYPES をここに集約する（rest.py _TRADING_DAY_TYPES と同値）。
  - obs.G（PC_GOVERNANCE）の根治用。gex_history.json の日付キーを
    「データが支配する取引セッション = next_business_day(trade_date)」に決定論的に
    決めるために使う。now() / today() への依存を鍵経路から排除する。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Callable

# 取引日とみなす calendar/on_date の type。
# early_close（短縮営業日）も EOD レポートは生成されるため取引日に含める。
# rest.py の ThetaRestAdapter._TRADING_DAY_TYPES と必ず同値に保つこと。
TRADING_DAY_TYPES: frozenset[str] = frozenset({"open", "early_close"})

# 未来向き走査の上限日数（無限ループ防止のサーキットブレーカー）。
# 米国市場の最長連続休場（年末年始の土日 + 祝日）でも数日。10 は十分な余裕。
# 上限到達は「カレンダー API の不整合」という真の異常を意味する。
DEFAULT_MAX_SCAN_DAYS: int = 10


def next_business_day(
    start: date,
    schedule_lookup: Callable[[date], str],
    *,
    max_scan_days: int = DEFAULT_MAX_SCAN_DAYS,
) -> date:
    """start より後の直近の取引日（open / early_close）を返す。

    rest.py の _resolve_trade_date（過去向き）の未来向き対称版。
    start 当日は対象に含めず、start + 1 日から未来へ走査する。

    Args:
        start: 起点となる取引日（通常は trade_date T）。
        schedule_lookup: 1 日の calendar type を返す callable。
            返り値は "open" / "early_close" / "full_close" / "weekend" を想定。
            実カレンダー取得（httpx）は呼び出し側の責務（このモジュールは純粋）。
        max_scan_days: 走査上限日数（サーキットブレーカー）。

    Returns:
        start より後の直近取引日。

    Raises:
        ValueError: max_scan_days 日走査しても取引日が見つからない場合
            （カレンダー異常）。Adapter 例外クラスへ依存しないため stdlib の
            ValueError を使う（このモジュールの純粋性を保つ）。
    """
    candidate = start + timedelta(days=1)
    for _ in range(max_scan_days):
        if schedule_lookup(candidate) in TRADING_DAY_TYPES:
            return candidate
        candidate += timedelta(days=1)

    raise ValueError(
        f"next_business_day: no trading day found within "
        f"{max_scan_days} days after {start}. Calendar may be inconsistent."
    )
