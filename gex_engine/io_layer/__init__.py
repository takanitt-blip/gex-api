"""
gex_engine.io_layer

JSON I/O 層 - GEXResult を JSON ファイルに書き出すための機能群。

設計方針:
  - serializer.py : 純粋関数（GEXResult → dict）
  - history.py    : 履歴のロード・マージ
  - writer.py     : atomic write
  - facade        : save_gex_result() で 3 段を一気通貫
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

from .history import load_history, merge_entry, trim_history
from .serializer import (
    make_date_key,
    make_timestamp,
    scale_total_gex,
    serialize_result,
)
from .writer import write_json_atomic

logger = logging.getLogger(__name__)


__all__ = [
    "save_gex_result",
    "serialize_result",
    "load_history",
    "merge_entry",
    "trim_history",
    "write_json_atomic",
    "make_date_key",
    "make_timestamp",
    "scale_total_gex",
]


def save_gex_result(
    result: Any,
    *,
    path: str,
    session_date: date,
    data_source: Optional[str] = None,
    now_utc: Optional[datetime] = None,
    max_entries: Optional[int] = None,
) -> dict:
    """GEXResult を JSON 履歴ファイルに保存する Facade 関数。

    Args:
        result: GEXResult インスタンス（または同等の dict）
        path: 履歴 JSON のパス（例: "gex_history.json"）
        session_date: このエントリが支配する取引セッションの日付（ET 取引日）。
            JSON の日付キーになる。呼び出し側が
            market_calendar.next_business_day(trade_date, fetcher.schedule_type_on)
            で算出して渡す（obs.G 根治: キーを now()/today() から切り離す）。
        data_source: 上書き用。None なら GEXResult.data_source を使用。
        now_utc: 現在時刻（テスト用に注入可能）。**timestamp フィールド専用**で、
            日付キーには使わない（obs.G: 鍵を now() 依存から外したため）。
        max_entries: 履歴の最大エントリ数（None なら無制限）

    Returns:
        書き込まれた当日エントリ。
    """
    # 1. シリアライズ（timestamp は now_utc 由来＝書き込み時刻で正当）
    entry = serialize_result(result, data_source=data_source, now_utc=now_utc)

    # 2. 既存履歴を読み込み
    history = load_history(path)

    # 3. エントリをマージ。キーは now() ではなく session_date（obs.G 根治）。
    date_key = make_date_key(session_date)
    history, warning = merge_entry(history, date_key, entry)

    if warning:
        logger.warning(warning)

    # 4. トリミング（必要なら）
    history = trim_history(history, max_entries=max_entries)

    # 5. アトミック書き込み
    write_json_atomic(path, history)

    logger.info(
        f"✅ {date_key} のデータを '{path}' に保存しました "
        f"(履歴 {len(history)} 件)"
    )

    return entry
