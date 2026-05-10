"""
履歴 JSON の読み込みとマージ

責務:
  - 既存履歴 JSON の安全な読み込み（破損時は空辞書）
  - 同日データの上書き + 警告ログ（論点D: D4 案）
  - エントリ数による履歴ファイルのトリミング（オプション）

純粋関数（書き込みは writer.py が担当）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# 読み込み
# ============================================================
def load_history(path: str) -> Dict[str, Dict[str, Any]]:
    """
    履歴 JSON を読み込む。

    存在しない / 破損している / 空ファイル / 辞書でない 場合は
    空辞書を返す（エラーで止めない、安全側に倒す）。

    既存 update_gex.py の同等実装より厳格:
      - JSONDecodeError 以外（例: ファイル権限エラー）も握りつぶさない
      - トップレベルが dict でない場合も警告を出す

    Args:
        path: 履歴 JSON のパス

    Returns:
        { "2026.05.09": {...}, "2026.05.08": {...}, ... }
    """
    if not os.path.exists(path):
        logger.info(f"履歴ファイルが存在しないため新規作成します: {path}")
        return {}

    if os.path.getsize(path) == 0:
        logger.warning(f"履歴ファイルが空です: {path}")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # 既存実装と同様に「破損なら新規作成」だが、
        # ログレベルを WARNING に上げて気づきやすくする
        logger.warning(f"履歴ファイルが破損しています、新規作成します: {path} ({e})")
        return {}

    if not isinstance(data, dict):
        logger.warning(
            f"履歴ファイルのトップレベルが dict ではありません: "
            f"{type(data).__name__}, 新規作成します"
        )
        return {}

    return data


# ============================================================
# 差分検知（論点D: D4 案 - 上書き + 警告ログ）
# ============================================================
def _values_differ_meaningfully(
    old: Dict[str, Any], new: Dict[str, Any]
) -> bool:
    """
    同日データに「意味のある」差分があるかを判定。

    timestamp と data_source は除外して比較する。
    理由: timestamp は実行のたびに変わるが、計算値が同じなら
          差分ログを出す価値はない（ノイズになる）。
    """
    # 比較対象から除外するメタフィールド
    META_FIELDS = {"timestamp", "data_source"}

    old_filtered = {k: v for k, v in old.items() if k not in META_FIELDS}
    new_filtered = {k: v for k, v in new.items() if k not in META_FIELDS}

    return old_filtered != new_filtered


# ============================================================
# マージ
# ============================================================
def merge_entry(
    history: Dict[str, Dict[str, Any]],
    date_key: str,
    new_entry: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    履歴に当日エントリを追加 or 上書きする。

    動作（論点D: D4 案）:
      - 同日キーが存在しない: 単純追加
      - 同日キーが存在し、計算値が同じ: 静かに上書き（timestamp 更新のみ）
      - 同日キーが存在し、計算値が異なる: 上書き + 警告メッセージ返却

    破壊的変更を避けるため、history は変更せず新しい dict を返す。

    Args:
        history: 既存履歴
        date_key: "2026.05.09" 形式
        new_entry: serialize_result() の戻り値

    Returns:
        (更新後の履歴, 警告メッセージ or None)
        警告メッセージは「同日に異なる値で上書きされた」場合のみ非 None。
    """
    new_history = dict(history)  # シャローコピー
    warning: Optional[str] = None

    if date_key in history:
        old_entry = history[date_key]
        if _values_differ_meaningfully(old_entry, new_entry):
            warning = (
                f"⚠️  同日 {date_key} のデータが上書きされました（計算値に差分あり）\n"
                f"   旧: {_format_for_log(old_entry)}\n"
                f"   新: {_format_for_log(new_entry)}"
            )

    new_history[date_key] = new_entry
    return new_history, warning


def _format_for_log(entry: Dict[str, Any]) -> str:
    """ログ出力用の簡潔な整形（主要 4 フィールドのみ）"""
    keys = ("call_wall", "put_wall", "zero_gamma", "total_gex")
    parts = []
    for k in keys:
        v = entry.get(k)
        parts.append(f"{k}={v}")
    return ", ".join(parts)


# ============================================================
# トリミング（オプション機能）
# ============================================================
def trim_history(
    history: Dict[str, Dict[str, Any]],
    max_entries: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    履歴を最新 max_entries 件に絞る。

    EA は InpDisplayDays=30 を見ているが、分析用途では履歴を
    長く残したい。デフォルト None（無制限）で運用し、必要に
    なってから有効化する。

    Args:
        history: 既存履歴
        max_entries: None なら無制限、整数なら最新 N 件のみ保持

    Returns:
        トリム済み履歴
    """
    if max_entries is None or len(history) <= max_entries:
        return history

    # 日付キーを昇順ソートして、最新の max_entries 件だけ残す
    # "YYYY.MM.DD" 形式は文字列ソートで日付順になる
    sorted_keys = sorted(history.keys())
    keep_keys = set(sorted_keys[-max_entries:])
    return {k: v for k, v in history.items() if k in keep_keys}
