"""
atomic write による安全な JSON 書き込み

責務:
  - 一時ファイルに書き、os.replace でアトミックに置換
  - 書き込み中の事故（タイムアウト、kill）で履歴ファイルが
    破壊されることを防ぐ（論点G: G2 案）

POSIX / Windows どちらでも os.replace は同一ボリューム内で
アトミック動作する。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def write_json_atomic(
    path: str,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    JSON ファイルをアトミックに書き込む。

    手順:
      1. 同じディレクトリに一時ファイルを作る（os.replace のために
         同一ボリュームである必要がある）
      2. 一時ファイルに JSON を書き、fsync で OS バッファをフラッシュ
      3. os.replace で目的のパスに置換（アトミック）

    例外発生時は一時ファイルを削除して例外を再送出する。

    Args:
        path: 書き込み先のパス
        data: JSON シリアライズ可能なオブジェクト
        indent: JSON のインデント（デフォルト 2、可読性重視）
        ensure_ascii: False なら日本語をそのまま出力（デフォルト）

    Raises:
        OSError: ディレクトリが存在しない、権限がない等
        TypeError: data がシリアライズ不可
    """
    dirname = os.path.dirname(path) or "."

    if not os.path.isdir(dirname):
        raise OSError(f"出力先ディレクトリが存在しません: {dirname}")

    # 同じディレクトリに一時ファイルを作る
    # （別ディレクトリだと os.replace が EXDEV エラーになる場合がある）
    fd, tmp_path = tempfile.mkstemp(
        dir=dirname,
        prefix=".tmp_",
        suffix=".json",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.flush()
            os.fsync(f.fileno())  # OS バッファを物理ディスクに書き出す

        # アトミックな置換
        os.replace(tmp_path, path)

    except Exception:
        # 失敗したら一時ファイルを掃除
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass  # 掃除に失敗しても、元の例外を優先
        raise
