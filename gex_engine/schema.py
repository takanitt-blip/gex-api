"""
統一スキーマ定義（GEX 環境判別エンジン）

PROJECT_CONTEXT v10 セクション7「統一スキーマ」に基づく。
全ての DataFetcher Adapter（Mock / REST / SDK）は、
このスキーマに準拠した DataFrame を返す契約。

設計方針:
    - 案1（dtype dict + 検証関数）を採用
    - YAGNI: pandera 等の追加依存は導入しない
    - 異常値はハードエラー / ソフト警告 / 許容 の3層
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# スキーマ定義
# ──────────────────────────────────────────────────────────

# 必須カラム: 全 Adapter が必ず返すべき列
REQUIRED_DTYPES: dict[str, str] = {
    "symbol": "string",              # 例: "SPY"
    "expiration": "datetime64[ns]",  # pd.Timestamp（v10 論点1で決定）
    "strike": "float64",             # ストライク価格
    "right": "string",               # "call" or "put"
    "bid": "float64",                # NBBO bid
    "ask": "float64",                # NBBO ask
    "implied_volatility": "float64", # decimal スケール（v10 論点2: 0.15 = 15%）
    "open_interest": "Int64",        # 建玉（pandas nullable int）
    "underlying_price": "float64",   # スポット価格
}

# 推奨カラム: あれば望ましい列。Adapter が出せれば出す
RECOMMENDED_DTYPES: dict[str, str] = {
    "mid": "float64",                # (bid + ask) / 2
    "dte": "Int64",                  # Days to Expiration
    "timestamp": "datetime64[ns]",   # データ取得時刻
    "data_source": "string",         # "mock" / "rest" / "sdk"
}

# right 列の許容値
VALID_RIGHTS: set[str] = {"call", "put"}

# データソース識別子の許容値
VALID_DATA_SOURCES: set[str] = {"mock", "rest", "sdk"}


# ──────────────────────────────────────────────────────────
# 検証結果のデータクラス
# ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """validate() の結果を保持する。

    Attributes:
        is_valid: ハードエラーが無ければ True
        errors: ハードエラーのメッセージ一覧
        warnings: ソフト警告のメッセージ一覧
        n_rows: 入力 DataFrame の行数
    """
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    n_rows: int

    def raise_if_invalid(self) -> None:
        """ハードエラーがあれば例外を投げる。"""
        if not self.is_valid:
            raise SchemaValidationError(
                f"Schema validation failed with {len(self.errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )


class SchemaValidationError(ValueError):
    """スキーマ検証で致命的エラーが見つかったときに送出。"""


# ──────────────────────────────────────────────────────────
# 検証本体
# ──────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, *, strict_dtypes: bool = False) -> ValidationResult:
    """DataFrame が統一スキーマに準拠しているか検証する。

    Args:
        df: 検証対象の DataFrame
        strict_dtypes: True なら dtype の完全一致を要求。
                       False（デフォルト）なら数値・日付の互換変換を許容。

    Returns:
        ValidationResult: errors と warnings を含む結果オブジェクト。

    Raises:
        例外は投げない。明示的にチェックしたい場合は
        result.raise_if_invalid() を呼ぶ。

    異常値の3層:
        ハードエラー（errors）:
            ・必須列の欠損
            ・dtype の不一致（strict_dtypes=True 時）
            ・strike <= 0
            ・bid > ask
            ・IV < 0
            ・right が "call"/"put" 以外
        ソフト警告（warnings）:
            ・OI = 0
            ・IV > 5.0（500%超）
            ・bid = 0 かつ ask = 0
        許容（チェックしない）:
            ・dte = 0
            ・OI が極端に大きい
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── 必須列の存在チェック ──
    missing = set(REQUIRED_DTYPES.keys()) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        # 必須列が無ければ後続の値チェックは不可能なので早期リターン
        return ValidationResult(
            is_valid=False, errors=errors, warnings=warnings, n_rows=len(df)
        )

    # ── dtype チェック ──
    for col, expected in REQUIRED_DTYPES.items():
        actual = str(df[col].dtype)
        if actual != expected:
            msg = f"Column '{col}' has dtype '{actual}', expected '{expected}'"
            if strict_dtypes:
                errors.append(msg)
            else:
                # 緩いモード: 数値系・日付系の互換は許容
                if not _is_compatible_dtype(df[col], expected):
                    errors.append(msg)

    # 値レベルのチェックに進む前に、dtype エラーがあれば中断
    # （型が違うと比較演算で TypeError になる）
    if errors:
        return ValidationResult(
            is_valid=False, errors=errors, warnings=warnings, n_rows=len(df)
        )

    # ── 値レベルのハードエラーチェック ──

    # right 列: "call" or "put" のみ許容
    invalid_rights = set(df["right"].dropna().unique()) - VALID_RIGHTS
    if invalid_rights:
        errors.append(
            f"Column 'right' contains invalid values: {sorted(invalid_rights)}. "
            f"Allowed: {sorted(VALID_RIGHTS)}"
        )

    # strike <= 0 は数学的にあり得ない
    n_bad_strike = (df["strike"] <= 0).sum()
    if n_bad_strike > 0:
        errors.append(f"{n_bad_strike} row(s) have strike <= 0")

    # bid > ask は板の論理破綻
    # 両方が NaN や 0 のケースは別途警告で扱うので、ここでは厳密な不等号のみ
    bid_gt_ask_mask = (df["bid"] > df["ask"]) & df["bid"].notna() & df["ask"].notna()
    n_bad_quote = bid_gt_ask_mask.sum()
    if n_bad_quote > 0:
        errors.append(f"{n_bad_quote} row(s) have bid > ask (crossed quote)")

    # IV < 0 は数学的にあり得ない
    n_bad_iv = (df["implied_volatility"] < 0).sum()
    if n_bad_iv > 0:
        errors.append(f"{n_bad_iv} row(s) have implied_volatility < 0")

    # ── ソフト警告チェック ──

    # OI = 0: 実在するが GEX 計算には寄与ゼロ
    n_zero_oi = (df["open_interest"] == 0).sum()
    if n_zero_oi > 0:
        warnings.append(f"{n_zero_oi} row(s) have open_interest = 0")

    # IV > 5.0 (500% 超): 極端だが ThetaData で発生し得る
    n_extreme_iv = (df["implied_volatility"] > 5.0).sum()
    if n_extreme_iv > 0:
        warnings.append(
            f"{n_extreme_iv} row(s) have implied_volatility > 5.0 (>500%)"
        )

    # bid = 0 かつ ask = 0: 板なし
    no_quote_mask = (df["bid"] == 0) & (df["ask"] == 0)
    n_no_quote = no_quote_mask.sum()
    if n_no_quote > 0:
        warnings.append(f"{n_no_quote} row(s) have no quote (bid=0 and ask=0)")

    # 警告は logger 経由でも流す（呼び出し側の利便性のため）
    for w in warnings:
        logger.warning("Schema validation: %s", w)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        n_rows=len(df),
    )


# ──────────────────────────────────────────────────────────
# 内部ヘルパー
# ──────────────────────────────────────────────────────────

def _is_compatible_dtype(series: pd.Series, expected: str) -> bool:
    """緩い dtype 一致判定。

    例:
        actual='int64', expected='Int64'        → True（int → nullable int）
        actual='datetime64[us]', expected='datetime64[ns]'  → True
        actual='object', expected='string'      → True（明示変換前提）
    """
    actual_dtype = series.dtype

    # 数値系の互換
    if expected.startswith("Int") or expected.startswith("int"):
        return pd.api.types.is_integer_dtype(actual_dtype)
    if expected == "float64":
        return pd.api.types.is_float_dtype(actual_dtype)

    # 日付系の互換（ns / us / ms 等の解像度差は許容）
    if expected.startswith("datetime64"):
        return pd.api.types.is_datetime64_any_dtype(actual_dtype)

    # 文字列系（pandas の string と object は実用上交換可能）
    if expected == "string":
        return (
            pd.api.types.is_string_dtype(actual_dtype)
            or actual_dtype == object
        )

    return False


def coerce_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame を必須カラムの dtype に強制変換する。

    Adapter が外部データを取得した直後に呼ぶことを想定。
    必須カラムが揃っていない場合は KeyError を投げる。

    Returns:
        新しい DataFrame（元の df は変更しない）
    """
    out = df.copy()
    for col, dtype in REQUIRED_DTYPES.items():
        if col not in out.columns:
            raise KeyError(f"Cannot coerce: required column '{col}' is missing")
        try:
            out[col] = out[col].astype(dtype)
        except (ValueError, TypeError) as e:
            raise SchemaValidationError(
                f"Failed to coerce column '{col}' to {dtype}: {e}"
            ) from e
    return out


def empty_dataframe() -> pd.DataFrame:
    """正しい dtype を持つ空の DataFrame を返す。

    休場日や該当データなしの場合に Adapter が返すべき形。
    """
    return pd.DataFrame({
        col: pd.Series(dtype=dtype)
        for col, dtype in REQUIRED_DTYPES.items()
    })
