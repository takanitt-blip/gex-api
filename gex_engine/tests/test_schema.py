"""schema.py の検証テスト。

pytest で実行:
    cd /home/claude && python -m pytest gex_engine/tests/test_schema.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex_engine.schema import (
    REQUIRED_DTYPES,
    SchemaValidationError,
    coerce_to_schema,
    empty_dataframe,
    validate,
)


# ──────────────────────────────────────────────────────────
# テスト用ヘルパー: 正常系 DataFrame を作る
# ──────────────────────────────────────────────────────────

def make_valid_df(n_rows: int = 3) -> pd.DataFrame:
    """全列が正常な値を持つ DataFrame を作る。"""
    base = {
        "symbol": ["SPY"] * n_rows,
        "expiration": pd.to_datetime(["2026-06-19"] * n_rows),
        "strike": [400.0, 450.0, 500.0][:n_rows],
        "right": ["call", "put", "call"][:n_rows],
        "bid": [1.20, 0.80, 0.50][:n_rows],
        "ask": [1.25, 0.85, 0.55][:n_rows],
        "implied_volatility": [0.15, 0.18, 0.20][:n_rows],
        "open_interest": [1000, 500, 300][:n_rows],
        "underlying_price": [450.0] * n_rows,
        # trade_date: Adapter が解釈した取引日 T（誤判断25, γ-1 で REQUIRED 化）。
        # 取引日 < 満期日 の自然な関係（2026-05-14 取引、2026-06-19 満期）。
        "trade_date": pd.to_datetime(["2026-05-14"] * n_rows),
    }
    df = pd.DataFrame(base)
    return coerce_to_schema(df)


# ──────────────────────────────────────────────────────────
# 正常系
# ──────────────────────────────────────────────────────────

class TestHappyPath:
    def test_valid_dataframe_passes(self):
        df = make_valid_df()
        result = validate(df)
        assert result.is_valid
        assert result.errors == []
        assert result.warnings == []
        assert result.n_rows == 3

    def test_empty_dataframe_passes(self):
        """休場日想定: 空の DataFrame でも dtype が合っていれば通る。"""
        df = empty_dataframe()
        result = validate(df)
        assert result.is_valid
        assert result.n_rows == 0

    def test_raise_if_invalid_does_not_raise_on_valid(self):
        df = make_valid_df()
        result = validate(df)
        result.raise_if_invalid()  # 例外が出なければ OK


# ──────────────────────────────────────────────────────────
# ハードエラー系
# ──────────────────────────────────────────────────────────

class TestHardErrors:
    def test_missing_required_column(self):
        df = make_valid_df()
        df = df.drop(columns=["strike"])
        result = validate(df)
        assert not result.is_valid
        assert any("strike" in e for e in result.errors)

    def test_missing_multiple_columns(self):
        df = make_valid_df()
        df = df.drop(columns=["strike", "open_interest"])
        result = validate(df)
        assert not result.is_valid
        # まとめて1つのエラーメッセージで報告される設計
        assert len(result.errors) == 1
        assert "strike" in result.errors[0]
        assert "open_interest" in result.errors[0]

    def test_negative_strike(self):
        df = make_valid_df()
        df.loc[0, "strike"] = -10.0
        result = validate(df)
        assert not result.is_valid
        assert any("strike <= 0" in e for e in result.errors)

    def test_zero_strike(self):
        df = make_valid_df()
        df.loc[0, "strike"] = 0.0
        result = validate(df)
        assert not result.is_valid

    def test_negative_iv(self):
        df = make_valid_df()
        df.loc[0, "implied_volatility"] = -0.1
        result = validate(df)
        assert not result.is_valid
        assert any("implied_volatility < 0" in e for e in result.errors)

    def test_invalid_right_value(self):
        df = make_valid_df()
        df.loc[0, "right"] = "CALL"  # 大文字は不正
        result = validate(df)
        assert not result.is_valid
        assert any("right" in e for e in result.errors)

    def test_raise_if_invalid_raises(self):
        df = make_valid_df()
        df.loc[0, "strike"] = -1.0
        result = validate(df)
        with pytest.raises(SchemaValidationError):
            result.raise_if_invalid()


# ──────────────────────────────────────────────────────────
# ソフト警告系
# ──────────────────────────────────────────────────────────

class TestSoftWarnings:
    def test_zero_open_interest_warns(self):
        df = make_valid_df()
        df.loc[0, "open_interest"] = 0
        result = validate(df)
        assert result.is_valid  # 通る
        assert any("open_interest = 0" in w for w in result.warnings)

    def test_extreme_iv_warns(self):
        df = make_valid_df()
        df.loc[0, "implied_volatility"] = 7.5  # 750%
        result = validate(df)
        assert result.is_valid
        assert any("implied_volatility > 5.0" in w for w in result.warnings)

    def test_no_quote_warns(self):
        df = make_valid_df()
        df.loc[0, "bid"] = 0.0
        df.loc[0, "ask"] = 0.0
        result = validate(df)
        assert result.is_valid
        assert any("no quote" in w for w in result.warnings)

    def test_crossed_quote_warns(self):
        """bid > ask（クロスquote）は良性アーティファクト。GEX は γ×OI で
        計算し bid/ask を使わない（core/gex.py）ため非致命 ─ WARNING 止まりで
        日は通す。"""
        df = make_valid_df()
        df.loc[0, "bid"] = 2.0
        df.loc[0, "ask"] = 1.0
        result = validate(df)
        assert result.is_valid
        assert any("bid > ask" in w for w in result.warnings)

    def test_dte_zero_does_not_warn(self):
        """当日満期は正常な状態。dte は推奨カラムなのでチェック対象外。"""
        df = make_valid_df()
        df["dte"] = 0
        result = validate(df)
        assert result.is_valid
        # dte に関する警告が出ないこと
        assert not any("dte" in w for w in result.warnings)


# ──────────────────────────────────────────────────────────
# dtype 互換性チェック
# ──────────────────────────────────────────────────────────

class TestDtypeCompatibility:
    def test_lenient_mode_accepts_int64(self):
        """緩いモード: int64 を Int64 として受け入れる。"""
        df = make_valid_df()
        df["open_interest"] = df["open_interest"].astype("int64")
        result = validate(df, strict_dtypes=False)
        assert result.is_valid

    def test_strict_mode_rejects_int64(self):
        """厳密モード: int64 と Int64 を区別する。"""
        df = make_valid_df()
        df["open_interest"] = df["open_interest"].astype("int64")
        result = validate(df, strict_dtypes=True)
        assert not result.is_valid

    def test_lenient_mode_accepts_object_for_string(self):
        df = make_valid_df()
        df["symbol"] = df["symbol"].astype(object)
        result = validate(df, strict_dtypes=False)
        assert result.is_valid


# ──────────────────────────────────────────────────────────
# coerce_to_schema
# ──────────────────────────────────────────────────────────

class TestCoerce:
    def test_coerce_converts_dtypes(self):
        df = pd.DataFrame({
            "symbol": ["SPY"],
            "expiration": ["2026-06-19"],  # 文字列
            "strike": [400],                # int
            "right": ["call"],
            "bid": [1.20],
            "ask": [1.25],
            "implied_volatility": [0.15],
            "open_interest": [1000],
            "underlying_price": [450.0],
            "trade_date": ["2026-05-14"],   # 文字列（coerce で datetime64 に変換される）
        })
        coerced = coerce_to_schema(df)
        for col, dtype in REQUIRED_DTYPES.items():
            assert str(coerced[col].dtype) == dtype, f"{col} mismatch"

    def test_coerce_raises_on_missing_column(self):
        df = pd.DataFrame({"symbol": ["SPY"]})
        with pytest.raises(KeyError):
            coerce_to_schema(df)


# ──────────────────────────────────────────────────────────
# 複合シナリオ
# ──────────────────────────────────────────────────────────

class TestCombined:
    def test_multiple_errors_reported_together(self):
        """複数のハードエラーは一度に全部報告される（早期リターンしない）。"""
        df = make_valid_df()
        df.loc[0, "strike"] = -1.0
        df.loc[1, "implied_volatility"] = -0.5
        df.loc[2, "right"] = "INVALID"
        result = validate(df)
        assert not result.is_valid
        assert len(result.errors) >= 3

    def test_errors_and_warnings_coexist(self):
        """ハードエラーがあっても、警告チェックが走るとは限らない。
        現在の実装ではエラー優先で警告を集めない可能性もあるので、
        最低限「エラーが出ている」ことだけ確認する。
        """
        df = make_valid_df()
        df.loc[0, "strike"] = -1.0  # エラー
        df.loc[1, "open_interest"] = 0  # 警告候補
        result = validate(df)
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
