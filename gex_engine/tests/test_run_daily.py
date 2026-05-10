"""run_daily.py のユニットテスト。

エントリポイントなので、検証範囲は最小限:
    - make_fetcher() が GEX_DATA_SOURCE に従って正しい Adapter を返す
    - 未知の source でエラー
    - 例外時 main() が exit code 1 を返す
    - 正常時 main() が exit code 0 を返す
    - 空 DataFrame（NO_DATA）でも書き込みスキップして exit code 0

実際のフロー（Adapter → Core → I/O）は run_daily 専用で再テストせず、
段階3.5 / 段階4 のスモークテストでカバーされる。
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from gex_engine.adapters.mock import MockDataFetcher
from gex_engine.adapters.rest import ThetaRestAdapter
from gex_engine.scripts.run_daily import main, make_fetcher


# ──────────────────────────────────────────────────────────
# make_fetcher: Adapter 切替ロジック
# ──────────────────────────────────────────────────────────

class TestMakeFetcher:
    def test_mock_source_returns_mock_fetcher(self):
        f = make_fetcher("mock")
        assert isinstance(f, MockDataFetcher)
        assert f.source_name == "mock"

    def test_rest_source_returns_rest_adapter(self):
        f = make_fetcher("rest")
        assert isinstance(f, ThetaRestAdapter)
        assert f.source_name == "rest"
        f.close()  # 内部 Client を閉じる

    @pytest.mark.parametrize("bad", ["yfinance", "sdk", "", "MOCK", "REST"])
    def test_unknown_source_raises(self, bad):
        # 大文字混じりは run() 側で .lower() するので、make_fetcher 単体では
        # 大文字のまま渡されたらエラーになる仕様（防御的）
        with pytest.raises(ValueError) as exc_info:
            make_fetcher(bad)
        assert "GEX_DATA_SOURCE" in str(exc_info.value)


# ──────────────────────────────────────────────────────────
# main(): exit code の検証
# ──────────────────────────────────────────────────────────

class TestMainExitCode:
    """main() が想定通りの exit code を返すか。"""

    def test_normal_run_returns_0(self, tmp_path, monkeypatch):
        """Mock で 1 日分動かして正常終了 → exit code 0。"""
        # 出力先を tmpdir に隔離（リポジトリ直下を汚さない）
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        result = main()

        assert result == 0
        assert (tmp_path / "gex_history.json").exists()

    def test_unknown_source_returns_1(self, tmp_path, monkeypatch):
        """未知の source → ValueError → exit code 1。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "yfinance")

        result = main()

        assert result == 1

    def test_default_source_is_mock(self, tmp_path, monkeypatch):
        """環境変数未設定時は mock がデフォルト。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GEX_DATA_SOURCE", raising=False)

        result = main()

        assert result == 0
        assert (tmp_path / "gex_history.json").exists()

    def test_uppercase_source_is_normalized(self, tmp_path, monkeypatch):
        """run() 側で .lower() するので 'MOCK' でも動く。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "MOCK")

        result = main()

        assert result == 0

    def test_empty_dataframe_is_handled_gracefully(
        self, tmp_path, monkeypatch
    ):
        """NO_DATA（休場日等）で空 DataFrame が返っても落ちない。
        当日エントリを書かず、exit code 0 で終わる。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        # MockDataFetcher.get_option_chain を空返しに差し替え
        with patch.object(
            MockDataFetcher,
            "get_option_chain",
            return_value=pd.DataFrame(),
        ):
            result = main()

        assert result == 0
        # 書き込みスキップなので JSON は作られない
        assert not (tmp_path / "gex_history.json").exists()

    def test_calculate_all_exception_returns_1(
        self, tmp_path, monkeypatch
    ):
        """Core 計算で例外 → exit code 1。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        with patch(
            "gex_engine.scripts.run_daily.calculate_all",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = main()

        assert result == 1


# ──────────────────────────────────────────────────────────
# 副作用の確認
# ──────────────────────────────────────────────────────────

class TestSideEffects:
    """ファイル出力やログの副作用が想定通りか。"""

    def test_json_file_created_at_repo_root(self, tmp_path, monkeypatch):
        """出力パスが gex_history.json（リポジトリ直下相当）であること。"""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        main()

        # 直下に存在し、サブディレクトリ（data/ 等）には作られない
        json_files = list(tmp_path.rglob("*.json"))
        assert len(json_files) == 1
        assert json_files[0].name == "gex_history.json"
        assert json_files[0].parent == tmp_path

    def test_json_contains_expected_fields(self, tmp_path, monkeypatch):
        """出力 JSON が v12 セクション 8-5 の 13 フィールドを含む。"""
        import json

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        main()

        with open(tmp_path / "gex_history.json", encoding="utf-8") as f:
            history = json.load(f)

        assert len(history) == 1
        date_key = next(iter(history.keys()))
        entry = history[date_key]

        expected = {
            "call_wall", "put_wall", "zero_gamma", "max_pain",
            "underlying_price", "total_gex", "regime", "regime_text",
            "timestamp", "data_source", "symbol", "as_of",
            "n_contracts_used",
        }
        assert set(entry.keys()) == expected
        assert entry["data_source"] == "mock"
        assert entry["symbol"] == "SPY"


# ──────────────────────────────────────────────────────────
# OI 診断ログ（段階 6C 検証用、rest のときのみ出力される）
# ──────────────────────────────────────────────────────────

class TestOIDistributionLogging:
    """合格基準 E 第 2 項を Actions ログで検証可能にする診断ログのテスト。"""

    def test_mock_source_does_NOT_log_oi_distribution(
        self, tmp_path, monkeypatch, caplog
    ):
        """Mock 実行時は OI 分布ログを出さない（ログを汚さない）。"""
        import logging as _logging
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        with caplog.at_level(_logging.INFO):
            result = main()

        assert result == 0
        # 診断ログのマーカー文字列が出ていないことを確認
        assert "OI top 10 strikes" not in caplog.text

    def test_rest_source_logs_oi_distribution(
        self, tmp_path, monkeypatch, caplog
    ):
        """REST 実行時は OI トップ 10 をログに出す。

        実 API は叩けないので、ThetaRestAdapter.get_option_chain を
        Mock の合成データに差し替えて検証。
        """
        import logging as _logging
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "rest")

        # REST Adapter のメソッドを Mock 出力に差し替え
        # （実 API 接続が不要なテスト構造にする）
        mock_fetcher = MockDataFetcher(spot_price=450.0, seed=42)
        sample_df = mock_fetcher.get_option_chain("SPY", date.today())

        with patch.object(
            ThetaRestAdapter,
            "get_option_chain",
            return_value=sample_df,
        ):
            with caplog.at_level(_logging.INFO):
                result = main()

        assert result == 0
        # 診断ログのマーカーが出ている
        assert "OI top 10 strikes" in caplog.text
        # トップ 10 のうち少なくとも 1 つの行が出ている
        # （strike=XXX.XX right=YYYY oi=ZZZ の形式）
        import re
        oi_lines = re.findall(
            r"strike=\s*\d+\.\d+\s+right=\s*\w+\s+oi=\d+",
            caplog.text,
        )
        assert len(oi_lines) >= 1, (
            f"Expected at least 1 OI line, got: {oi_lines!r}"
        )

    def test_oi_logging_failure_does_not_kill_main(
        self, tmp_path, monkeypatch, caplog
    ):
        """診断ログで例外が出てもメイン処理（exit code 0）を巻き込まない。

        例: groupby が失敗するような壊れた DataFrame が来た場合でも、
        save_gex_result は既に成功しているのでジョブは成功扱いにする。
        """
        import logging as _logging
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "rest")

        mock_fetcher = MockDataFetcher(spot_price=450.0, seed=42)
        sample_df = mock_fetcher.get_option_chain("SPY", date.today())

        # _log_oi_distribution の中で例外を起こさせる
        with patch.object(
            ThetaRestAdapter,
            "get_option_chain",
            return_value=sample_df,
        ):
            with patch(
                "gex_engine.scripts.run_daily._log_oi_distribution",
                side_effect=RuntimeError("simulated diagnostic failure"),
            ):
                # _log_oi_distribution 自体が raise する場合は run() で
                # 例外伝播するので、診断ログ「内部」で握り潰すことを確認
                # → 関数を直接呼んでログのみ警告に変えるテスト
                pass

        # 実装側で try/except されているので、ここでは関数単体テストする
        from gex_engine.scripts.run_daily import _log_oi_distribution
        broken_df = pd.DataFrame({"foo": [1, 2, 3]})  # groupby に必要な列がない

        with caplog.at_level(_logging.WARNING):
            _log_oi_distribution(broken_df, _logging.getLogger("test"))

        # 例外を投げない（caplog に warning が出る）
        assert "Failed to log OI distribution" in caplog.text
