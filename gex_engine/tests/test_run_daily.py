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
        """出力 JSON が v17 の 12 フィールドを含む（正常時 = anomaly_detail なし）。

        v17 で regime / regime_text を削除し、data_quality を追加。
        正常な mock 実行では data_quality="ok" なので anomaly_detail は出ない。
        """
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
            "data_quality",
            "call_wall", "put_wall", "zero_gamma", "max_pain",
            "underlying_price", "total_gex", "z_position",
            "timestamp", "data_source", "symbol", "as_of",
            "n_contracts_used",
        }
        assert set(entry.keys()) == expected
        assert entry["data_quality"] == "ok"
        assert entry["z_position"] == "inside"  # 正常な mock は整序
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

        obs.G 後は run_daily が next_business_day(trade_date,
        fetcher.schedule_type_on) で session_date を求めるため、
        schedule_type_on も差し替えてカレンダーの生 HTTP を断つ
        （Terminal 非依存にする）。
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
        ), patch.object(
            ThetaRestAdapter,
            "schedule_type_on",
            return_value="open",   # 取引日扱い → next_business_day が生 HTTP を叩かない
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


# ──────────────────────────────────────────────────────────
# 誤判断25: trade_date が df 経由で calculate_all に渡る契約
# ──────────────────────────────────────────────────────────

class TestTradeDateFlow:
    """obs.F (run_daily as_of=today バグ) の再発防止テスト。

    run_daily は Adapter が出した trade_date 列から as_of を抽出して
    calculate_all に渡す契約 (誤判断25, 2026-05-24)。
    旧コードは date.today() を渡しており、土曜 cron で Adapter の
    解決値 (前金曜) と食い違って非取引日キーが JSON に書き込まれた。
    """

    def test_calculate_all_receives_trade_date_from_df(
        self, tmp_path, monkeypatch
    ):
        """calculate_all は df["trade_date"] の値を as_of として受け取る。

        cron 起動日 (today) ではなく、Adapter が解釈した T が渡されることを
        構造的に検証する。MockDataFetcher を差し替えて特定の trade_date を
        持つ df を返し、calculate_all の呼び出し引数を捕まえる。
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        # Adapter が「金曜の T」を返したシミュレーション
        # (cron の today が土曜でも Adapter は前金曜を解決する想定)
        adapter_resolved_t = date(2026, 5, 22)  # 金曜

        # 正常な df を Mock で作り、trade_date 列だけ差し替える
        base_fetcher = MockDataFetcher(spot_price=450.0, seed=42)
        sample_df = base_fetcher.get_option_chain("SPY", date.today())
        sample_df["trade_date"] = pd.Timestamp(adapter_resolved_t)

        # calculate_all の呼び出しを捕まえる
        captured_kwargs = {}

        def fake_calculate_all(df, **kwargs):
            captured_kwargs.update(kwargs)
            # 実物の calculate_all を呼んで成果物を返す
            # (main の続きが破綻しないように)
            from gex_engine.core.gex import calculate_all as real
            return real(df, **kwargs)

        with patch.object(
            MockDataFetcher, "get_option_chain", return_value=sample_df
        ):
            with patch(
                "gex_engine.scripts.run_daily.calculate_all",
                side_effect=fake_calculate_all,
            ):
                result = main()

        assert result == 0
        # ★ 契約: as_of は cron の today ではなく Adapter の T
        assert captured_kwargs.get("as_of") == adapter_resolved_t, (
            f"as_of must come from df['trade_date'] "
            f"(expected {adapter_resolved_t}, "
            f"got {captured_kwargs.get('as_of')})"
        )

    def test_missing_trade_date_column_raises(
        self, tmp_path, monkeypatch
    ):
        """Adapter が trade_date 列を忘れたら run_daily の assert で即死。

        将来 SDK Adapter 等の新規実装が trade_date 列を出し忘れた場合、
        run_daily 自身が拒否することを保証する構造的契約のテスト。
        AssertionError は main の try/except に拾われて exit code 1。
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEX_DATA_SOURCE", "mock")

        # trade_date 列を抜いた df (γ-1 以前の状態を模擬)
        base_fetcher = MockDataFetcher(spot_price=450.0, seed=42)
        sample_df = base_fetcher.get_option_chain("SPY", date.today())
        broken_df = sample_df.drop(columns=["trade_date"])
        assert "trade_date" not in broken_df.columns  # sanity check

        with patch.object(
            MockDataFetcher, "get_option_chain", return_value=broken_df
        ):
            result = main()

        # AssertionError が main の try/except で捕まり exit code 1
        assert result == 1
        # JSON は書かれていない (assert は GEX 計算の前)
        assert not (tmp_path / "gex_history.json").exists()
