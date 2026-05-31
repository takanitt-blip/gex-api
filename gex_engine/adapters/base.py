"""DataFetcher 抽象化層（Adapter パターンの契約定義）。

PROJECT_CONTEXT v10 セクション7 で確定したアーキテクチャ:

    Core Logic
        ↓ (この Protocol を介して)
    DataFetcher Interface
        ↓
    [Mock] [REST] [SDK]

全 Adapter はこの Protocol に準拠した DataFrame を返す。
Core Logic は具体的な Adapter を知らない。

設計方針:
    - typing.Protocol を採用（duck typing 的、継承不要）
    - ABC は使わない（Mock を作る際の継承の手間を避ける）
    - メソッド数は最小限（YAGNI）
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DataFetcher(Protocol):
    """オプションチェーンデータを取得する契約。

    実装クラスは get_option_chain() を提供する必要がある。
    返り値は schema.REQUIRED_DTYPES に準拠した DataFrame であること。

    Attributes:
        source_name: "mock" / "rest" / "sdk" のいずれか。
                     ログ出力やデバッグで Adapter を識別するため。
    """

    source_name: str

    def get_option_chain(
        self, symbol: str, as_of: date
    ) -> pd.DataFrame:
        """指定銘柄・指定日のオプションチェーンを取得する。

        Args:
            symbol: 取得対象シンボル（例: "SPY"）
            as_of: 処理基準日（通常は実行日 today）。
                   EOD データの取得において「どの取引日のデータが
                   必要か」への変換（前営業日への調整等）は、各 Adapter が
                   必要に応じて内部で行う。呼び出し側は素直に処理基準日を
                   渡せばよい（DESIGN_history_rest_adapter.md 4.2）。

        Returns:
            schema.REQUIRED_DTYPES に準拠した DataFrame。
            データなし（休場日等）の場合は schema.empty_dataframe() を返す。

        Raises:
            実装によって異なるが、ネットワーク・認証エラーは
            Adapter 内で握り潰さず呼び出し側に伝播させること。
        """
        ...

    def schedule_type_on(self, target: date) -> str:
        """指定日の市場スケジュール type を返す。

        obs.G 根治（gex_history.json の日付キーを
        next_business_day(trade_date) に決定論化）のために、
        market_calendar.next_business_day へ注入する calendar lookup。

        Returns:
            "open" / "early_close" / "full_close" / "weekend" のいずれか。
        実装:
            rest -> calendar/on_date を叩く実カレンダー。
            mock -> 平日 open / 土日 weekend の素朴版（祝日は判定しない、CI 用）。
        """
        ...
