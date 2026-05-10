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
            as_of: 基準日。EOD 計算では「前営業日」を渡す想定。

        Returns:
            schema.REQUIRED_DTYPES に準拠した DataFrame。
            データなし（休場日等）の場合は schema.empty_dataframe() を返す。

        Raises:
            実装によって異なるが、ネットワーク・認証エラーは
            Adapter 内で握り潰さず呼び出し側に伝播させること。
        """
        ...
