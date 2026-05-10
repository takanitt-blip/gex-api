"""GEX 計算の結果オブジェクト。

設計方針:
    - frozen dataclass で「結果の改ざん」を防ぐ
    - 4 状態判定は別レイヤー（Step 1B）に任せる
      ここでは「水準と現在価格」だけ持つ
    - JSON シリアライズ可能な形（MT5 EA に渡す前提）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class GEXResult:
    """GEX 計算の出力。

    Attributes:
        symbol: シンボル
        as_of: 計算基準日時（ISO 8601 文字列）
        underlying_price: スポット価格
        call_wall: Call Wall 水準（最大の正の Net Gamma を持つストライク）
        put_wall: Put Wall 水準（最大の負の Net Gamma を持つストライク）
        zero_gamma: Zero Gamma 水準（None: 解なし）
        max_pain: Max Pain ストライク
        total_gex: ネット GEX 合計（参考値）
        n_contracts_used: 計算に使ったコントラクト数
        data_source: "mock" / "rest" / "sdk"
    """

    symbol: str
    as_of: str
    underlying_price: float
    call_wall: float
    put_wall: float
    zero_gamma: float | None
    max_pain: float
    total_gex: float
    n_contracts_used: int
    data_source: str

    def to_dict(self) -> dict[str, Any]:
        """JSON シリアライズ可能な dict に変換。"""
        return asdict(self)
