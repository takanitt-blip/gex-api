"""GEX 計算の結果オブジェクト。

設計方針:
    - frozen dataclass で「結果の改ざん」を防ぐ
    - 4 状態（regime）判定は EA 側の責務（v17 で確定、PC_CORE §2.1）。
      ここでは「水準と現在価格」+「地図の品質（data_quality）」だけ持つ。
    - JSON シリアライズ可能な形（MT5 EA に渡す前提）

v17 変更（data_quality 導入）:
    - data_quality / anomaly_detail を追加（PC_CORE §3）。
    - regime / regime_text は GEXResult には持たせない（Python は regime を
      判定しない。serializer も v17 で出力を削除）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GEXResult:
    """GEX 計算の出力。

    Attributes:
        symbol: シンボル
        as_of: 計算基準日時（ISO 8601 文字列、= 実取引日 T）
        underlying_price: スポット価格
        call_wall: Call Wall 水準（スポット以上で最大の正 Net GEX を持つストライク。
                   見つからない場合は spot にフォールバック ─ そのとき
                   data_quality="data_error" になる）
        put_wall: Put Wall 水準（スポット以下で最小の負 Net GEX を持つストライク。
                  同上のフォールバックあり）
        zero_gamma: Zero Gamma 水準（None: 解なし。v17 では None も
                    data_quality="data_error" として扱う ─ 論点c=c-1）
        max_pain: Max Pain ストライク
        total_gex: ネット GEX 合計（素の単位。serializer がスケール変換）
        n_contracts_used: 計算に使ったコントラクト数
        data_source: "mock" / "rest" / "sdk"
        data_quality: 地図の品質（PC_CORE §3）。
            "ok"          C >= Z >= P が成立、地図は使用可能
            "data_error"  Wall が見つからずフォールバック、または zero_gamma 解なし
                          （現行の spot 固定 Wall 探索では P>C は構造的に
                            起きないため、Wall 不検出が真のデータ品質エラー信号。
                            PC_CORE §3.3 の「P>C」文面は doc 側で要再整合）
            "anomaly"     Z > C または Z < P（Brent 解が Wall レンジ外 ＝ 構造崩壊）
        anomaly_detail: 異常時の自由形式説明。正常時（"ok"）は None。
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
    data_quality: str
    anomaly_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON シリアライズ可能な dict に変換。"""
        return asdict(self)
