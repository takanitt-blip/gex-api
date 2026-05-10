"""Core Logic の計算パラメータ。

PROJECT_CONTEXT v10 + 数式設計議論で確定した値。
全てデフォルト値を持つが、必要に応じて上書き可能。

設計方針:
    - frozen dataclass で「うっかり書き換え」を防ぐ
    - 設定を変更したい場合は新インスタンスを作る
    - マジックナンバーは全てここに集約
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GEXConfig:
    """GEX 計算の設定。

    Attributes:
        risk_free_rate: 無リスク金利（年率、decimal）
            ・gamma への影響は微小（< 1%）
            ・設定可能だが頻繁に変更する必要なし
        dividend_yield: 配当利回り（年率、decimal）
            ・銘柄横断比較の一貫性のためデフォルト 0
        contract_size: 1コントラクトあたりの株数
            ・米国オプションは 100 株固定
        min_time_to_expiry: T の最小値（年単位）
            ・0DTE のゼロ除算防止
            ・0.5/365 ≈ 12時間相当
        zero_gamma_search_pct: Zero Gamma 探索範囲のフォールバック幅
            ・ストライク min/max が使えない場合に使用
            ・S × (1 ± この値) の範囲を探索
    """

    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    contract_size: int = 100
    min_time_to_expiry: float = 0.5 / 365.0
    zero_gamma_search_pct: float = 0.20


# モジュールレベルのデフォルト
# 大半のコードはこれを使う。テストや特殊用途で上書きする場合のみ
# 新しい GEXConfig() を作る。
DEFAULT_CONFIG = GEXConfig()
