"""
GEXResult → JSON 出力用 dict への変換

責務:
  - GEX のスケール変換（素の単位 → × S² × 0.01）
  - 各値の桁数丸め（論点E: 値ごとに桁数最適化）
  - None の明示的な null 化
  - 日付キーの生成（論点C: ドット区切り、ET 基準）

純粋関数（ファイル I/O なし）。テスト容易性のため独立。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, Optional


# ============================================================
# タイムゾーン定数
# ============================================================
# ET（米国東部時間）: 夏時間中は UTC-4、標準時は UTC-5
# zoneinfo で DST を正確に扱う
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    ET_TZ = ZoneInfo("America/New_York")
except ImportError:  # pragma: no cover
    # フォールバック: 固定オフセット（DST を考慮しない）
    ET_TZ = timezone(timedelta(hours=-5))


# ============================================================
# スケール変換（論点A の決定事項）
# ============================================================
def scale_total_gex(raw_total_gex: float, spot: float) -> float:
    """
    Total GEX を業界標準スケール（× S^2 × 0.01）に変換。

    意味: 原資産が 1% 動いたときのドル建てデルタ変化額。
    既存 update_gex.py および SpotGamma 等の業界標準と桁を揃える。

    Args:
        raw_total_gex: Core Logic が返す素の Total GEX（gamma × OI × 100 の合計）
        spot: 計算時点の原資産価格

    Returns:
        スケール済み Total GEX（$ / 1% move）
    """
    return raw_total_gex * (spot ** 2) * 0.01


# ============================================================
# 日付・時刻
# ============================================================
def make_date_key(session_date: date) -> str:
    """EA が読む日付キーを生成（"YYYY.MM.DD" 形式）。

    obs.G 根治: キーは「このEOD地図が支配する取引セッション
    = next_business_day(trade_date)」を表す。呼び出し側が
    market_calendar.next_business_day で算出して渡す。now() には依存しない。
    session_date は既にカレンダー由来の取引日なので TZ 変換は不要。
    """
    return session_date.strftime("%Y.%m.%d")


def make_timestamp(now_utc: Optional[datetime] = None) -> str:
    """
    取得時刻のタイムスタンプを ISO 8601 (UTC) で生成。

    用途:
      - 同日複数回実行の警告ログ（論点D）でいつのデータか比較
      - データ鮮度の確認

    Returns:
        "2026-05-09T22:30:15Z" 形式
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    # マイクロ秒は捨てる（JSON サイズ削減 + 過剰精度の排除）
    now_utc = now_utc.replace(microsecond=0)
    # "+00:00" を "Z" に変換（ISO 8601 標準形）
    return now_utc.isoformat().replace("+00:00", "Z")


# ============================================================
# 丸め（論点E）
# ============================================================
def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    """None を保ったまま丸める。NaN/Inf は None に正規化。"""
    if value is None:
        return None
    # NaN チェック（NaN は自分自身と等しくない）
    if value != value:
        return None
    if value in (float("inf"), float("-inf")):
        return None
    return round(float(value), digits)


def _to_int_or_none(value: Optional[float]) -> Optional[int]:
    """None を保ったまま int に丸める。NaN/Inf は None に正規化。"""
    if value is None:
        return None
    if value != value:  # NaN
        return None
    if value in (float("inf"), float("-inf")):
        return None
    return int(round(float(value)))


# ============================================================
# レジーム判定（既存 update_gex.py 互換の最小実装）
# ============================================================
def _derive_regime(scaled_total_gex: float) -> tuple[str, str]:
    """
    Total GEX の符号からレジーム判定。

    注意:
      - これは v11 セクション9 で「Step 1B 予定」とされた静的判定の
        最小実装。Step 1B では「現在価格 vs Wall/Zero Gamma」の
        位置関係も加味した 4 状態判定に拡張される。
      - ここでは既存 update_gex.py 互換のため total_gex の符号のみで判定。
      - スケール変換は符号を変えないため、scaled でも raw でも結果は同じ。
    """
    if scaled_total_gex > 0:
        return "range", "レンジ相場・低ボラティリティ"
    else:
        return "trend", "トレンド相場・高ボラティリティ"


# ============================================================
# メイン関数
# ============================================================
def serialize_result(
    result: Any,
    *,
    data_source: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    GEXResult を JSON 出力用 dict に変換する。

    実 GEXResult の構造（gex_engine.core.result.GEXResult、frozen dataclass）:
        - symbol: str
        - as_of: str                    # ISO 8601 文字列
        - underlying_price: float
        - call_wall: float
        - put_wall: float
        - zero_gamma: float | None      # 解なしのとき None
        - max_pain: float               # 数学的に必ず解あり（None 不可）
        - total_gex: float              # 素の単位（gamma × OI × 100）
        - n_contracts_used: int
        - data_source: str              # Adapter から伝搬

    Args:
        result: GEXResult インスタンス（または dict / 任意オブジェクト）
        data_source: 上書き用。None なら GEXResult.data_source を使用。
                     Core Logic が伝搬した値を尊重しつつ、必要時のみ
                     外部から上書き可能（DRY 原則）。
        now_utc: 現在時刻（テスト用に注入可能）

    Returns:
        EA 互換のフラット dict + 分析用フィールド
    """
    # dataclass でも dict でも受け付けられるよう正規化
    if is_dataclass(result):
        d = asdict(result)
    elif isinstance(result, dict):
        d = result
    else:
        # __dict__ がある任意のオブジェクト
        d = {k: v for k, v in vars(result).items() if not k.startswith("_")}

    # 必須フィールドの取得
    spot = d.get("underlying_price")
    raw_total_gex = d.get("total_gex")

    if spot is None or raw_total_gex is None:
        raise ValueError(
            "serialize_result: result に underlying_price と total_gex は必須"
        )

    # スケール変換
    scaled_total_gex = scale_total_gex(raw_total_gex, spot)

    # レジーム判定（GEXResult が既に持っていればそれを優先、なければ導出）
    regime = d.get("regime")
    regime_text = d.get("regime_text")
    if regime is None or regime_text is None:
        regime, regime_text = _derive_regime(scaled_total_gex)

    # data_source 解決: 引数 > GEXResult.data_source > "unknown"
    # Core Logic が伝搬した値を優先（DRY 原則）。
    # 引数を明示的に渡された場合のみ override。
    actual_data_source = (
        data_source
        if data_source is not None
        else d.get("data_source", "unknown")
    )

    return {
        # 価格水準（小数 2 桁）
        "call_wall":        _round_or_none(d.get("call_wall"),        2),
        "put_wall":         _round_or_none(d.get("put_wall"),         2),
        "zero_gamma":       _round_or_none(d.get("zero_gamma"),       2),
        "max_pain":         _round_or_none(d.get("max_pain"),         2),
        "underlying_price": _round_or_none(spot,                      2),

        # GEX 値（整数、スケール変換済み × S^2 × 0.01）
        # int 変換で JSON 上 "13004123" と表記（".0" 抑止で可読性 ↑）
        "total_gex":        _to_int_or_none(scaled_total_gex),

        # メタ
        "regime":           regime,
        "regime_text":      regime_text,
        "timestamp":        make_timestamp(now_utc),
        "data_source":      actual_data_source,

        # 分析用（GEXResult が持っていれば出力、なければ None）
        "symbol":           d.get("symbol"),
        "as_of":            d.get("as_of"),
        "n_contracts_used": d.get("n_contracts_used"),
    }
