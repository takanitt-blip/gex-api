"""
GEXResult → JSON 出力用 dict への変換

責務:
  - GEX のスケール変換（素の単位 → × S² × 0.01）
  - 各値の桁数丸め（論点E: 値ごとに桁数最適化）
  - None の明示的な null 化
  - 日付キーの生成（論点C: ドット区切り。session_date ベース、obs.G 根治後。
    キーは呼び出し側が market_calendar.next_business_day で算出して渡す）

v17 変更（data_quality 導入）:
  - data_quality を出力の先頭に配置（EA が最初に読む）。
  - anomaly_detail は異常時のみ出力（"ok" のときキー自体を出さない）。
  - regime / regime_text の出力を削除（Python は regime を判定しない。
    判定は EA の責務 ─ PC_CORE §2.1）。_derive_regime も削除。
  - obs.G 根治で不要になった ET_TZ（死にコード）を撤去。

純粋関数（ファイル I/O なし）。テスト容易性のため独立。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional


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
        - as_of: str                    # ISO 8601 文字列（= 実取引日 T）
        - underlying_price: float
        - call_wall: float
        - put_wall: float
        - zero_gamma: float | None      # 解なしのとき None
        - max_pain: float
        - total_gex: float              # 素の単位（gamma × OI × 100）
        - n_contracts_used: int
        - data_source: str              # Adapter から伝搬
        - data_quality: str             # "ok" / "data_error" / "anomaly"
        - anomaly_detail: str | None    # 異常時のみ。正常時 None

    Args:
        result: GEXResult インスタンス（または dict / 任意オブジェクト）
        data_source: 上書き用。None なら GEXResult.data_source を使用。
                     Core Logic が伝搬した値を尊重しつつ、必要時のみ
                     外部から上書き可能（DRY 原則）。
        now_utc: 現在時刻（テスト用に注入可能）

    Returns:
        EA 互換のフラット dict + 分析用フィールド。
        data_quality が先頭。anomaly_detail は異常時のみ含む。
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

    # data_source 解決: 引数 > GEXResult.data_source > "unknown"
    # Core Logic が伝搬した値を優先（DRY 原則）。
    # 引数を明示的に渡された場合のみ override。
    actual_data_source = (
        data_source
        if data_source is not None
        else d.get("data_source", "unknown")
    )

    # data_quality は GEXResult が必ず持つ。欠けている入力（非 GEXResult の
    # 生 dict 等）は None のまま null 出力する ─ "ok" を黙ってデフォルトしない。
    # null は EA 側で != "ok" として全戦略 OFF に倒れる「目に見える失敗」になる。
    data_quality = d.get("data_quality")
    anomaly_detail = d.get("anomaly_detail")

    # data_quality を先頭に置く（dict の挿入順 = JSON のキー順）
    out: Dict[str, Any] = {"data_quality": data_quality}

    # anomaly_detail は異常時のみ出力（正常時はキー自体を出さない）
    if anomaly_detail is not None:
        out["anomaly_detail"] = anomaly_detail

    out.update({
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
        "timestamp":        make_timestamp(now_utc),
        "data_source":      actual_data_source,

        # 分析用（GEXResult が持っていれば出力、なければ None）
        "symbol":           d.get("symbol"),
        "as_of":            d.get("as_of"),
        "n_contracts_used": d.get("n_contracts_used"),
    })

    return out
