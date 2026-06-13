"""当日満期（expiration <= as_of, DTE<=0）除外フィルタの回帰テスト（誤判断32）。

背景:
    地図は EOD(T) から計算され翌セッション T+1 をガバナンスする。にもかかわらず
    expiration <= as_of の当日満期/期限切れオプションを GEX 計算に含めていたため、
    T→0 で γ が床値まで爆発し、その建玉が spot 近傍の 1 strike に Net GEX を集中させて
    Call/Put Wall を spot にピンさせていた。Zero Gamma は本体（長期物）の符号反転点に
    座るので、両者の DTE 不整合から Z∉[P,C]（anomaly）が量産されていた。

    実データ（例 2025.03.13）では当日満期除外で Call Wall が 559→650 へ戻り anomaly が
    解消することを確認済み。本テストはその機構を合成チェーンで固定する。
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from gex_engine.core.gex import calculate_all, calculate_gex_per_strike, find_call_wall
from gex_engine.schema import coerce_to_schema

AS_OF = date(2025, 6, 2)            # 月曜（取引日 T）
SAME_DAY = pd.Timestamp("2025-06-02")   # 当日満期（DTE=0, 死んだ建玉）
NEXT_SESSION = pd.Timestamp("2025-06-03")  # 翌セッション満期（DTE=1, 生きている）
THIRTY_D = pd.Timestamp("2025-07-02")      # 30日物
SPOT = 600.0


def _row(exp, strike, right, oi):
    return dict(
        symbol="SPY", expiration=exp, strike=float(strike), right=right,
        bid=1.0, ask=1.1, implied_volatility=0.18, open_interest=int(oi),
        underlying_price=SPOT, trade_date=pd.Timestamp(AS_OF),
    )


def _chain(rows):
    return coerce_to_schema(pd.DataFrame(rows))


def test_same_day_expiry_does_not_pin_wall():
    """当日満期の ATM パイルが壁を spot にピンさせるが、フィルタ後は本来の OTM 壁が出る。"""
    rows = []
    # 背景（balanced、影響なし）
    for K in range(560, 641, 5):
        rows += [_row(THIRTY_D, K, "call", 800), _row(THIRTY_D, K, "put", 800)]
    # 本来の壁：30日物 OTM（call 山=625, put 山=575）
    rows += [_row(THIRTY_D, 625, "call", 6000), _row(THIRTY_D, 575, "put", 6000)]
    # ノイズ源：当日満期の ATM net-call パイル（601）。0DTE の γ 爆発で壁を spot にピンさせる
    rows += [_row(SAME_DAY, 601, "call", 8000), _row(SAME_DAY, 601, "put", 500)]
    df = _chain(rows)

    # 前提の確認：フィルタ無しの per-strike GEX では 601(≈spot) が argmax＝ピンが実在する
    gx_full = calculate_gex_per_strike(df, AS_OF)
    pinned = find_call_wall(gx_full, SPOT)
    assert pinned == 601.0, f"前提崩れ: 当日満期がピンを作っていない (got {pinned})"

    # 本題：calculate_all（フィルタ込み）では当日満期が除外され、本来の OTM 壁に戻る
    res = calculate_all(df, as_of=AS_OF, data_source="rest")
    assert res.call_wall == 625.0, f"当日満期が除外されず壁がピンしている (got {res.call_wall})"
    assert res.call_wall > SPOT + 15, "壁が spot 近傍にピンしたまま"
    assert res.data_quality == "ok", f"anomaly が解消していない (dq={res.data_quality})"


def test_next_session_expiry_is_retained():
    """T+1 満期（DTE=1, セッションの 0DTE）は除外されず壁の計算に使われる。"""
    rows = []
    for K in range(560, 641, 5):
        rows += [_row(THIRTY_D, K, "call", 500), _row(THIRTY_D, K, "put", 500)]
    # 唯一の支配的 net 構造を T+1 満期にのみ置く
    rows += [_row(NEXT_SESSION, 615, "call", 9000), _row(NEXT_SESSION, 615, "put", 500)]
    rows += [_row(NEXT_SESSION, 585, "put", 9000), _row(NEXT_SESSION, 585, "call", 500)]
    df = _chain(rows)

    res = calculate_all(df, as_of=AS_OF, data_source="rest")
    # T+1 が保持されていれば 615/585 が壁になる（落とされていれば検出されない）
    assert res.call_wall == 615.0, f"T+1 満期が落とされている (call_wall={res.call_wall})"
    assert res.put_wall == 585.0, f"T+1 満期が落とされている (put_wall={res.put_wall})"


def test_all_same_day_expiry_raises():
    """全行が当日満期なら、生きた建玉ゼロで明示的に失敗する（黙って壊れない）。"""
    rows = [_row(SAME_DAY, 600, "call", 1000), _row(SAME_DAY, 600, "put", 1000)]
    df = _chain(rows)
    try:
        calculate_all(df, as_of=AS_OF, data_source="rest")
        assert False, "空になるべきところで例外が出ていない"
    except ValueError as e:
        assert "live options" in str(e)
