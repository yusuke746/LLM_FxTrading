"""
セッション判定モジュール
現在時刻からアジア/ロンドン/NY/ロンドン×NY重複セッションを判定する

GMT+9（JST）基準:
  - アジア:           06:00〜15:00
  - ロンドン直前:     15:00〜16:00（準備フェーズ・全エンジン停止）
  - ロンドン:         16:00〜21:00
  - ロンドン×NY重複:  21:00〜翌01:00（ボラ最大）
  - NY後半:           01:00〜06:00
  - 週末:             土曜06:00〜月曜06:00
"""

from datetime import datetime
from enum import Enum
from typing import Optional

import pytz

JST = pytz.timezone("Asia/Tokyo")


class Session(Enum):
    """トレーディングセッション"""
    ASIA = "asia"
    LONDON_PREP = "london_prep"    # ロンドン直前（準備フェーズ）
    LONDON = "london"
    OVERLAP = "overlap"            # ロンドン×NY重複
    NY_LATE = "ny_late"
    CLOSED = "closed"              # 週末・市場閉鎖


# GMT+9基準のセッション境界 (start_hour, end_hour)
# 25=翌日01:00, 30=翌日06:00 として表現
SESSION_HOURS_JST = {
    Session.ASIA:        (6, 15),
    Session.LONDON_PREP: (15, 16),
    Session.LONDON:      (16, 21),
    Session.OVERLAP:     (21, 25),   # 21:00〜翌01:00
    Session.NY_LATE:     (25, 30),   # 翌01:00〜翌06:00 (=01:00〜06:00)
}


def get_current_session() -> Session:
    """
    現在のJST時刻からセッションを判定
    
    Returns:
        Session: 現在のセッション
    """
    now = datetime.now(JST)
    return get_session_for_time(now)


def get_session_for_time(dt: datetime) -> Session:
    """
    指定時刻のセッションを判定
    
    Args:
        dt: 判定対象の時刻（timezone-awareを推奨）
        
    Returns:
        Session: 該当セッション
    """
    # タイムゾーン変換（naive datetimeの場合はJSTと仮定）
    if dt.tzinfo is None:
        dt = JST.localize(dt)
    else:
        dt = dt.astimezone(JST)

    # 週末チェック
    if _is_weekend(dt):
        return Session.CLOSED

    hour = dt.hour

    # セッション判定
    if 6 <= hour < 15:
        return Session.ASIA
    elif hour == 15:
        return Session.LONDON_PREP
    elif 16 <= hour < 21:
        return Session.LONDON
    elif 21 <= hour <= 23:
        return Session.OVERLAP
    elif 0 <= hour < 1:
        return Session.OVERLAP        # 翌日 00:00〜01:00 もロンドン×NY重複
    elif 1 <= hour < 6:
        return Session.NY_LATE
    else:
        return Session.CLOSED


def _is_weekend(dt: datetime) -> bool:
    """
    週末判定（FX市場の週末）
    土曜 06:00 JST 〜 月曜 06:00 JST が市場閉鎖
    """
    weekday = dt.weekday()  # 0=月, 5=土, 6=日
    hour = dt.hour

    # 土曜 06:00以降
    if weekday == 5 and hour >= 6:
        return True
    # 日曜（終日）
    if weekday == 6:
        return True
    # 月曜 06:00未満
    if weekday == 0 and hour < 6:
        return True
    # 土曜 06:00未満（金曜深夜の延長 → まだ市場オープン中）
    return False


def is_market_open() -> bool:
    """市場がオープンしているか"""
    return get_current_session() != Session.CLOSED


def get_session_info() -> dict:
    """
    現在のセッション情報を辞書で返す
    
    Returns:
        dict: セッション名, 重み, 市場オープン状態
    """
    session = get_current_session()
    from session.weights import get_engine_weights

    weights = get_engine_weights(session)
    return {
        "session": session.value,
        "is_open": session != Session.CLOSED,
        "is_prep": session == Session.LONDON_PREP,
        "weights": weights,
        "timestamp": datetime.now(JST).isoformat(),
    }
