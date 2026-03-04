"""
セッション判定モジュール
現在時刻からアジア/ロンドン/NY/ロンドン×NY重複セッションを判定する

MT5サーバー時間（EET: UTC+2冬/UTC+3夏）基準:
  - アジア:           00:00〜09:00
  - ロンドン直前:     09:00〜10:00（準備フェーズ・全エンジン停止）
  - ロンドン:         10:00〜15:00
  - ロンドン×NY重複:  15:00〜19:00（ボラ最大）
  - NY後半:           19:00〜00:00
  - 週末:             土曜〜日曜（金曜24:00クローズ = 土曜00:00）

※ EET/EESTはLondon/NYのDSTと連動するため、
  セッション境界はサーバー時間では季節を問わず固定。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

import pytz

# MT5サーバー時間: XMTradingは EET (Eastern European Time)
# 冬: UTC+2, 夏: UTC+3（pytzが自動切替）
SERVER_TZ = pytz.timezone("EET")

# 表示用 (ログ・ダッシュボード)
JST = pytz.timezone("Asia/Tokyo")


class Session(Enum):
    """トレーディングセッション"""
    ASIA = "asia"
    LONDON_PREP = "london_prep"    # ロンドン直前（準備フェーズ）
    LONDON = "london"
    OVERLAP = "overlap"            # ロンドン×NY重複
    NY_LATE = "ny_late"
    CLOSED = "closed"              # 週末・市場閉鎖


def get_current_session() -> Session:
    """
    現在のMT5サーバー時刻からセッションを判定

    Returns:
        Session: 現在のセッション
    """
    now = datetime.now(SERVER_TZ)
    return get_session_for_time(now)


def get_session_for_time(dt: datetime) -> Session:
    """
    指定時刻のセッションを判定

    Args:
        dt: 判定対象の時刻（timezone-awareを推奨）

    Returns:
        Session: 該当セッション
    """
    # タイムゾーン変換
    if dt.tzinfo is None:
        # naive datetime → サーバー時間と仮定
        dt = SERVER_TZ.localize(dt)
    else:
        dt = dt.astimezone(SERVER_TZ)

    # 週末チェック
    if _is_weekend(dt):
        return Session.CLOSED

    hour = dt.hour

    # セッション判定（サーバー時間基準）
    # London/NYのDSTとサーバーDSTが連動するため、
    # セッション時間は季節を問わず固定
    if 0 <= hour < 9:
        return Session.ASIA
    elif 9 <= hour < 10:
        return Session.LONDON_PREP
    elif 10 <= hour < 15:
        return Session.LONDON
    elif 15 <= hour < 19:
        return Session.OVERLAP
    elif 19 <= hour < 24:
        return Session.NY_LATE
    else:
        return Session.CLOSED


def _is_weekend(dt: datetime) -> bool:
    """
    週末判定（FX市場の週末）
    サーバー時間 土曜 00:00 〜 月曜 00:00 が市場閉鎖
    （金曜 17:00 ET = 金曜 24:00 サーバー = 土曜 00:00 サーバー）
    """
    weekday = dt.weekday()  # 0=月, 5=土, 6=日

    # 土曜・日曜 → クローズ
    if weekday >= 5:
        return True

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
        "timestamp_server": datetime.now(SERVER_TZ).isoformat(),
        "timestamp_jst": datetime.now(JST).isoformat(),
    }
