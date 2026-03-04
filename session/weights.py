"""
セッション別エンジン重みモジュール
各セッションにおけるトレンド/逆張り/ブレイクアウトエンジンの重み付けを管理

★の数がシグナルスコアへの寄与度:
  アジア:           逆張り★★★, トレンド★, ブレイクアウト★（抑制）
  ロンドン直前:     全停止
  ロンドン:         トレンド★★★, ブレイクアウト★★★, 逆張り★（抑制）
  ロンドン×NY重複:  ブレイクアウト★★★★, トレンド★★★, 逆張り❌
  NY後半:           トレンド★★, 逆張り★★, ブレイクアウト★
"""

from typing import Dict, Tuple

from session.detector import Session


# セッション別エンジン重み: (trend, mean_reversion, breakout)
ENGINE_WEIGHTS: Dict[Session, Tuple[float, float, float]] = {
    #                          trend  mean_rev  breakout
    Session.ASIA:             (0.5,   1.0,      0.3),
    Session.LONDON_PREP:      (0.0,   0.0,      0.0),    # 全停止（準備）
    Session.LONDON:           (1.0,   0.3,      1.0),
    Session.OVERLAP:          (1.0,   0.0,      1.5),    # breakout最優先
    Session.NY_LATE:          (0.8,   0.7,      0.5),
    Session.CLOSED:           (0.0,   0.0,      0.0),    # 週末停止
}


def get_engine_weights(session: Session) -> Dict[str, float]:
    """
    指定セッションのエンジン重みを辞書で返す
    
    Args:
        session: セッション
        
    Returns:
        dict: {"trend": float, "mean_rev": float, "breakout": float}
    """
    w = ENGINE_WEIGHTS.get(session, (0.0, 0.0, 0.0))
    return {
        "trend": w[0],
        "mean_rev": w[1],
        "breakout": w[2],
    }


def apply_session_weights(signals: Dict[str, float]) -> Dict[str, float]:
    """
    生のシグナルスコアにセッション重みを掛けて返す
    
    Args:
        signals: {"trend": float, "mean_rev": float, "breakout": float}
                 各値は -1.0 ~ +1.0
    
    Returns:
        dict: 重み付けされたシグナル + セッション情報
    """
    from session.detector import get_current_session

    session = get_current_session()
    weights = get_engine_weights(session)

    return {
        "trend":    signals.get("trend", 0.0) * weights["trend"],
        "mean_rev": signals.get("mean_rev", 0.0) * weights["mean_rev"],
        "breakout": signals.get("breakout", 0.0) * weights["breakout"],
        "session":  session.value,
    }


def is_engine_active(session: Session, engine: str) -> bool:
    """
    指定セッションでエンジンがアクティブかどうか
    
    Args:
        session: セッション
        engine: "trend", "mean_rev", "breakout"
    """
    weights = get_engine_weights(session)
    return weights.get(engine, 0.0) > 0.0
