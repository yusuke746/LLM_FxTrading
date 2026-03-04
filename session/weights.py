"""
セッション別エンジン重みモジュール
各セッションにおける7エンジンの重み付けを管理

エンジン: trend / mean_rev / breakout / momentum_div / supply_demand / session_orb / market_structure

★の数がシグナルスコアへの寄与度:
  アジア:           逆張り★★★, S/D★★, モメンタム★★, ストラクチャー★, トレンド★, ブレイク★, ORB❌
  ロンドン直前:     全停止
  ロンドン:         トレンド★★★, ブレイク★★★, ORB★★★, ストラクチャー★★, モメンタム★★, S/D★, 逆張り★
  ロンドン×NY重複:  ブレイク★★★★, トレンド★★★, ORB★★★, ストラクチャー★★, モメンタム★★, S/D★, 逆張り❌
  NY後半:           トレンド★★, 逆張り★★, S/D★★, モメンタム★★, ストラクチャー★, ブレイク★, ORB❌
"""

from typing import Dict, Tuple

from session.detector import Session


# セッション別エンジン重み:
# (trend, mean_rev, breakout, momentum_div, supply_demand, session_orb, market_structure)
ENGINE_WEIGHTS: Dict[Session, Tuple[float, ...]] = {
    #                          trend  mean_rev  breakout  mom_div  s/d   orb   mkt_str
    Session.ASIA:             (0.5,   1.0,      0.3,      0.7,    0.8,  0.0,  0.5),
    Session.LONDON_PREP:      (0.0,   0.0,      0.0,      0.0,    0.0,  0.0,  0.0),     # 全停止
    Session.LONDON:           (1.0,   0.3,      1.0,      0.8,    0.5,  1.0,  0.8),
    Session.OVERLAP:          (1.0,   0.0,      1.5,      0.8,    0.5,  1.0,  0.8),     # breakout最優先
    Session.NY_LATE:          (0.8,   0.7,      0.5,      0.7,    0.8,  0.0,  0.5),
    Session.CLOSED:           (0.0,   0.0,      0.0,      0.0,    0.0,  0.0,  0.0),     # 週末停止
}

ENGINE_KEYS = ["trend", "mean_rev", "breakout", "momentum_div", "supply_demand", "session_orb", "market_structure"]


def get_engine_weights(session: Session) -> Dict[str, float]:
    """
    指定セッションのエンジン重みを辞書で返す
    
    Args:
        session: セッション
        
    Returns:
        dict: 各エンジンの重み
    """
    w = ENGINE_WEIGHTS.get(session, (0.0,) * len(ENGINE_KEYS))
    return {key: w[i] for i, key in enumerate(ENGINE_KEYS)}


def apply_session_weights(signals: Dict[str, float]) -> Dict[str, float]:
    """
    生のシグナルスコアにセッション重みを掛けて返す
    
    Args:
        signals: 各エンジンのスコア（-1.0 ~ +1.0）
    
    Returns:
        dict: 重み付けされたシグナル + セッション情報
    """
    from session.detector import get_current_session

    session = get_current_session()
    weights = get_engine_weights(session)

    result = {
        key: signals.get(key, 0.0) * weights[key]
        for key in ENGINE_KEYS
    }
    result["session"] = session.value
    return result


def is_engine_active(session: Session, engine: str) -> bool:
    """
    指定セッションでエンジンがアクティブかどうか
    
    Args:
        session: セッション
        engine: エンジン名
    """
    weights = get_engine_weights(session)
    return weights.get(engine, 0.0) > 0.0
