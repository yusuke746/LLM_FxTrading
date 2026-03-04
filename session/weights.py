"""
セッション × レジーム 統合重みモジュール

旧: セッション別に7エンジンの重みを固定値で管理
新: セッション別 × レジーム別 の2次元マトリクスでゲーティング

設計思想:
  1. エンジンを3カテゴリに分類
     - 順張り (trend_follow): Trend, Breakout, SessionORB
     - 逆張り (mean_revert):  MeanReversion, MomentumDivergence
     - 構造   (structural):   SupplyDemand, MarketStructure

  2. レジームがカテゴリのON/OFFを決める（regime.pyのgates）
  3. セッションがカテゴリ内の個別エンジン重みを決める

  → 最終重み = session_weight[engine] × regime_gate[category]

  これにより:
  - トレンド中にRSI逆張りが相殺してゼロシグナルになる問題を解消
  - レンジ中にブレイクアウトのダマシを排除
  - 構造系は常に参考として聞く（離散的に発火するため）
"""

from typing import Dict, Tuple

from session.detector import Session


# ========================================
# エンジン → カテゴリ マッピング
# ========================================

ENGINE_CATEGORIES: Dict[str, str] = {
    "trend":            "trend_follow",
    "breakout":         "trend_follow",
    "session_orb":      "trend_follow",
    "mean_rev":         "mean_revert",
    "momentum_div":     "mean_revert",
    "supply_demand":    "structural",
    "market_structure": "structural",
}

ENGINE_KEYS = [
    "trend", "mean_rev", "breakout",
    "momentum_div", "supply_demand",
    "session_orb", "market_structure",
]


# ========================================
# セッション別エンジン重み（レジームゲーティング前）
# ========================================
# カテゴリ内のエンジン間の相対的重要度を決める
# レジームゲートが掛かった後のカテゴリ合計で方向判定

SESSION_WEIGHTS: Dict[Session, Dict[str, float]] = {
    Session.ASIA: {
        "trend":            0.4,  # アジアのトレンドは弱い
        "breakout":         0.3,  # ブレイクアウトは少ない
        "session_orb":      0.0,  # アジアORBは不要
        "mean_rev":         0.8,  # アジアはレンジ逆張りの稼ぎ場
        "momentum_div":     0.6,  # ダイバージェンスは有効
        "supply_demand":    0.7,  # S/Dゾーンはアジアで機能しやすい
        "market_structure": 0.5,  # 構造は参考
    },
    Session.LONDON_PREP: {
        "trend": 0.0, "breakout": 0.0, "session_orb": 0.0,
        "mean_rev": 0.0, "momentum_div": 0.0,
        "supply_demand": 0.0, "market_structure": 0.0,
    },
    Session.LONDON: {
        "trend":            1.0,  # ロンドンはトレンドの要
        "breakout":         0.9,  # セッションブレイク多発
        "session_orb":      1.0,  # ロンドンORBは最重要
        "mean_rev":         0.3,  # 逆張りは限定的
        "momentum_div":     0.7,  # ダイバージェンスは転換の先行指標
        "supply_demand":    0.5,  # S/Dは控えめ
        "market_structure": 0.8,  # BoS/CHoCHはロンドンで頻出
    },
    Session.OVERLAP: {
        "trend":            1.0,  # 最もトレンドが出る時間帯
        "breakout":         1.2,  # ブレイクアウト最優先（ボラ最大）
        "session_orb":      0.8,  # NY ORBチェック
        "mean_rev":         0.0,  # 逆張り禁止（ボラ大で逆張りは致命傷）
        "momentum_div":     0.6,  # ダイバージェンスは補助
        "supply_demand":    0.5,  # S/Dは参考
        "market_structure": 0.8,  # 構造転換は重要
    },
    Session.NY_LATE: {
        "trend":            0.6,  # トレンド弱化
        "breakout":         0.4,  # ブレイクは減少
        "session_orb":      0.0,  # ORB不要
        "mean_rev":         0.7,  # NY後半は巻き戻しが多い
        "momentum_div":     0.6,  # ダイバージェンスで転換検出
        "supply_demand":    0.7,  # S/Dゾーンは有効（戻りを拾う）
        "market_structure": 0.5,  # 構造は参考
    },
    Session.CLOSED: {
        "trend": 0.0, "breakout": 0.0, "session_orb": 0.0,
        "mean_rev": 0.0, "momentum_div": 0.0,
        "supply_demand": 0.0, "market_structure": 0.0,
    },
}

# 後方互換: 旧タプル形式 ENGINE_WEIGHTS
ENGINE_WEIGHTS: Dict[Session, Tuple[float, ...]] = {
    session: tuple(w_dict.get(k, 0.0) for k in ENGINE_KEYS)
    for session, w_dict in SESSION_WEIGHTS.items()
}


def get_session_weights(session: Session) -> Dict[str, float]:
    """セッション別の生エンジン重みを取得"""
    return SESSION_WEIGHTS.get(session, {k: 0.0 for k in ENGINE_KEYS})


def get_engine_weights(session: Session) -> Dict[str, float]:
    """後方互換: get_session_weights のエイリアス"""
    return get_session_weights(session)


def apply_regime_gated_weights(
    raw_signals: Dict[str, float],
    session: Session,
    regime_gates: Dict[str, float],
) -> Dict[str, float]:
    """
    レジームゲーティング付きの重み適用

    Args:
        raw_signals: 各エンジンの生シグナル (-1.0 ~ +1.0)
        session: 現在のセッション
        regime_gates: レジームのカテゴリ別ゲート値
            {"trend_follow": 0-1, "mean_revert": 0-1, "structural": 0-1}

    Returns:
        dict: 最終重み付けされたシグナル
    """
    session_w = get_session_weights(session)

    result = {}
    for engine in ENGINE_KEYS:
        category = ENGINE_CATEGORIES[engine]
        gate = regime_gates.get(category, 0.5)
        weight = session_w.get(engine, 0.0)
        raw_signal = raw_signals.get(engine, 0.0)

        # 最終重み = セッション重み × レジームゲート
        result[engine] = raw_signal * weight * gate

    return result


def get_category_scores(weighted_signals: Dict[str, float]) -> Dict[str, float]:
    """
    カテゴリ別のスコアを算出（コンフルエンス判定用）

    Returns:
        dict: {"trend_follow": float, "mean_revert": float, "structural": float}
    """
    categories = {"trend_follow": 0.0, "mean_revert": 0.0, "structural": 0.0}
    for engine, value in weighted_signals.items():
        if engine in ENGINE_CATEGORIES:
            cat = ENGINE_CATEGORIES[engine]
            categories[cat] += value
    return categories


def apply_session_weights(signals: Dict[str, float]) -> Dict[str, float]:
    """
    後方互換: 生のシグナルスコアにセッション重みを掛けて返す
    （レジームゲーティングなし）
    """
    from session.detector import get_current_session
    session = get_current_session()
    weights = get_session_weights(session)
    result = {key: signals.get(key, 0.0) * weights.get(key, 0.0) for key in ENGINE_KEYS}
    result["session"] = session.value
    return result


def is_engine_active(session: Session, engine: str) -> bool:
    """指定セッションでエンジンがアクティブか"""
    weights = get_session_weights(session)
    return weights.get(engine, 0.0) > 0.0
