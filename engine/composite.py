"""
プロ仕様 合成シグナルモジュール (v2)

旧版: 7エンジン × セッション重み → 単純合算 → 閾値比較
新版: レジーム検出 → カテゴリ分類 → レジームゲーティング
      → コンフルエンス判定（2+カテゴリ合意必須）→ 最終スコア

アーキテクチャ:
  1. 7エンジンから生シグナル取得
  2. RegimeDetector で現在のレジーム判定 (TRENDING/RANGING/VOLATILE/QUIET)
  3. セッション重み × レジームゲート = 最終重み
  4. 3カテゴリ (trend_follow / mean_revert / structural) のスコア算出
  5. コンフルエンス: 2カテゴリ以上が同方向 > 閾値で初めてシグナル発火
  6. 閾値はレジームで動的調整（VOLATILE→引き上げ）

これにより:
  - トレンド中の逆張りがシグナルを相殺する問題を解消
  - レンジ中のブレイクアウトダマシを排除
  - 単一エンジンの暴走を防止（カテゴリ分散が必須）
"""

import pandas as pd
from typing import Dict, Optional

from config_manager import get
from engine.trend import TrendEngine
from engine.mean_reversion import MeanReversionEngine
from engine.breakout import BreakoutEngine
from engine.momentum_divergence import MomentumDivergenceEngine
from engine.supply_demand import SupplyDemandEngine
from engine.session_orb import SessionORBEngine
from engine.market_structure import MarketStructureEngine
from engine.regime import RegimeDetector, Regime
from session.detector import get_current_session
from session.weights import (
    ENGINE_KEYS, ENGINE_CATEGORIES,
    apply_regime_gated_weights, get_category_scores, get_session_weights,
)
from logger_setup import get_logger

logger = get_logger("engine.composite")

# === コンフルエンス設定 ===
# 最低何カテゴリが同方向に揃う必要があるか
MIN_CONFLUENCE_CATEGORIES = 2
# カテゴリスコアが「有効」とみなされる最低絶対値
CATEGORY_ACTIVATION_THRESHOLD = 0.15

# === エンジンのシングルトンインスタンス ===
_trend_engine: Optional[TrendEngine] = None
_mean_rev_engine: Optional[MeanReversionEngine] = None
_breakout_engine: Optional[BreakoutEngine] = None
_momentum_div_engine: Optional[MomentumDivergenceEngine] = None
_supply_demand_engine: Optional[SupplyDemandEngine] = None
_session_orb_engine: Optional[SessionORBEngine] = None
_market_structure_engine: Optional[MarketStructureEngine] = None
_regime_detector: Optional[RegimeDetector] = None


def _get_engines():
    """エンジンインスタンスを取得（遅延初期化）"""
    global _trend_engine, _mean_rev_engine, _breakout_engine
    global _momentum_div_engine, _supply_demand_engine
    global _session_orb_engine, _market_structure_engine
    global _regime_detector

    if _trend_engine is None:
        _trend_engine = TrendEngine()
    if _mean_rev_engine is None:
        _mean_rev_engine = MeanReversionEngine()
    if _breakout_engine is None:
        _breakout_engine = BreakoutEngine()
    if _momentum_div_engine is None:
        _momentum_div_engine = MomentumDivergenceEngine()
    if _supply_demand_engine is None:
        _supply_demand_engine = SupplyDemandEngine()
    if _session_orb_engine is None:
        _session_orb_engine = SessionORBEngine()
    if _market_structure_engine is None:
        _market_structure_engine = MarketStructureEngine()
    if _regime_detector is None:
        _regime_detector = RegimeDetector()

    return (
        _trend_engine, _mean_rev_engine, _breakout_engine,
        _momentum_div_engine, _supply_demand_engine,
        _session_orb_engine, _market_structure_engine,
        _regime_detector,
    )


def _check_confluence(category_scores: Dict[str, float]) -> Dict:
    """
    コンフルエンス判定: 独立した2カテゴリ以上が同方向に揃っているか

    Args:
        category_scores: {"trend_follow": float, "mean_revert": float, "structural": float}

    Returns:
        dict: {
            "passed": bool,
            "direction": "BUY" / "SELL" / "NONE",
            "agreeing_categories": list,
            "details": str,
        }
    """
    bullish = []
    bearish = []

    for cat, score in category_scores.items():
        if score > CATEGORY_ACTIVATION_THRESHOLD:
            bullish.append(cat)
        elif score < -CATEGORY_ACTIVATION_THRESHOLD:
            bearish.append(cat)

    if len(bullish) >= MIN_CONFLUENCE_CATEGORIES:
        return {
            "passed": True,
            "direction": "BUY",
            "agreeing_categories": bullish,
            "details": f"BUYコンフルエンス: {', '.join(bullish)} ({len(bullish)}カテゴリ合意)",
        }
    elif len(bearish) >= MIN_CONFLUENCE_CATEGORIES:
        return {
            "passed": True,
            "direction": "SELL",
            "agreeing_categories": bearish,
            "details": f"SELLコンフルエンス: {', '.join(bearish)} ({len(bearish)}カテゴリ合意)",
        }
    else:
        return {
            "passed": False,
            "direction": "NONE",
            "agreeing_categories": [],
            "details": f"コンフルエンス不足 (BUY:{len(bullish)} SELL:{len(bearish)} < {MIN_CONFLUENCE_CATEGORIES})",
        }


def calc_composite_signal(
    df: pd.DataFrame,
    h4_df: Optional[pd.DataFrame] = None,
    entry_threshold: Optional[float] = None,
) -> Dict:
    """
    全エンジンのシグナルをレジーム・コンフルエンス付きで合成してスコア化

    Args:
        df: H1足 OHLCVデータ
        h4_df: H4足データ（マルチタイムフレーム確認用）
        entry_threshold: エントリー閾値（Noneの場合はconfig.yamlから取得）

    Returns:
        dict: {
            "direction": "BUY" / "SELL" / "NONE",
            "score": 0.0 ~ 1.0+,
            "composite_raw": float,
            "raw_signals": {各エンジンの生スコア},
            "weighted_signals": {重み付け後のスコア},
            "session": セッション名,
            "regime": レジーム名,
            "regime_confidence": float,
            "category_scores": {カテゴリ別スコア},
            "confluence": {コンフルエンス判定結果},
        }
    """
    if entry_threshold is None:
        entry_threshold = get("session.entry_threshold", 0.6)

    (
        trend_engine, mean_rev_engine, breakout_engine,
        momentum_div_engine, supply_demand_engine,
        session_orb_engine, market_structure_engine,
        regime_detector,
    ) = _get_engines()

    # === Step 1: 各エンジンからシグナル取得 (-1.0 ~ +1.0) ===
    raw_signals = {
        "trend": trend_engine.get_signal(df, h4_df),
        "mean_rev": mean_rev_engine.get_signal(df),
        "breakout": breakout_engine.get_signal(df),
        "momentum_div": momentum_div_engine.get_signal(df),
        "supply_demand": supply_demand_engine.get_signal(df),
        "session_orb": session_orb_engine.get_signal(df),
        "market_structure": market_structure_engine.get_signal(df),
    }

    # === Step 2: レジーム検出 ===
    regime_result = regime_detector.detect(df)
    regime = regime_result["regime"]
    regime_gates = regime_result["gates"]
    regime_confidence = regime_result["confidence"]
    threshold_adj = regime_result.get("details", {}).get("threshold_adjustment", 0.0)

    # レジームに応じて閾値を動的調整
    effective_threshold = entry_threshold + threshold_adj

    # === Step 3: セッション重み × レジームゲート ===
    session = get_current_session()
    weighted = apply_regime_gated_weights(raw_signals, session, regime_gates)

    # === Step 4: カテゴリ別スコア算出 ===
    category_scores = get_category_scores(weighted)

    # === Step 5: コンフルエンス判定 ===
    confluence = _check_confluence(category_scores)

    # === Step 6: 最終スコア計算 ===
    # コンフルエンス通過時のみスコアを算出
    if confluence["passed"]:
        # 合意したカテゴリのスコアのみ合算（反対方向のノイズは除外）
        direction = confluence["direction"]
        composite = sum(weighted[k] for k in ENGINE_KEYS)

        # 方向一致チェック
        if direction == "BUY" and composite > 0:
            score = composite
        elif direction == "SELL" and composite < 0:
            score = abs(composite)
        else:
            # カテゴリは合意しているが合算が反対方向 → NONE
            direction = "NONE"
            score = 0.0
            composite = 0.0

        # 閾値チェック
        if score < effective_threshold:
            direction = "NONE"
            score = 0.0
    else:
        direction = "NONE"
        score = 0.0
        composite = sum(weighted[k] for k in ENGINE_KEYS)

    result = {
        "direction": direction,
        "score": round(score, 4),
        "composite_raw": round(composite, 4),
        "raw_signals": {k: round(v, 4) for k, v in raw_signals.items()},
        "weighted_signals": {k: round(v, 4) for k, v in weighted.items()},
        "session": session.value,
        "regime": regime.value,
        "regime_confidence": regime_confidence,
        "regime_gates": {k: round(v, 2) for k, v in regime_gates.items()},
        "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
        "confluence": confluence,
        "effective_threshold": round(effective_threshold, 4),
    }

    if direction != "NONE":
        logger.info(
            f"シグナル検出: {direction} | スコア={score:.4f} | "
            f"レジーム={regime.value} (確信度{regime_confidence:.0%}) | "
            f"セッション={session.value} | "
            f"コンフルエンス={confluence['details']} | "
            f"カテゴリ: TF={category_scores['trend_follow']:.3f} "
            f"MR={category_scores['mean_revert']:.3f} "
            f"ST={category_scores['structural']:.3f} | "
            f"T={raw_signals['trend']:.3f} M={raw_signals['mean_rev']:.3f} "
            f"B={raw_signals['breakout']:.3f} "
            f"MD={raw_signals['momentum_div']:.3f} SD={raw_signals['supply_demand']:.3f} "
            f"ORB={raw_signals['session_orb']:.3f} MS={raw_signals['market_structure']:.3f}"
        )
    else:
        # NONEでもデバッグログは出す
        logger.debug(
            f"シグナルなし | レジーム={regime.value} | "
            f"コンフルエンス={confluence['details']} | "
            f"composite_raw={composite:.4f} vs threshold={effective_threshold:.4f} | "
            f"カテゴリ: TF={category_scores['trend_follow']:.3f} "
            f"MR={category_scores['mean_revert']:.3f} "
            f"ST={category_scores['structural']:.3f}"
        )

    return result


def calc_composite_signal_from_raw(
    raw_signals: Dict[str, float],
    regime_gates: Dict[str, float],
    session_name: str,
    entry_threshold: float = 0.6,
    threshold_adj: float = 0.0,
) -> Dict:
    """
    バックテスト用: 事前計算済みの生シグナルとレジームから合成シグナルを算出

    エンジンのインスタンス化不要。高速。

    Args:
        raw_signals: 各エンジンの生シグナル
        regime_gates: レジームゲート値
        session_name: セッション名（文字列）
        entry_threshold: エントリー閾値
        threshold_adj: レジームによる閾値調整

    Returns:
        dict: direction, score, composite_raw, ...
    """
    from session.detector import Session

    # セッションを文字列からEnum変換
    try:
        session = Session(session_name)
    except (ValueError, KeyError):
        session = Session.CLOSED

    # 重みゲーティング
    weighted = apply_regime_gated_weights(raw_signals, session, regime_gates)

    # カテゴリスコア
    category_scores = get_category_scores(weighted)

    # コンフルエンス判定
    confluence = _check_confluence(category_scores)

    # 最終スコア
    effective_threshold = entry_threshold + threshold_adj

    if confluence["passed"]:
        direction = confluence["direction"]
        composite = sum(weighted[k] for k in ENGINE_KEYS)

        if direction == "BUY" and composite > 0:
            score = composite
        elif direction == "SELL" and composite < 0:
            score = abs(composite)
        else:
            direction = "NONE"
            score = 0.0
            composite = 0.0

        if score < effective_threshold:
            direction = "NONE"
            score = 0.0
    else:
        direction = "NONE"
        score = 0.0
        composite = sum(weighted[k] for k in ENGINE_KEYS)

    return {
        "direction": direction,
        "score": round(score, 4),
        "composite_raw": round(composite, 4),
    }


def reset_engines():
    """エンジンインスタンスをリセット（テスト用）"""
    global _trend_engine, _mean_rev_engine, _breakout_engine
    global _momentum_div_engine, _supply_demand_engine
    global _session_orb_engine, _market_structure_engine
    global _regime_detector
    _trend_engine = None
    _mean_rev_engine = None
    _breakout_engine = None
    _momentum_div_engine = None
    _supply_demand_engine = None
    _session_orb_engine = None
    _market_structure_engine = None
    _regime_detector = None
