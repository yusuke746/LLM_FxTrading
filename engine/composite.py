"""
合成シグナルモジュール
7つのエンジンのシグナルをセッション重み付きで合成し、最終シグナルを生成

エンジン構成:
  1. トレンドフォロー（EMA + ADX）
  2. 逆張り（RSI + BB）
  3. ブレイクアウト（レンジブレイク + ATR）
  4. モメンタム・ダイバージェンス（RSI/MACDダイバージェンス）
  5. サプライ/デマンド（オーダーブロック）
  6. セッションORB（開場レンジブレイク）
  7. マーケットストラクチャー（HH/HL/LL/LH + BoS/CHoCH）
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
from session.weights import apply_session_weights
from logger_setup import get_logger

logger = get_logger("engine.composite")

# エンジンのシングルトンインスタンス
_trend_engine: Optional[TrendEngine] = None
_mean_rev_engine: Optional[MeanReversionEngine] = None
_breakout_engine: Optional[BreakoutEngine] = None
_momentum_div_engine: Optional[MomentumDivergenceEngine] = None
_supply_demand_engine: Optional[SupplyDemandEngine] = None
_session_orb_engine: Optional[SessionORBEngine] = None
_market_structure_engine: Optional[MarketStructureEngine] = None


def _get_engines():
    """エンジンインスタンスを取得（遅延初期化）"""
    global _trend_engine, _mean_rev_engine, _breakout_engine
    global _momentum_div_engine, _supply_demand_engine
    global _session_orb_engine, _market_structure_engine

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

    return (
        _trend_engine, _mean_rev_engine, _breakout_engine,
        _momentum_div_engine, _supply_demand_engine,
        _session_orb_engine, _market_structure_engine,
    )


def calc_composite_signal(
    df: pd.DataFrame,
    h4_df: Optional[pd.DataFrame] = None,
    entry_threshold: Optional[float] = None,
) -> Dict:
    """
    全エンジンのシグナルを合成してスコア化
    
    Args:
        df: H1足 OHLCVデータ
        h4_df: H4足データ（マルチタイムフレーム確認用）
        entry_threshold: エントリー閾値（Noneの場合はconfig.yamlから取得）
    
    Returns:
        dict: {
            "direction": "BUY" / "SELL" / "NONE",
            "score": 0.0 ~ 1.0+,
            "raw_signals": {各エンジンの生スコア},
            "weighted_signals": {重み付け後のスコア},
            "session": セッション名,
        }
    """
    if entry_threshold is None:
        entry_threshold = get("session.entry_threshold", 0.6)

    (
        trend_engine, mean_rev_engine, breakout_engine,
        momentum_div_engine, supply_demand_engine,
        session_orb_engine, market_structure_engine,
    ) = _get_engines()

    # 各エンジンからシグナル取得 (-1.0 ~ +1.0)
    raw_signals = {
        "trend": trend_engine.get_signal(df, h4_df),
        "mean_rev": mean_rev_engine.get_signal(df),
        "breakout": breakout_engine.get_signal(df),
        "momentum_div": momentum_div_engine.get_signal(df),
        "supply_demand": supply_demand_engine.get_signal(df),
        "session_orb": session_orb_engine.get_signal(df),
        "market_structure": market_structure_engine.get_signal(df),
    }

    # セッション重み適用
    weighted = apply_session_weights(raw_signals)
    session = weighted.pop("session")

    # 合成スコア（全7エンジンの重み付き合計）
    engine_keys = [
        "trend", "mean_rev", "breakout",
        "momentum_div", "supply_demand",
        "session_orb", "market_structure",
    ]
    composite = sum(weighted[k] for k in engine_keys)

    # 方向判定
    if composite >= entry_threshold:
        direction = "BUY"
        score = composite
    elif composite <= -entry_threshold:
        direction = "SELL"
        score = abs(composite)
    else:
        direction = "NONE"
        score = 0.0

    result = {
        "direction": direction,
        "score": round(score, 4),
        "composite_raw": round(composite, 4),
        "raw_signals": {k: round(v, 4) for k, v in raw_signals.items()},
        "weighted_signals": {k: round(v, 4) for k, v in weighted.items()},
        "session": session,
    }

    if direction != "NONE":
        logger.info(
            f"シグナル検出: {direction} | スコア={score:.4f} | "
            f"セッション={session} | "
            f"T={raw_signals['trend']:.3f} M={raw_signals['mean_rev']:.3f} B={raw_signals['breakout']:.3f} "
            f"MD={raw_signals['momentum_div']:.3f} SD={raw_signals['supply_demand']:.3f} "
            f"ORB={raw_signals['session_orb']:.3f} MS={raw_signals['market_structure']:.3f}"
        )

    return result


def reset_engines():
    """エンジンインスタンスをリセット（テスト用）"""
    global _trend_engine, _mean_rev_engine, _breakout_engine
    global _momentum_div_engine, _supply_demand_engine
    global _session_orb_engine, _market_structure_engine
    _trend_engine = None
    _mean_rev_engine = None
    _breakout_engine = None
    _momentum_div_engine = None
    _supply_demand_engine = None
    _session_orb_engine = None
    _market_structure_engine = None
