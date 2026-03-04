"""
セッション管理パッケージ
アジア/ロンドン/NY/重複セッションの自動判定とエンジン重み管理
"""

from session.detector import Session, get_current_session, get_session_for_time
from session.weights import (
    ENGINE_WEIGHTS, apply_session_weights, get_engine_weights,
    ENGINE_CATEGORIES, ENGINE_KEYS, SESSION_WEIGHTS,
    apply_regime_gated_weights, get_category_scores, get_session_weights,
)

__all__ = [
    "Session",
    "get_current_session",
    "get_session_for_time",
    "ENGINE_WEIGHTS",
    "SESSION_WEIGHTS",
    "ENGINE_CATEGORIES",
    "ENGINE_KEYS",
    "apply_session_weights",
    "apply_regime_gated_weights",
    "get_engine_weights",
    "get_session_weights",
    "get_category_scores",
]
