"""
市場レジーム検出エンジン（定量ベース・LLM不使用）

ADX + BBバンド幅 + ATRパーセンタイルから4レジームを判定:
  TRENDING:  ADX≥25 かつ BB幅>中央値 → トレンドフォロー系を優先
  RANGING:   ADX<20 かつ BB幅<中央値 → 逆張り系を優先
  VOLATILE:  ADX任意 かつ ATR>75パーセンタイル → 慎重 (閾値UP)
  QUIET:     ATR<25パーセンタイル → 見送り推奨

レジームの意義:
  トレンド系エンジン (Trend/Breakout/ORB) と逆張り系 (MeanRev/MomDiv) は
  本来「同じ状況で逆のことを言う」。両方を常時ONにすると相殺して
  中途半場なシグナルになる。レジームでゲーティングすることで
  「今の相場に合ったエンジンだけを聴く」プロの判断を実装する。
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from enum import Enum
from typing import Dict, Optional

from logger_setup import get_logger

logger = get_logger("engine.regime")

# パラメータ
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25      # これ以上 → トレンド
ADX_RANGE_THRESHOLD = 20      # これ以下 → レンジ
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
ATR_LOOKBACK = 100             # パーセンタイル算出用
ATR_HIGH_PERCENTILE = 75       # 高ボラ判定
ATR_LOW_PERCENTILE = 25        # 低ボラ判定


class Regime(Enum):
    """市場レジーム"""
    TRENDING = "trending"       # 強いトレンド
    RANGING = "ranging"         # レンジ・横ばい
    VOLATILE = "volatile"       # 高ボラティリティ
    QUIET = "quiet"             # 低ボラティリティ


# レジーム別: 各カテゴリのゲート倍率
# カテゴリ: trend_follow(順張り), mean_revert(逆張り), structural(構造)
REGIME_GATES: Dict[Regime, Dict[str, float]] = {
    Regime.TRENDING: {
        "trend_follow": 1.0,    # 順張り全開
        "mean_revert": 0.0,     # 逆張り完全OFF (トレンド中の逆張りは自殺行為)
        "structural": 0.8,      # 構造は補助的に
    },
    Regime.RANGING: {
        "trend_follow": 0.2,    # トレンド系はほぼ不要（ダマシだらけ）
        "mean_revert": 1.0,     # 逆張り全開
        "structural": 0.7,      # S/Dは有効
    },
    Regime.VOLATILE: {
        "trend_follow": 0.6,    # トレンドあるかもだが慎重に
        "mean_revert": 0.3,     # 逆張りは危険（V字でやられる）
        "structural": 0.5,      # 構造は半信半疑
    },
    Regime.QUIET: {
        "trend_follow": 0.3,    # 動かない相場でブレイク待ち
        "mean_revert": 0.5,     # 小幅レンジ逆張りは可
        "structural": 0.4,      # 構造は弱い
    },
}


class RegimeDetector:
    """定量的市場レジーム検出器"""

    def __init__(self):
        self._last_regime: Optional[Regime] = None
        self._regime_bars: int = 0  # 同一レジーム継続バー数

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        現在の市場レジームを検出

        Args:
            df: H1足 OHLCVデータ（最低100本以上）

        Returns:
            dict: {
                "regime": Regime,
                "confidence": float (0-1),
                "gates": {"trend_follow": float, "mean_revert": float, "structural": float},
                "details": {...},
            }
        """
        min_bars = max(ATR_LOOKBACK, BB_PERIOD, ADX_PERIOD) + 10
        if len(df) < min_bars:
            return self._default_result()

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # === ADX: トレンド強度 ===
            adx_df = ta.adx(high, low, close, length=ADX_PERIOD)
            current_adx = adx_df[f"ADX_{ADX_PERIOD}"].iloc[-1]
            if pd.isna(current_adx):
                return self._default_result()

            # === BBバンド幅: レンジ/トレンドの幅感 ===
            bb = ta.bbands(close, length=BB_PERIOD, std=BB_STD)
            bb_upper_col = [c for c in bb.columns if c.startswith("BBU_")][0]
            bb_lower_col = [c for c in bb.columns if c.startswith("BBL_")][0]
            bb_width = (bb[bb_upper_col] - bb[bb_lower_col]) / close
            bb_width = bb_width.dropna()
            current_bb_width = bb_width.iloc[-1]
            bb_width_median = bb_width.tail(ATR_LOOKBACK).median()

            # === ATRパーセンタイル: ボラティリティの相対位置 ===
            atr_series = ta.atr(high, low, close, length=ATR_PERIOD)
            atr_clean = atr_series.dropna().tail(ATR_LOOKBACK)
            current_atr = atr_series.iloc[-1]
            if pd.isna(current_atr) or len(atr_clean) < 20:
                return self._default_result()

            atr_pct = float((atr_clean < current_atr).sum() / len(atr_clean) * 100)

            # === レジーム判定ロジック ===
            regime, confidence = self._classify(
                current_adx, current_bb_width, bb_width_median, atr_pct
            )

            # レジーム継続性チェック（頻繁な切り替えを防止）
            if regime == self._last_regime:
                self._regime_bars += 1
            else:
                # 新レジームに切替わる前に最低3バーの確認を要求
                if self._regime_bars < 3 and self._last_regime is not None:
                    regime = self._last_regime
                    confidence *= 0.8  # 信頼度を下げる
                else:
                    self._last_regime = regime
                    self._regime_bars = 0

            gates = REGIME_GATES[regime].copy()

            # ボラタイル時は閾値引き上げの推奨
            threshold_adjustment = 0.0
            if regime == Regime.VOLATILE:
                threshold_adjustment = 0.1   # 閾値+0.1
            elif regime == Regime.QUIET:
                threshold_adjustment = 0.05  # 閾値+0.05

            details = {
                "adx": round(current_adx, 2),
                "bb_width": round(current_bb_width, 5),
                "bb_width_median": round(bb_width_median, 5),
                "atr_percentile": round(atr_pct, 1),
                "regime_bars": self._regime_bars,
                "threshold_adjustment": threshold_adjustment,
            }

            logger.info(
                f"レジーム: {regime.value} (信頼度{confidence:.0%}) | "
                f"ADX={current_adx:.1f} BB幅={current_bb_width:.5f} ATR%ile={atr_pct:.0f}"
            )

            return {
                "regime": regime,
                "confidence": round(confidence, 3),
                "gates": gates,
                "details": details,
            }

        except Exception as e:
            logger.error(f"レジーム検出エラー: {e}", exc_info=True)
            return self._default_result()

    def _classify(
        self,
        adx: float,
        bb_width: float,
        bb_median: float,
        atr_pct: float,
    ) -> tuple:
        """
        指標値からレジームを分類

        Returns:
            (Regime, confidence)
        """
        # スコアリング方式: 各レジームのスコアを算出し最高スコアを選出
        scores = {
            Regime.TRENDING: 0.0,
            Regime.RANGING: 0.0,
            Regime.VOLATILE: 0.0,
            Regime.QUIET: 0.0,
        }

        # --- ADX でトレンド/レンジを判定 ---
        if adx >= ADX_TREND_THRESHOLD:
            # ADX高い → トレンドの可能性
            trend_strength = min((adx - ADX_TREND_THRESHOLD) / 15, 1.0)
            scores[Regime.TRENDING] += 0.5 + 0.3 * trend_strength
        elif adx <= ADX_RANGE_THRESHOLD:
            # ADX低い → レンジの可能性
            range_strength = min((ADX_RANGE_THRESHOLD - adx) / 10, 1.0)
            scores[Regime.RANGING] += 0.5 + 0.3 * range_strength
        else:
            # 中間 → 弱めの判定
            scores[Regime.TRENDING] += 0.2
            scores[Regime.RANGING] += 0.2

        # --- BB幅 でトレンド/レンジを補完 ---
        if bb_median > 0:
            bb_ratio = bb_width / bb_median
            if bb_ratio > 1.3:
                scores[Regime.TRENDING] += 0.2
                scores[Regime.VOLATILE] += 0.15
            elif bb_ratio < 0.7:
                scores[Regime.RANGING] += 0.2
                scores[Regime.QUIET] += 0.15

        # --- ATRパーセンタイル でボラティリティ判定 ---
        if atr_pct >= ATR_HIGH_PERCENTILE:
            scores[Regime.VOLATILE] += 0.4
            # 高ボラ+トレンドは「強トレンド」
            if adx >= ADX_TREND_THRESHOLD:
                scores[Regime.TRENDING] += 0.15
        elif atr_pct <= ATR_LOW_PERCENTILE:
            scores[Regime.QUIET] += 0.4
            # 低ボラ+レンジは「デッドマーケット」
            if adx <= ADX_RANGE_THRESHOLD:
                scores[Regime.QUIET] += 0.15

        # 最高スコアのレジームを選出
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]

        # 信頼度: 最高と2番目の差が大きいほど確信度が高い
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
        confidence = min(0.5 + margin, 1.0)

        return best_regime, confidence

    def _default_result(self) -> Dict:
        """データ不足時のデフォルト"""
        return {
            "regime": Regime.RANGING,
            "confidence": 0.3,
            "gates": REGIME_GATES[Regime.RANGING].copy(),
            "details": {"adx": 0, "bb_width": 0, "atr_percentile": 50},
        }

    @staticmethod
    def detect_from_indicators(adx: float, bb_width: float, bb_median: float, atr_pct: float) -> Dict:
        """
        バックテスト用: 事前計算済みの指標値からレジームを判定
        （DataFrameを渡す必要なし、高速版）
        """
        detector = RegimeDetector()
        regime, confidence = detector._classify(adx, bb_width, bb_median, atr_pct)
        return {
            "regime": regime,
            "confidence": confidence,
            "gates": REGIME_GATES[regime].copy(),
        }
