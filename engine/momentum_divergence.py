"""
モメンタム・ダイバージェンスエンジン（EUR/USD H1専用チューニング）

パラメータ:
  - RSIダイバージェンス: 期間14, 検出窓20本
  - MACDヒストグラムダイバージェンス: 12/26/9
  - ストキャスティクスRSIによるモメンタム確認

シグナル:
  +1.0: 強気ダイバージェンス（価格安値更新 + RSI安値切上げ + MACD確認）
  -1.0: 弱気ダイバージェンス（価格高値更新 + RSI高値切下げ + MACD確認）
   0.0: ダイバージェンスなし

戦略本質:
  「価格は新値だがモメンタムが追従しない」→ 反転が近い
  機関投資家が注目する古典的高勝率シグナル
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from logger_setup import get_logger

logger = get_logger("engine.momentum_divergence")

# デフォルトパラメータ
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
DIVERGENCE_LOOKBACK = 20  # ダイバージェンス検出窓（本数）
SWING_TOLERANCE = 3       # スイングポイント検出の前後バー数


class MomentumDivergenceEngine:
    """RSI/MACDダイバージェンスによる反転検出エンジン"""

    def __init__(
        self,
        rsi_period: int = RSI_PERIOD,
        macd_fast: int = MACD_FAST,
        macd_slow: int = MACD_SLOW,
        macd_signal: int = MACD_SIGNAL,
        divergence_lookback: int = DIVERGENCE_LOOKBACK,
        swing_tolerance: int = SWING_TOLERANCE,
    ):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_lookback = divergence_lookback
        self.swing_tolerance = swing_tolerance

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        モメンタム・ダイバージェンスシグナルを算出

        Args:
            df: H1足 OHLCVデータ（少なくとも50本以上）

        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        min_bars = self.macd_slow + self.divergence_lookback + 10
        if len(df) < min_bars:
            logger.warning(f"データ不足: {len(df)}本（最低{min_bars}本必要）")
            return 0.0

        try:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values

            # RSI算出
            rsi_series = ta.rsi(df["close"], length=self.rsi_period)
            rsi = rsi_series.values

            # MACDヒストグラム算出
            macd_df = ta.macd(
                df["close"],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal,
            )
            macd_hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
            macd_hist = macd_df[macd_hist_col].values

            signal = 0.0
            window = self.divergence_lookback

            # === 強気ダイバージェンス検出（BUY） ===
            bullish_score = self._detect_bullish_divergence(
                close, low, rsi, macd_hist, window
            )

            # === 弱気ダイバージェンス検出（SELL） ===
            bearish_score = self._detect_bearish_divergence(
                close, high, rsi, macd_hist, window
            )

            # ストキャスティクスRSIでモメンタム確認
            stoch_rsi = ta.stochrsi(df["close"], length=self.rsi_period)
            if stoch_rsi is not None and len(stoch_rsi) > 0:
                stoch_k_col = [c for c in stoch_rsi.columns if "STOCHRSIk" in c]
                if stoch_k_col:
                    stoch_k = stoch_rsi[stoch_k_col[0]].iloc[-1]
                    if not pd.isna(stoch_k):
                        # 強気ダイバ + StochRSI低い → 更に信頼度UP
                        if bullish_score > 0 and stoch_k < 20:
                            bullish_score += 0.15
                        # 弱気ダイバ + StochRSI高い → 更に信頼度UP
                        if bearish_score > 0 and stoch_k > 80:
                            bearish_score += 0.15

            # 最終スコア
            if bullish_score > bearish_score and bullish_score > 0:
                signal = min(bullish_score, 1.0)
            elif bearish_score > bullish_score and bearish_score > 0:
                signal = -min(bearish_score, 1.0)

            return round(signal, 4)

        except Exception as e:
            logger.error(f"モメンタム・ダイバージェンス算出エラー: {e}", exc_info=True)
            return 0.0

    def _find_swing_lows(self, data: np.ndarray, tolerance: int) -> list:
        """スイング安値のインデックスを検出"""
        swings = []
        for i in range(tolerance, len(data) - tolerance):
            if all(data[i] <= data[i - j] for j in range(1, tolerance + 1)) and \
               all(data[i] <= data[i + j] for j in range(1, tolerance + 1)):
                swings.append(i)
        return swings

    def _find_swing_highs(self, data: np.ndarray, tolerance: int) -> list:
        """スイング高値のインデックスを検出"""
        swings = []
        for i in range(tolerance, len(data) - tolerance):
            if all(data[i] >= data[i - j] for j in range(1, tolerance + 1)) and \
               all(data[i] >= data[i + j] for j in range(1, tolerance + 1)):
                swings.append(i)
        return swings

    def _detect_bullish_divergence(
        self,
        close: np.ndarray,
        low: np.ndarray,
        rsi: np.ndarray,
        macd_hist: np.ndarray,
        window: int,
    ) -> float:
        """
        強気ダイバージェンス検出
        価格が安値更新 → RSIが安値切上げ
        """
        score = 0.0
        n = len(close)
        start = max(0, n - window - self.swing_tolerance)

        # 検出窓内のスイング安値を探す
        swing_lows = self._find_swing_lows(low[start:], self.swing_tolerance)
        # 実インデックスに変換
        swing_lows = [s + start for s in swing_lows]

        # 最新付近（最後5本以内）のスイング安値が必要
        recent_swings = [s for s in swing_lows if s >= n - 5 - self.swing_tolerance]
        if not recent_swings:
            # スイングポイントが見つからない場合、直近3本の最安値を使用
            if low[-1] <= min(low[-3:]):
                recent_swings = [n - 1]
            else:
                return 0.0

        older_swings = [s for s in swing_lows if s < recent_swings[0] and s >= start]
        if not older_swings:
            return 0.0

        recent_idx = recent_swings[-1]
        older_idx = older_swings[-1]

        # 価格: 安値更新（現在の方が低い）
        if low[recent_idx] < low[older_idx]:
            # RSI: 安値切り上がり（現在の方が高い） → RSIダイバージェンス
            if not pd.isna(rsi[recent_idx]) and not pd.isna(rsi[older_idx]):
                if rsi[recent_idx] > rsi[older_idx]:
                    score = 0.5  # RSIダイバージェンス確認

                    # MACD確認: ヒストグラムも切り上がっていれば強化
                    if not pd.isna(macd_hist[recent_idx]) and not pd.isna(macd_hist[older_idx]):
                        if macd_hist[recent_idx] > macd_hist[older_idx]:
                            score += 0.25  # MACDダブル確認

                    # RSIの絶対水準が低いほど信頼度UP
                    if rsi[recent_idx] < 35:
                        score += 0.15

        return score

    def _detect_bearish_divergence(
        self,
        close: np.ndarray,
        high: np.ndarray,
        rsi: np.ndarray,
        macd_hist: np.ndarray,
        window: int,
    ) -> float:
        """
        弱気ダイバージェンス検出
        価格が高値更新 → RSIが高値切下げ
        """
        score = 0.0
        n = len(close)
        start = max(0, n - window - self.swing_tolerance)

        # 検出窓内のスイング高値を探す
        swing_highs = self._find_swing_highs(high[start:], self.swing_tolerance)
        swing_highs = [s + start for s in swing_highs]

        # 最新付近のスイング高値が必要
        recent_swings = [s for s in swing_highs if s >= n - 5 - self.swing_tolerance]
        if not recent_swings:
            if high[-1] >= max(high[-3:]):
                recent_swings = [n - 1]
            else:
                return 0.0

        older_swings = [s for s in swing_highs if s < recent_swings[0] and s >= start]
        if not older_swings:
            return 0.0

        recent_idx = recent_swings[-1]
        older_idx = older_swings[-1]

        # 価格: 高値更新（現在の方が高い）
        if high[recent_idx] > high[older_idx]:
            # RSI: 高値切り下がり → RSIダイバージェンス
            if not pd.isna(rsi[recent_idx]) and not pd.isna(rsi[older_idx]):
                if rsi[recent_idx] < rsi[older_idx]:
                    score = 0.5

                    # MACD確認
                    if not pd.isna(macd_hist[recent_idx]) and not pd.isna(macd_hist[older_idx]):
                        if macd_hist[recent_idx] < macd_hist[older_idx]:
                            score += 0.25

                    # RSIの絶対水準が高いほど信頼度UP
                    if rsi[recent_idx] > 65:
                        score += 0.15

        return score

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """現在のインジケーター値を取得（ダッシュボード表示用）"""
        try:
            rsi = ta.rsi(df["close"], length=self.rsi_period)
            macd_df = ta.macd(
                df["close"],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal,
            )
            macd_hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]

            return {
                "rsi": round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None,
                "macd_hist": round(float(macd_df[macd_hist_col].iloc[-1]), 6)
                    if not pd.isna(macd_df[macd_hist_col].iloc[-1]) else None,
                "signal_score": self.get_signal(df),
            }
        except Exception:
            return {}
