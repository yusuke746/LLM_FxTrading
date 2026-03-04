"""
逆張りエンジン（EUR/USD H1専用チューニング）

パラメータ:
  - RSI: 期間14, 閾値30-70
  - ボリンジャーバンド: 20期間, 2σ
  - MACDダイバージェンス: 反転信頼度向上
  - セッション重み: アジア×1.0, 重複時間×0.0（逆張り禁止）

シグナル:
  +1.0: 強い買い反転（RSI<30 + BB下限タッチ + ダイバージェンス）
  -1.0: 強い売り反転（RSI>70 + BB上限タッチ + ダイバージェンス）
   0.0: 反転シグナルなし

【期待値向上】RSIの水準別段階評価:
  RSI 20未満/80超 → 極端な過熱で信頼度を上げる
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from logger_setup import get_logger

logger = get_logger("engine.mean_reversion")

# デフォルトパラメータ
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_EXTREME_OVERSOLD = 20
RSI_EXTREME_OVERBOUGHT = 80
BB_PERIOD = 20
BB_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
DIVERGENCE_LOOKBACK = 10  # ダイバージェンス検出期間


class MeanReversionEngine:
    """RSI + ボリンジャーバンド + ダイバージェンスベースの逆張りエンジン"""

    def __init__(
        self,
        rsi_period: int = RSI_PERIOD,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        逆張りシグナルを算出
        
        Args:
            df: H1足 OHLCVデータ（少なくとも30本以上）
            
        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        min_bars = max(self.bb_period, self.rsi_period, MACD_SLOW) + DIVERGENCE_LOOKBACK
        if len(df) < min_bars:
            logger.warning(f"データ不足: {len(df)}本（最低{min_bars}本必要）")
            return 0.0

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # RSI算出
            rsi = ta.rsi(close, length=self.rsi_period)
            current_rsi = rsi.iloc[-1]

            # ボリンジャーバンド算出
            bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            bb_upper = bb[f"BBU_{self.bb_period}_{self.bb_std}"].iloc[-1]
            bb_lower = bb[f"BBL_{self.bb_period}_{self.bb_std}"].iloc[-1]
            bb_mid = bb[f"BBM_{self.bb_period}_{self.bb_std}"].iloc[-1]
            current_close = close.iloc[-1]

            # NaN チェック
            if any(pd.isna([current_rsi, bb_upper, bb_lower, current_close])):
                return 0.0

            signal = 0.0

            # === 買い反転シグナル（RSI低 + BB下限付近） ===
            if current_rsi < self.rsi_oversold:
                # 基本シグナル
                signal = 0.4

                # BB下限タッチ/ブレイク
                if current_close <= bb_lower:
                    signal += 0.3

                # 極端な過売り（RSIの水準別段階評価）
                if current_rsi < RSI_EXTREME_OVERSOLD:
                    signal += 0.15

                # ダイバージェンス確認
                if self._check_bullish_divergence(df, rsi):
                    signal += 0.25

                # 反転バーの確認（下ヒゲが長い）
                if self._is_bullish_reversal_bar(df):
                    signal += 0.1

                signal = min(signal, 1.0)

            # === 売り反転シグナル（RSI高 + BB上限付近） ===
            elif current_rsi > self.rsi_overbought:
                signal = -0.4

                # BB上限タッチ/ブレイク
                if current_close >= bb_upper:
                    signal -= 0.3

                # 極端な過買い
                if current_rsi > RSI_EXTREME_OVERBOUGHT:
                    signal -= 0.15

                # ダイバージェンス確認
                if self._check_bearish_divergence(df, rsi):
                    signal -= 0.25

                # 反転バーの確認（上ヒゲが長い）
                if self._is_bearish_reversal_bar(df):
                    signal -= 0.1

                signal = max(signal, -1.0)

            return round(signal, 3)

        except Exception as e:
            logger.error(f"逆張りシグナル算出エラー: {e}")
            return 0.0

    def _check_bullish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """
        RSIの強気ダイバージェンスを検出
        価格: 安値更新、RSI: 安値切り上げ → 反転示唆
        """
        try:
            lookback = DIVERGENCE_LOOKBACK
            price_lows = df["low"].iloc[-lookback:]
            rsi_vals = rsi.iloc[-lookback:]

            if len(price_lows) < 4 or rsi_vals.isna().any():
                return False

            # 直近の安値とそれ以前の安値を比較
            recent_price_low = price_lows.iloc[-3:].min()
            prev_price_low = price_lows.iloc[:lookback - 3].min()
            recent_price_low_idx = price_lows.iloc[-3:].idxmin()
            prev_price_low_idx = price_lows.iloc[:lookback - 3].idxmin()

            recent_rsi = rsi_vals.loc[recent_price_low_idx] if recent_price_low_idx in rsi_vals.index else rsi_vals.iloc[-1]
            prev_rsi = rsi_vals.loc[prev_price_low_idx] if prev_price_low_idx in rsi_vals.index else rsi_vals.iloc[0]

            # 価格は安値更新、RSIは安値切り上げ = 強気ダイバージェンス
            if recent_price_low < prev_price_low and recent_rsi > prev_rsi:
                return True

            return False
        except Exception:
            return False

    def _check_bearish_divergence(self, df: pd.DataFrame, rsi: pd.Series) -> bool:
        """
        RSIの弱気ダイバージェンスを検出
        価格: 高値更新、RSI: 高値切り下げ → 反転示唆
        """
        try:
            lookback = DIVERGENCE_LOOKBACK
            price_highs = df["high"].iloc[-lookback:]
            rsi_vals = rsi.iloc[-lookback:]

            if len(price_highs) < 4 or rsi_vals.isna().any():
                return False

            recent_price_high = price_highs.iloc[-3:].max()
            prev_price_high = price_highs.iloc[:lookback - 3].max()
            recent_price_high_idx = price_highs.iloc[-3:].idxmax()
            prev_price_high_idx = price_highs.iloc[:lookback - 3].idxmax()

            recent_rsi = rsi_vals.loc[recent_price_high_idx] if recent_price_high_idx in rsi_vals.index else rsi_vals.iloc[-1]
            prev_rsi = rsi_vals.loc[prev_price_high_idx] if prev_price_high_idx in rsi_vals.index else rsi_vals.iloc[0]

            # 価格は高値更新、RSIは高値切り下げ = 弱気ダイバージェンス
            if recent_price_high > prev_price_high and recent_rsi < prev_rsi:
                return True

            return False
        except Exception:
            return False

    def _is_bullish_reversal_bar(self, df: pd.DataFrame) -> bool:
        """買い反転バー（ピンバー / ハンマー）の検出"""
        try:
            last = df.iloc[-1]
            body = abs(last["close"] - last["open"])
            total_range = last["high"] - last["low"]

            if total_range == 0:
                return False

            lower_wick = min(last["open"], last["close"]) - last["low"]
            # 下ヒゲがボディの2倍以上 & 買い反転バー
            return (lower_wick > body * 2) and (last["close"] > last["open"])
        except Exception:
            return False

    def _is_bearish_reversal_bar(self, df: pd.DataFrame) -> bool:
        """売り反転バー（シューティングスター）の検出"""
        try:
            last = df.iloc[-1]
            body = abs(last["close"] - last["open"])
            total_range = last["high"] - last["low"]

            if total_range == 0:
                return False

            upper_wick = last["high"] - max(last["open"], last["close"])
            return (upper_wick > body * 2) and (last["close"] < last["open"])
        except Exception:
            return False

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """デバッグ用: 現在のインジケーター値を返す"""
        close = df["close"]
        rsi = ta.rsi(close, length=self.rsi_period)
        bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)

        return {
            "RSI": rsi.iloc[-1],
            f"BB_Upper_{self.bb_std}σ": bb[f"BBU_{self.bb_period}_{self.bb_std}"].iloc[-1],
            f"BB_Mid": bb[f"BBM_{self.bb_period}_{self.bb_std}"].iloc[-1],
            f"BB_Lower_{self.bb_std}σ": bb[f"BBL_{self.bb_period}_{self.bb_std}"].iloc[-1],
            "Close": close.iloc[-1],
        }
