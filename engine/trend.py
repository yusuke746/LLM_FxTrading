"""
トレンドフォローエンジン（EUR/USD H1専用チューニング）

パラメータ:
  - EMAクロス: 9/21/50
  - ADXフィルター: ≥ 25（明確なトレンドのみ）
  - セッション重み: ロンドン×1.0, アジア×0.5

シグナル:
  +1.0: 強い上昇トレンド（EMA9>21>50 + ADX≥25）
  -1.0: 強い下降トレンド（EMA9<21<50 + ADX≥25）
   0.0: トレンドなし

【期待値向上】マルチタイムフレーム確認:
  H4のEMA方向と一致する場合のみシグナルを強化（+0.2ボーナス）
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional

from logger_setup import get_logger

logger = get_logger("engine.trend")

# デフォルトパラメータ
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
ADX_PERIOD = 14
ADX_THRESHOLD = 25


class TrendEngine:
    """EMAクロス + ADXベースのトレンドフォローエンジン"""

    def __init__(
        self,
        ema_fast: int = EMA_FAST,
        ema_mid: int = EMA_MID,
        ema_slow: int = EMA_SLOW,
        adx_period: int = ADX_PERIOD,
        adx_threshold: float = ADX_THRESHOLD,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def get_signal(self, df: pd.DataFrame, h4_df: Optional[pd.DataFrame] = None) -> float:
        """
        トレンドフォローシグナルを算出
        
        Args:
            df: H1足 OHLCVデータ（少なくとも60本以上）
            h4_df: H4足データ（マルチタイムフレーム確認用、省略可）
            
        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        if len(df) < self.ema_slow + 10:
            logger.warning(f"データ不足: {len(df)}本（最低{self.ema_slow + 10}本必要）")
            return 0.0

        try:
            close = df["close"]

            # EMA算出
            ema_fast = ta.ema(close, length=self.ema_fast)
            ema_mid = ta.ema(close, length=self.ema_mid)
            ema_slow = ta.ema(close, length=self.ema_slow)

            # ADX算出
            adx_result = ta.adx(df["high"], df["low"], close, length=self.adx_period)
            adx = adx_result[f"ADX_{self.adx_period}"]
            plus_di = adx_result[f"DMP_{self.adx_period}"]
            minus_di = adx_result[f"DMN_{self.adx_period}"]

            # 最新値を取得
            current_ema_fast = ema_fast.iloc[-1]
            current_ema_mid = ema_mid.iloc[-1]
            current_ema_slow = ema_slow.iloc[-1]
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]

            # NaN チェック
            if any(pd.isna([current_ema_fast, current_ema_mid, current_ema_slow, current_adx])):
                return 0.0

            signal = 0.0

            # パーフェクトオーダー判定
            if current_ema_fast > current_ema_mid > current_ema_slow:
                # 上昇トレンド
                signal = 0.5
                if current_adx >= self.adx_threshold and current_plus_di > current_minus_di:
                    signal = 1.0
                elif current_adx >= self.adx_threshold * 0.8:
                    signal = 0.7

            elif current_ema_fast < current_ema_mid < current_ema_slow:
                # 下降トレンド
                signal = -0.5
                if current_adx >= self.adx_threshold and current_minus_di > current_plus_di:
                    signal = -1.0
                elif current_adx >= self.adx_threshold * 0.8:
                    signal = -0.7

            # EMAクロス直近発生の強化（勢い確認）
            if len(ema_fast) >= 3:
                prev_fast = ema_fast.iloc[-2]
                prev_mid = ema_mid.iloc[-2]
                # ゴールデンクロス直後
                if prev_fast <= prev_mid and current_ema_fast > current_ema_mid and signal > 0:
                    signal = min(signal + 0.2, 1.0)
                # デッドクロス直後
                elif prev_fast >= prev_mid and current_ema_fast < current_ema_mid and signal < 0:
                    signal = max(signal - 0.2, -1.0)

            # 【期待値向上】マルチタイムフレーム確認（H4）
            if h4_df is not None and len(h4_df) >= self.ema_slow + 5:
                mtf_bonus = self._check_h4_alignment(h4_df, signal)
                signal = np.clip(signal + mtf_bonus, -1.0, 1.0)

            return round(signal, 3)

        except Exception as e:
            logger.error(f"トレンドシグナル算出エラー: {e}")
            return 0.0

    def _check_h4_alignment(self, h4_df: pd.DataFrame, h1_signal: float) -> float:
        """
        H4足のEMA方向と一致性を確認
        一致していればボーナス、逆方向ならペナルティ
        """
        try:
            h4_ema_mid = ta.ema(h4_df["close"], length=self.ema_mid)
            h4_ema_slow = ta.ema(h4_df["close"], length=self.ema_slow)

            if pd.isna(h4_ema_mid.iloc[-1]) or pd.isna(h4_ema_slow.iloc[-1]):
                return 0.0

            h4_trend = 1.0 if h4_ema_mid.iloc[-1] > h4_ema_slow.iloc[-1] else -1.0

            if h1_signal > 0 and h4_trend > 0:
                return 0.2   # 上位足と一致 → 強化
            elif h1_signal < 0 and h4_trend < 0:
                return -0.2  # 上位足と一致 → 強化（下方向）
            elif (h1_signal > 0 and h4_trend < 0) or (h1_signal < 0 and h4_trend > 0):
                return -0.15 * np.sign(h1_signal)  # 上位足と不一致 → 弱化

            return 0.0
        except Exception:
            return 0.0

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """デバッグ用: 現在のインジケーター値を返す"""
        close = df["close"]
        ema_fast = ta.ema(close, length=self.ema_fast)
        ema_mid = ta.ema(close, length=self.ema_mid)
        ema_slow = ta.ema(close, length=self.ema_slow)
        adx_result = ta.adx(df["high"], df["low"], close, length=self.adx_period)

        return {
            f"EMA_{self.ema_fast}": ema_fast.iloc[-1],
            f"EMA_{self.ema_mid}": ema_mid.iloc[-1],
            f"EMA_{self.ema_slow}": ema_slow.iloc[-1],
            f"ADX_{self.adx_period}": adx_result[f"ADX_{self.adx_period}"].iloc[-1],
            f"DI+": adx_result[f"DMP_{self.adx_period}"].iloc[-1],
            f"DI-": adx_result[f"DMN_{self.adx_period}"].iloc[-1],
        }
