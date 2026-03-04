"""
ブレイクアウトエンジン（EUR/USD H1専用チューニング）

パラメータ:
  - ブレイク判定: 過去20本の高値/安値
  - ATRフィルター: 現ATR ≥ 直近20本ATR平均
  - ボリューム確認: H1ティックボリューム
  - セッション重み: 重複時間×1.5, アジア×0.3

シグナル:
  +1.0: 強い上方ブレイクアウト（高値更新 + ATR拡大 + ボリューム増）
  -1.0: 強い下方ブレイクアウト（安値更新 + ATR拡大 + ボリューム増）
   0.0: ブレイクアウトなし

【期待値向上】ブレイクアウト前のスクイーズ検出:
  BBバンド幅がATRに対して縮小している → ブレイクアウト信頼度UP
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from logger_setup import get_logger

logger = get_logger("engine.breakout")

# デフォルトパラメータ
LOOKBACK_PERIOD = 20    # 過去20本の高値/安値
ATR_PERIOD = 14
ATR_LOOKBACK = 20       # ATR平均の算出期間
VOLUME_LOOKBACK = 20    # ボリューム平均の算出期間
VOLUME_MULTIPLIER = 1.2  # ボリュームが平均の1.2倍以上で確認
BB_SQUEEZE_PERIOD = 20
BB_SQUEEZE_STD = 2.0


class BreakoutEngine:
    """ATRブレイク + ボリューム確認ベースのブレイクアウトエンジン"""

    def __init__(
        self,
        lookback: int = LOOKBACK_PERIOD,
        atr_period: int = ATR_PERIOD,
        atr_lookback: int = ATR_LOOKBACK,
        volume_multiplier: float = VOLUME_MULTIPLIER,
    ):
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.volume_multiplier = volume_multiplier

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        ブレイクアウトシグナルを算出
        
        Args:
            df: H1足 OHLCVデータ（少なくとも40本以上）
            
        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        min_bars = max(self.lookback, self.atr_period, self.atr_lookback) + 10
        if len(df) < min_bars:
            logger.warning(f"データ不足: {len(df)}本（最低{min_bars}本必要）")
            return 0.0

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # 過去N本の高値/安値（最新バーを除く）
            range_high = high.iloc[-(self.lookback + 1):-1].max()
            range_low = low.iloc[-(self.lookback + 1):-1].min()
            current_close = close.iloc[-1]
            current_high = high.iloc[-1]
            current_low = low.iloc[-1]

            # ATR算出
            atr = ta.atr(high, low, close, length=self.atr_period)
            current_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-self.atr_lookback:].mean()

            # NaN チェック
            if any(pd.isna([range_high, range_low, current_atr, avg_atr])):
                return 0.0

            signal = 0.0

            # === 上方ブレイクアウト ===
            if current_close > range_high:
                signal = 0.4

                # ATR拡大確認（現在ATR ≥ 平均ATR）
                if current_atr >= avg_atr:
                    atr_ratio = min(current_atr / avg_atr, 2.0) if avg_atr > 0 else 1.0
                    signal += 0.2 * (atr_ratio - 1.0) + 0.2

                # ボリューム確認
                if self._check_volume_confirmation(df, bullish=True):
                    signal += 0.15

                # ブレイクアウトの強さ（レンジ幅に対するブレイク幅）
                range_width = range_high - range_low
                if range_width > 0:
                    break_strength = (current_close - range_high) / range_width
                    signal += min(break_strength * 0.3, 0.15)

                # 【期待値向上】スクイーズからのブレイクアウトはボーナス
                if self._detect_squeeze(df):
                    signal += 0.2

                signal = min(signal, 1.0)

            # === 下方ブレイクアウト ===
            elif current_close < range_low:
                signal = -0.4

                if current_atr >= avg_atr:
                    atr_ratio = min(current_atr / avg_atr, 2.0) if avg_atr > 0 else 1.0
                    signal -= 0.2 * (atr_ratio - 1.0) + 0.2

                if self._check_volume_confirmation(df, bullish=False):
                    signal -= 0.15

                range_width = range_high - range_low
                if range_width > 0:
                    break_strength = (range_low - current_close) / range_width
                    signal -= min(break_strength * 0.3, 0.15)

                if self._detect_squeeze(df):
                    signal -= 0.2

                signal = max(signal, -1.0)

            return round(signal, 3)

        except Exception as e:
            logger.error(f"ブレイクアウトシグナル算出エラー: {e}")
            return 0.0

    def _check_volume_confirmation(self, df: pd.DataFrame, bullish: bool) -> bool:
        """
        ボリューム確認（ティックボリューム）
        ブレイク方向にボリュームが平均以上あるか
        """
        try:
            # tick_volumeまたはvolumeカラムを使用
            vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
            if vol_col not in df.columns:
                return False

            volume = df[vol_col]
            current_vol = volume.iloc[-1]
            avg_vol = volume.iloc[-VOLUME_LOOKBACK:].mean()

            return current_vol >= avg_vol * self.volume_multiplier
        except Exception:
            return False

    def _detect_squeeze(self, df: pd.DataFrame) -> bool:
        """
        BBスクイーズの検出
        BBバンド幅が直近の最小レベルにある → ブレイクアウト前の圧縮状態
        """
        try:
            close = df["close"]
            bb = ta.bbands(close, length=BB_SQUEEZE_PERIOD, std=BB_SQUEEZE_STD)
            bb_upper = bb[[c for c in bb.columns if c.startswith("BBU_")][0]]
            bb_lower = bb[[c for c in bb.columns if c.startswith("BBL_")][0]]

            # バンド幅（%）
            bb_width = (bb_upper - bb_lower) / close
            bb_width = bb_width.dropna()

            if len(bb_width) < 20:
                return False

            # 直前バーのバンド幅が直近20本の下位25%以内
            current_width = bb_width.iloc[-2]  # ブレイク前のバー
            percentile_25 = bb_width.iloc[-20:].quantile(0.25)

            return current_width <= percentile_25
        except Exception:
            return False

    def get_breakout_levels(self, df: pd.DataFrame) -> dict:
        """現在のブレイクアウトレベルを返す"""
        high = df["high"]
        low = df["low"]

        range_high = high.iloc[-(self.lookback + 1):-1].max()
        range_low = low.iloc[-(self.lookback + 1):-1].min()

        return {
            "range_high": range_high,
            "range_low": range_low,
            "range_width": range_high - range_low,
            "current_close": df["close"].iloc[-1],
        }
