"""
マーケットストラクチャー（構造転換）エンジン（EUR/USD H1専用チューニング）

パラメータ:
  - スイングポイント検出: 前後3本比較
  - レッグ判定: Higher High/Higher Low (上昇) or Lower Low/Lower High (下降)
  - Break of Structure (BoS): 構造が崩れた瞬間を検出
  - Change of Character (CHoCH): トレンド転換の初動

シグナル:
  +1.0: 下降構造が崩壊し上昇構造に転換（CHoCH上）
  -1.0: 上昇構造が崩壊し下降構造に転換（CHoCH下）
  +0.5: 上昇構造継続中のBoS（押し目買い）
  -0.5: 下降構造継続中のBoS（戻り売り）
   0.0: 構造不明瞭

戦略本質:
  Smart Money Concept (SMC) の基盤。機関トレーダーは
  HH/HL/LL/LH のスイング構造を重視し、BoS/CHoCH を
  エントリーの根拠にする。
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from logger_setup import get_logger

logger = get_logger("engine.market_structure")

# デフォルトパラメータ
SWING_TOLERANCE = 3     # スイングポイント検出の前後バー数
MIN_SWINGS = 4          # 構造判定に必要な最低スイング数
LOOKBACK_BARS = 60      # 構造分析の対象バー数


class SwingPoint:
    """スイングポイントデータ"""
    def __init__(self, index: int, price: float, swing_type: str):
        self.index = index
        self.price = price
        self.swing_type = swing_type  # "high" or "low"


class MarketStructureEngine:
    """マーケットストラクチャー（HH/HL/LL/LH + BoS/CHoCH）エンジン"""

    def __init__(
        self,
        swing_tolerance: int = SWING_TOLERANCE,
        min_swings: int = MIN_SWINGS,
        lookback_bars: int = LOOKBACK_BARS,
    ):
        self.swing_tolerance = swing_tolerance
        self.min_swings = min_swings
        self.lookback_bars = lookback_bars

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        マーケットストラクチャーシグナルを算出

        Args:
            df: H1足 OHLCVデータ（少なくとも60本以上）

        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        min_bars = self.lookback_bars + self.swing_tolerance
        if len(df) < min_bars:
            logger.warning(f"データ不足: {len(df)}本（最低{min_bars}本必要）")
            return 0.0

        try:
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            n = len(close)

            # 分析範囲を限定
            start = max(0, n - self.lookback_bars)

            # 1. スイングポイントを検出
            swing_highs = self._find_swing_highs(high, start, n)
            swing_lows = self._find_swing_lows(low, start, n)

            # 全スイングをマージして時系列順に
            all_swings = sorted(swing_highs + swing_lows, key=lambda s: s.index)

            if len(all_swings) < self.min_swings:
                return 0.0

            # 2. 市場構造を判定
            structure = self._analyze_structure(swing_highs, swing_lows)

            # 3. Break of Structure / Change of Character 検出
            signal = self._detect_bos_choch(
                structure, swing_highs, swing_lows, close, high, low, n
            )

            return round(signal, 4)

        except Exception as e:
            logger.error(f"マーケットストラクチャー算出エラー: {e}", exc_info=True)
            return 0.0

    def _find_swing_highs(self, high: np.ndarray, start: int, end: int) -> List[SwingPoint]:
        """スイング高値を検出"""
        swings = []
        t = self.swing_tolerance
        for i in range(max(start, t), end - t):
            is_swing = True
            for j in range(1, t + 1):
                if high[i] < high[i - j] or high[i] < high[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append(SwingPoint(i, float(high[i]), "high"))
        return swings

    def _find_swing_lows(self, low: np.ndarray, start: int, end: int) -> List[SwingPoint]:
        """スイング安値を検出"""
        swings = []
        t = self.swing_tolerance
        for i in range(max(start, t), end - t):
            is_swing = True
            for j in range(1, t + 1):
                if low[i] > low[i - j] or low[i] > low[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append(SwingPoint(i, float(low[i]), "low"))
        return swings

    def _analyze_structure(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> str:
        """
        直近のスイング構造を分析

        Returns:
            "bullish": HH + HL の上昇構造
            "bearish": LL + LH の下降構造
            "neutral": 不明瞭
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "neutral"

        # 直近2つのスイング高値/安値を比較
        sh_1 = swing_highs[-1]  # 最新
        sh_2 = swing_highs[-2]  # 1つ前
        sl_1 = swing_lows[-1]
        sl_2 = swing_lows[-2]

        hh = sh_1.price > sh_2.price   # Higher High
        hl = sl_1.price > sl_2.price   # Higher Low
        ll = sl_1.price < sl_2.price   # Lower Low
        lh = sh_1.price < sh_2.price   # Lower High

        if hh and hl:
            return "bullish"
        elif ll and lh:
            return "bearish"
        elif hh and ll:
            return "neutral"  # 拡大レンジ
        elif lh and hl:
            return "neutral"  # 縮小レンジ
        else:
            return "neutral"

    def _detect_bos_choch(
        self,
        structure: str,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        n: int,
    ) -> float:
        """
        Break of Structure (BoS) / Change of Character (CHoCH) を検出

        BoS: 構造が継続する方向へのブレイク（トレンド継続）
        CHoCH: 構造が反転する方向へのブレイク（トレンド転換）
        """
        signal = 0.0
        current_close = close[-1]

        # 直近数本（swing_tolerance以内）での判定
        recency_limit = n - 1 - self.swing_tolerance

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # --- 上昇構造（bullish）の場合 ---
        if structure == "bullish":
            last_swing_low = swing_lows[-1]

            # BoS上方: 直近のスイング高値を上抜け → 上昇トレンド継続
            last_swing_high = swing_highs[-1]
            if current_close > last_swing_high.price and last_swing_high.index >= recency_limit:
                # トレンド継続のBoS（中程度のシグナル）
                signal = 0.5

            # CHoCH下方: 直近のスイング安値を下抜け → 構造崩壊・転換
            if current_close < last_swing_low.price and last_swing_low.index >= recency_limit:
                # 上昇構造が崩壊 → 売りシグナル（強い）
                signal = -0.8

                # 複数足で下抜け確定していれば更に強化
                if n >= 3 and close[-2] < last_swing_low.price:
                    signal = -1.0

        # --- 下降構造（bearish）の場合 ---
        elif structure == "bearish":
            last_swing_high = swing_highs[-1]

            # BoS下方: 直近のスイング安値を下抜け → 下降トレンド継続
            last_swing_low = swing_lows[-1]
            if current_close < last_swing_low.price and last_swing_low.index >= recency_limit:
                signal = -0.5

            # CHoCH上方: 直近のスイング高値を上抜け → 構造崩壊・転換
            if current_close > last_swing_high.price and last_swing_high.index >= recency_limit:
                signal = 0.8

                if n >= 3 and close[-2] > last_swing_high.price:
                    signal = 1.0

        # --- ニュートラル（レンジ）の場合 ---
        elif structure == "neutral":
            # レンジの端でのブレイクアウトを検出
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                range_high = max(s.price for s in swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1].price
                range_low = min(s.price for s in swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1].price

                if current_close > range_high:
                    signal = 0.6  # レンジ上限ブレイク
                elif current_close < range_low:
                    signal = -0.6  # レンジ下限ブレイク

        return signal

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """現在の構造情報を取得（ダッシュボード表示用）"""
        try:
            high = df["high"].values
            low = df["low"].values
            n = len(high)
            start = max(0, n - self.lookback_bars)

            swing_highs = self._find_swing_highs(high, start, n)
            swing_lows = self._find_swing_lows(low, start, n)
            structure = self._analyze_structure(swing_highs, swing_lows)

            return {
                "structure": structure,
                "swing_high_count": len(swing_highs),
                "swing_low_count": len(swing_lows),
                "last_swing_high": round(swing_highs[-1].price, 5) if swing_highs else None,
                "last_swing_low": round(swing_lows[-1].price, 5) if swing_lows else None,
                "signal_score": self.get_signal(df),
            }
        except Exception:
            return {}
