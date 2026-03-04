"""
サプライ/デマンド（オーダーブロック）エンジン（EUR/USD H1専用チューニング）

パラメータ:
  - オーダーブロック検出: 急騰/急落の起点ゾーン
  - ゾーン有効期限: 72本（3日間）
  - ATR基準: 1.5倍以上のインパルスムーブ
  - ゾーン内反発でエントリー

シグナル:
  +1.0: デマンドゾーン（買い注文集中ゾーン）で反発確認
  -1.0: サプライゾーン（売り注文集中ゾーン）で反発確認
   0.0: ゾーン外 or 有効ゾーンなし

戦略本質:
  機関投資家の大口注文が残るゾーンに価格が戻った時、
  再度同方向に反発しやすい性質を利用
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Optional

from logger_setup import get_logger

logger = get_logger("engine.supply_demand")

# デフォルトパラメータ
ATR_PERIOD = 14
IMPULSE_MULTIPLIER = 1.5   # ATR × N 以上の値動きをインパルスとみなす
ZONE_EXPIRY_BARS = 72      # ゾーンの有効期限（H1 x 72 = 3日）
ZONE_TOUCH_TOLERANCE = 0.3 # ゾーン幅に対するタッチ許容率
MAX_ZONES = 5              # 保持する直近ゾーン数


class SupplyDemandZone:
    """サプライ/デマンドゾーンデータクラス"""
    def __init__(self, zone_type: str, top: float, bottom: float, bar_index: int, strength: float):
        self.zone_type = zone_type  # "demand" or "supply"
        self.top = top
        self.bottom = bottom
        self.bar_index = bar_index
        self.strength = strength    # ゾーンの強さ (0-1)
        self.touches = 0           # タッチ回数（多いほど弱くなる）

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def width(self) -> float:
        return self.top - self.bottom


class SupplyDemandEngine:
    """オーダーブロック / サプライ・デマンドゾーンベースのエンジン"""

    def __init__(
        self,
        atr_period: int = ATR_PERIOD,
        impulse_multiplier: float = IMPULSE_MULTIPLIER,
        zone_expiry_bars: int = ZONE_EXPIRY_BARS,
        max_zones: int = MAX_ZONES,
    ):
        self.atr_period = atr_period
        self.impulse_multiplier = impulse_multiplier
        self.zone_expiry_bars = zone_expiry_bars
        self.max_zones = max_zones

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        サプライ/デマンドシグナルを算出

        Args:
            df: H1足 OHLCVデータ（少なくとも50本以上）

        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        min_bars = self.atr_period + self.zone_expiry_bars
        if len(df) < min_bars:
            logger.warning(f"データ不足: {len(df)}本（最低{min_bars}本必要）")
            return 0.0

        try:
            open_p = df["open"].values
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values

            # ATR算出
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            atr = atr_series.values

            n = len(close)

            # 1. ゾーン検出
            zones = self._detect_zones(open_p, high, low, close, atr, n)

            if not zones:
                return 0.0

            # 2. 現在価格がゾーン内にあるか判定
            current_close = close[-1]
            current_low = low[-1]
            current_high = high[-1]
            prev_close = close[-2]

            signal = 0.0

            for zone in zones:
                # ゾーン有効期限チェック
                age = n - 1 - zone.bar_index
                if age > self.zone_expiry_bars:
                    continue

                # 鮮度による減衰（新しいゾーンほど強い）
                freshness = max(0.0, 1.0 - (age / self.zone_expiry_bars) * 0.5)

                # タッチ回数による減衰（3回以上タッチされたゾーンは無効）
                if zone.touches >= 3:
                    continue
                touch_decay = 1.0 - zone.touches * 0.25

                zone_tolerance = zone.width * ZONE_TOUCH_TOLERANCE

                if zone.zone_type == "demand":
                    # デマンドゾーン: 価格が上からゾーン付近に下がってきた場合
                    if current_low <= zone.top + zone_tolerance and current_close > zone.bottom:
                        # 反発確認: 現在の足が陽線（始値 < 終値）
                        if close[-1] > open_p[-1]:
                            zone_score = zone.strength * freshness * touch_decay
                            # ゾーン内のどこにいるかで強さ調整
                            if current_low <= zone.bottom:
                                zone_score *= 1.2  # ゾーン下限を試した → 強い反発
                            signal = max(signal, min(zone_score, 1.0))

                elif zone.zone_type == "supply":
                    # サプライゾーン: 価格が下からゾーン付近に上がってきた場合
                    if current_high >= zone.bottom - zone_tolerance and current_close < zone.top:
                        # 反発確認: 現在の足が陰線（始値 > 終値）
                        if close[-1] < open_p[-1]:
                            zone_score = zone.strength * freshness * touch_decay
                            if current_high >= zone.top:
                                zone_score *= 1.2
                            neg_score = min(zone_score, 1.0)
                            if neg_score > abs(signal):
                                signal = -neg_score

            return round(signal, 4)

        except Exception as e:
            logger.error(f"サプライ/デマンド算出エラー: {e}", exc_info=True)
            return 0.0

    def _detect_zones(
        self,
        open_p: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray,
        n: int,
    ) -> List[SupplyDemandZone]:
        """
        オーダーブロック / サプライ・デマンドゾーンを検出

        インパルスムーブ（急騰/急落）の起点をゾーンとして記録
        """
        zones: List[SupplyDemandZone] = []
        scan_start = max(self.atr_period + 5, n - self.zone_expiry_bars - 10)

        for i in range(scan_start, n - 1):
            if pd.isna(atr[i]):
                continue

            body_size = abs(close[i] - open_p[i])
            candle_range = high[i] - low[i]
            threshold = atr[i] * self.impulse_multiplier

            # インパルスキャンドル検出（ATR × N 以上の大きなボディ）
            if body_size < threshold:
                continue

            # 方向判定
            if close[i] > open_p[i]:
                # 大陽線 → その起点（前の足のレンジ）がデマンドゾーン
                if i > 0:
                    zone_bottom = min(low[i - 1], low[i])
                    zone_top = max(open_p[i], close[i - 1])
                    if zone_top <= zone_bottom:
                        continue
                    # 強さ: ボディサイズ / ATR で正規化
                    strength = min(body_size / (atr[i] * 2), 1.0)
                    # 出来高確認（ティックボリュームがあれば）
                    strength = max(strength, 0.3)

                    zones.append(SupplyDemandZone(
                        zone_type="demand",
                        top=zone_top,
                        bottom=zone_bottom,
                        bar_index=i,
                        strength=strength,
                    ))

            elif close[i] < open_p[i]:
                # 大陰線 → その起点がサプライゾーン
                if i > 0:
                    zone_top = max(high[i - 1], high[i])
                    zone_bottom = min(open_p[i], close[i - 1])
                    if zone_top <= zone_bottom:
                        continue
                    strength = min(body_size / (atr[i] * 2), 1.0)
                    strength = max(strength, 0.3)

                    zones.append(SupplyDemandZone(
                        zone_type="supply",
                        top=zone_top,
                        bottom=zone_bottom,
                        bar_index=i,
                        strength=strength,
                    ))

        # ゾーンタッチ回数をカウント
        for zone in zones:
            for j in range(zone.bar_index + 1, n):
                if zone.zone_type == "demand" and low[j] <= zone.top:
                    zone.touches += 1
                elif zone.zone_type == "supply" and high[j] >= zone.bottom:
                    zone.touches += 1

        # 新しいゾーンを優先して保持
        zones.sort(key=lambda z: z.bar_index, reverse=True)
        return zones[:self.max_zones]

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """現在のゾーン情報を取得（ダッシュボード表示用）"""
        try:
            open_p = df["open"].values
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            atr = atr_series.values
            n = len(close)

            zones = self._detect_zones(open_p, high, low, close, atr, n)
            active_zones = [
                z for z in zones
                if (n - 1 - z.bar_index) <= self.zone_expiry_bars and z.touches < 3
            ]

            return {
                "active_demand_zones": len([z for z in active_zones if z.zone_type == "demand"]),
                "active_supply_zones": len([z for z in active_zones if z.zone_type == "supply"]),
                "nearest_zone_type": active_zones[0].zone_type if active_zones else None,
                "nearest_zone_mid": round(active_zones[0].mid, 5) if active_zones else None,
                "signal_score": self.get_signal(df),
            }
        except Exception:
            return {}
