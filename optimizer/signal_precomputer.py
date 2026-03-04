"""
バックテスト用シグナル事前計算モジュール

旧 _get_simple_signal() の置き換え:
  旧: ライブとは全く異なる簡易ロジック → 最適化が無意味
  新: 全7エンジンのロジックを忠実に再現（インジケータは1回だけ計算）

設計:
  1. pandas_ta で全インジケータを1回計算（O(N)）
  2. 各エンジンのロジックを事前計算済み配列上で再現（O(N)）
  3. レジーム判定もバー毎に事前計算
  4. グリッドサーチではSL/TP/BE/Trailのみ変化 → シグナル配列は再利用

パフォーマンス:
  計算1回: ~960本 × 8エンジン ≈ 100ms
  グリッドサーチ: 600コンボ × シグナル参照 → 高速
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple

from logger_setup import get_logger

logger = get_logger("optimizer.precomputer")


class SignalPrecomputer:
    """全エンジンのシグナルを事前計算"""

    def __init__(self):
        # EET/EEST サーバー時間でのセッション開始時刻 (hour)
        self.london_open_hour = 10   # EET
        self.ny_open_hour = 16       # EET

    def precompute(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        全エンジンのシグナルを事前計算

        Args:
            df: H1足 OHLCVデータ（datetime, open, high, low, close, tick_volume）

        Returns:
            dict: {
                "composite_direction": int配列 (+1=BUY, -1=SELL, 0=NONE),
                "composite_score": float配列,
                "regime": str配列,
                "session": str配列,
                "raw_signals": {engine_name: float配列},
            }
        """
        n = len(df)
        if n < 100:
            return self._empty_precompute(n)

        # === Step 1: 全インジケータ計算（1回のみ） ===
        indicators = self._compute_all_indicators(df)

        # === Step 2: 各エンジンのシグナルを事前計算 ===
        raw_signals = {
            "trend": self._compute_trend(df, indicators, n),
            "mean_rev": self._compute_mean_reversion(df, indicators, n),
            "breakout": self._compute_breakout(df, indicators, n),
            "momentum_div": self._compute_momentum_divergence(df, indicators, n),
            "supply_demand": self._compute_supply_demand(df, indicators, n),
            "session_orb": self._compute_session_orb(df, indicators, n),
            "market_structure": self._compute_market_structure(df, indicators, n),
        }

        # === Step 3: レジーム事前計算 ===
        regimes = self._compute_regimes(indicators, n)

        # === Step 4: セッション事前計算 ===
        sessions = self._compute_sessions(df, n)

        # === Step 5: 合成シグナル（レジームゲーティング + コンフルエンス） ===
        composite_dir, composite_score = self._compute_composite(
            raw_signals, regimes, sessions, n
        )

        return {
            "composite_direction": composite_dir,
            "composite_score": composite_score,
            "regime": regimes["regime_name"],
            "session": sessions,
            "raw_signals": raw_signals,
        }

    # =========================================================================
    # インジケータ一括計算
    # =========================================================================

    def _compute_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """全エンジンが必要とするインジケータを一括計算"""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]

        # Volume（存在しない場合はゼロ埋め）
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        volume = df[vol_col].values if vol_col in df.columns else np.zeros(len(df))

        # --- EMA ---
        ema9 = ta.ema(close, length=9).values
        ema21 = ta.ema(close, length=21).values
        ema50 = ta.ema(close, length=50).values

        # --- ADX + DI ---
        adx_df = ta.adx(high, low, close, length=14)
        adx = adx_df["ADX_14"].values
        di_plus = adx_df["DMP_14"].values
        di_minus = adx_df["DMN_14"].values

        # --- RSI ---
        rsi = ta.rsi(close, length=14).values

        # --- ボリンジャーバンド ---
        bb = ta.bbands(close, length=20, std=2.0)
        bb_upper_col = [c for c in bb.columns if c.startswith("BBU_")][0]
        bb_lower_col = [c for c in bb.columns if c.startswith("BBL_")][0]
        bb_mid_col = [c for c in bb.columns if c.startswith("BBM_")][0]
        bb_upper = bb[bb_upper_col].values
        bb_lower = bb[bb_lower_col].values
        bb_mid = bb[bb_mid_col].values

        # --- ATR ---
        atr = ta.atr(high, low, close, length=14).values

        # --- MACD ---
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_hist_col = [c for c in macd_df.columns if "MACDh" in c or "MACD_" not in c and "s_" not in c]
        # pandas_ta の MACD 列名: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        macd_hist = macd_df[[c for c in macd_df.columns if c.startswith("MACDh_")][0]].values

        # --- StochRSI ---
        stoch_rsi = ta.stochrsi(close, length=14)
        stoch_k_col = [c for c in stoch_rsi.columns if c.startswith("STOCHRSIk_")][0]
        stoch_k = stoch_rsi[stoch_k_col].values

        # --- BB幅（レジーム用） ---
        bb_width = (bb_upper - bb_lower) / close.values

        return {
            "close": close.values,
            "high": high.values,
            "low": low.values,
            "open": open_.values,
            "volume": volume,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "adx": adx,
            "di_plus": di_plus,
            "di_minus": di_minus,
            "rsi": rsi,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "atr": atr,
            "macd_hist": macd_hist,
            "stoch_k": stoch_k,
            "bb_width": bb_width,
        }

    # =========================================================================
    # エンジン 1: Trend
    # =========================================================================

    def _compute_trend(self, df, ind, n) -> np.ndarray:
        """TrendEngine ロジック再現"""
        signals = np.zeros(n)
        ema9, ema21, ema50 = ind["ema9"], ind["ema21"], ind["ema50"]
        adx, dip, dim = ind["adx"], ind["di_plus"], ind["di_minus"]

        for i in range(60, n):
            if np.isnan(ema9[i]) or np.isnan(adx[i]):
                continue

            signal = 0.0

            # パーフェクトオーダー
            if ema9[i] > ema21[i] > ema50[i]:
                signal = 0.5
                if adx[i] >= 25 and dip[i] > dim[i]:
                    signal = 1.0
                elif adx[i] >= 20:
                    signal = 0.7
            elif ema9[i] < ema21[i] < ema50[i]:
                signal = -0.5
                if adx[i] >= 25 and dim[i] > dip[i]:
                    signal = -1.0
                elif adx[i] >= 20:
                    signal = -0.7

            # ゴールデン/デッドクロス
            if i > 0 and not np.isnan(ema9[i - 1]):
                if ema9[i] > ema21[i] and ema9[i - 1] <= ema21[i - 1]:
                    signal += 0.2
                elif ema9[i] < ema21[i] and ema9[i - 1] >= ema21[i - 1]:
                    signal -= 0.2

            signals[i] = np.clip(signal, -1.0, 1.0)

        return signals

    # =========================================================================
    # エンジン 2: MeanReversion
    # =========================================================================

    def _compute_mean_reversion(self, df, ind, n) -> np.ndarray:
        """MeanReversionEngine ロジック再現"""
        signals = np.zeros(n)
        close, high, low, open_ = ind["close"], ind["high"], ind["low"], ind["open"]
        rsi, bb_upper, bb_lower = ind["rsi"], ind["bb_upper"], ind["bb_lower"]

        for i in range(60, n):
            if np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
                continue

            signal = 0.0

            if rsi[i] < 30:
                signal += 0.4
                if close[i] <= bb_lower[i]:
                    signal += 0.3
                if rsi[i] < 20:
                    signal += 0.15

                # 強気ダイバージェンス（簡易版）
                if i >= 10:
                    lookback = min(10, i)
                    recent_lows = low[i - lookback:i]
                    recent_rsi = rsi[i - lookback:i]
                    if len(recent_lows) >= 7:
                        price_recent = np.min(recent_lows[-3:])
                        price_older = np.min(recent_lows[:-3])
                        valid_rsi_recent = recent_rsi[-3:][~np.isnan(recent_rsi[-3:])]
                        valid_rsi_older = recent_rsi[:-3][~np.isnan(recent_rsi[:-3])]
                        if (len(valid_rsi_recent) > 0 and len(valid_rsi_older) > 0
                                and price_recent < price_older
                                and np.min(valid_rsi_recent) > np.min(valid_rsi_older)):
                            signal += 0.25

                # 反転バー（下ヒゲ長い陽線）
                body = abs(close[i] - open_[i])
                lower_wick = min(close[i], open_[i]) - low[i]
                if body > 0 and lower_wick > body * 2 and close[i] > open_[i]:
                    signal += 0.1

            elif rsi[i] > 70:
                signal -= 0.4
                if close[i] >= bb_upper[i]:
                    signal -= 0.3
                if rsi[i] > 80:
                    signal -= 0.15

                # 弱気ダイバージェンス
                if i >= 10:
                    lookback = min(10, i)
                    recent_highs = high[i - lookback:i]
                    recent_rsi = rsi[i - lookback:i]
                    if len(recent_highs) >= 7:
                        price_recent = np.max(recent_highs[-3:])
                        price_older = np.max(recent_highs[:-3])
                        valid_rsi_recent = recent_rsi[-3:][~np.isnan(recent_rsi[-3:])]
                        valid_rsi_older = recent_rsi[:-3][~np.isnan(recent_rsi[:-3])]
                        if (len(valid_rsi_recent) > 0 and len(valid_rsi_older) > 0
                                and price_recent > price_older
                                and np.max(valid_rsi_recent) < np.max(valid_rsi_older)):
                            signal -= 0.25

                # 反転バー（上ヒゲ長い陰線）
                body = abs(close[i] - open_[i])
                upper_wick = high[i] - max(close[i], open_[i])
                if body > 0 and upper_wick > body * 2 and close[i] < open_[i]:
                    signal -= 0.1

            signals[i] = np.clip(signal, -1.0, 1.0)

        return signals

    # =========================================================================
    # エンジン 3: Breakout
    # =========================================================================

    def _compute_breakout(self, df, ind, n) -> np.ndarray:
        """BreakoutEngine ロジック再現"""
        signals = np.zeros(n)
        close, high, low = ind["close"], ind["high"], ind["low"]
        atr, volume = ind["atr"], ind["volume"]
        bb_upper, bb_lower = ind["bb_upper"], ind["bb_lower"]

        for i in range(60, n):
            if np.isnan(atr[i]):
                continue

            # 過去20本のレンジ（最新バー除く）
            if i < 20:
                continue
            range_high = np.max(high[i - 20:i])
            range_low = np.min(low[i - 20:i])
            range_width = range_high - range_low
            if range_width <= 0:
                continue

            # ATR比
            atr_lookback = atr[max(0, i - 20):i]
            atr_lookback = atr_lookback[~np.isnan(atr_lookback)]
            avg_atr = np.mean(atr_lookback) if len(atr_lookback) > 0 else atr[i]

            # ボリューム平均
            vol_lookback = volume[max(0, i - 20):i]
            avg_vol = np.mean(vol_lookback) if len(vol_lookback) > 0 else 1.0

            # BB スクイーズ判定
            is_squeeze = False
            if i > 0 and not np.isnan(bb_upper[i - 1]) and not np.isnan(bb_lower[i - 1]):
                bw_current = bb_upper[i - 1] - bb_lower[i - 1]
                bw_lookback = []
                for j in range(max(0, i - 20), i):
                    if not np.isnan(bb_upper[j]) and not np.isnan(bb_lower[j]):
                        bw_lookback.append(bb_upper[j] - bb_lower[j])
                if bw_lookback:
                    bw_pct = sum(1 for bw in bw_lookback if bw < bw_current) / len(bw_lookback)
                    is_squeeze = bw_pct <= 0.25

            signal = 0.0

            if close[i] > range_high:
                signal = 0.4
                # ATR 確認
                if avg_atr > 0:
                    atr_ratio = min(atr[i] / avg_atr, 2.0)
                    if atr_ratio >= 1.0:
                        signal += 0.2 * (atr_ratio - 1.0) + 0.2
                # ボリューム確認
                if avg_vol > 0 and volume[i] >= avg_vol * 1.2:
                    signal += 0.15
                # ブレイク強度
                break_strength = min((close[i] - range_high) / range_width * 0.3, 0.15)
                signal += break_strength
                # スクイーズボーナス
                if is_squeeze:
                    signal += 0.2

            elif close[i] < range_low:
                signal = -0.4
                if avg_atr > 0:
                    atr_ratio = min(atr[i] / avg_atr, 2.0)
                    if atr_ratio >= 1.0:
                        signal -= 0.2 * (atr_ratio - 1.0) + 0.2
                if avg_vol > 0 and volume[i] >= avg_vol * 1.2:
                    signal -= 0.15
                break_strength = min((range_low - close[i]) / range_width * 0.3, 0.15)
                signal -= break_strength
                if is_squeeze:
                    signal -= 0.2

            signals[i] = np.clip(signal, -1.0, 1.0)

        return signals

    # =========================================================================
    # エンジン 4: MomentumDivergence
    # =========================================================================

    def _compute_momentum_divergence(self, df, ind, n) -> np.ndarray:
        """MomentumDivergenceEngine ロジック再現"""
        signals = np.zeros(n)
        close, high, low = ind["close"], ind["high"], ind["low"]
        rsi, macd_hist, stoch_k = ind["rsi"], ind["macd_hist"], ind["stoch_k"]

        swing_tolerance = 3
        div_lookback = 20

        for i in range(60, n):
            if np.isnan(rsi[i]) or np.isnan(macd_hist[i]):
                continue

            start = max(0, i - div_lookback)
            window_lows = low[start:i + 1]
            window_highs = high[start:i + 1]
            window_rsi = rsi[start:i + 1]
            window_hist = macd_hist[start:i + 1]

            # スイング安値検出
            swing_low_indices = []
            for j in range(swing_tolerance, len(window_lows) - swing_tolerance):
                is_swing = True
                for k in range(1, swing_tolerance + 1):
                    if window_lows[j] > window_lows[j - k] or window_lows[j] > window_lows[j + k]:
                        is_swing = False
                        break
                if is_swing:
                    swing_low_indices.append(j)

            # スイング高値検出
            swing_high_indices = []
            for j in range(swing_tolerance, len(window_highs) - swing_tolerance):
                is_swing = True
                for k in range(1, swing_tolerance + 1):
                    if window_highs[j] < window_highs[j - k] or window_highs[j] < window_highs[j + k]:
                        is_swing = False
                        break
                if is_swing:
                    swing_high_indices.append(j)

            bullish_score = 0.0
            bearish_score = 0.0

            # --- 強気ダイバージェンス ---
            if len(swing_low_indices) >= 2:
                recent_sw = swing_low_indices[-1]
                older_sw = swing_low_indices[-2]
                if (recent_sw > len(window_lows) - 1 - swing_tolerance - 5):
                    # 価格安値更新 + RSI安値切上げ
                    if (window_lows[recent_sw] < window_lows[older_sw]
                            and not np.isnan(window_rsi[recent_sw])
                            and not np.isnan(window_rsi[older_sw])
                            and window_rsi[recent_sw] > window_rsi[older_sw]):
                        bullish_score += 0.5
                        # MACD確認
                        if (not np.isnan(window_hist[recent_sw])
                                and not np.isnan(window_hist[older_sw])
                                and window_hist[recent_sw] > window_hist[older_sw]):
                            bullish_score += 0.25
            elif len(swing_low_indices) == 0:
                # スイングなし: 直近3本の最安値チェック
                if len(window_lows) >= 3:
                    last3 = window_lows[-3:]
                    min_idx = np.argmin(last3)
                    if min_idx == len(last3) - 1:  # 末尾が最安値
                        pass  # スイング確定していないので無視

            if bullish_score > 0:
                if not np.isnan(rsi[i]) and rsi[i] < 35:
                    bullish_score += 0.15
                if not np.isnan(stoch_k[i]) and stoch_k[i] < 20:
                    bullish_score += 0.15

            # --- 弱気ダイバージェンス ---
            if len(swing_high_indices) >= 2:
                recent_sw = swing_high_indices[-1]
                older_sw = swing_high_indices[-2]
                if (recent_sw > len(window_highs) - 1 - swing_tolerance - 5):
                    if (window_highs[recent_sw] > window_highs[older_sw]
                            and not np.isnan(window_rsi[recent_sw])
                            and not np.isnan(window_rsi[older_sw])
                            and window_rsi[recent_sw] < window_rsi[older_sw]):
                        bearish_score += 0.5
                        if (not np.isnan(window_hist[recent_sw])
                                and not np.isnan(window_hist[older_sw])
                                and window_hist[recent_sw] < window_hist[older_sw]):
                            bearish_score += 0.25

            if bearish_score > 0:
                if not np.isnan(rsi[i]) and rsi[i] > 65:
                    bearish_score += 0.15
                if not np.isnan(stoch_k[i]) and stoch_k[i] > 80:
                    bearish_score += 0.15

            # 最終判定
            if bullish_score > bearish_score:
                signals[i] = min(bullish_score, 1.0)
            elif bearish_score > bullish_score:
                signals[i] = -min(bearish_score, 1.0)

        return signals

    # =========================================================================
    # エンジン 5: SupplyDemand
    # =========================================================================

    def _compute_supply_demand(self, df, ind, n) -> np.ndarray:
        """SupplyDemandEngine ロジック再現（ステートフル: ゾーン追跡）"""
        signals = np.zeros(n)
        close, high, low, open_ = ind["close"], ind["high"], ind["low"], ind["open"]
        atr = ind["atr"]

        # ゾーン管理
        demand_zones = []  # [(zone_low, zone_high, creation_bar, touches, strength)]
        supply_zones = []

        max_zones = 5
        zone_expiry = 72
        max_touches = 3

        for i in range(60, n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            current_atr = atr[i]
            body = abs(close[i] - open_[i])

            # ゾーン検出: インパルスムーブ
            if body >= current_atr * 1.5:
                strength = min(body / (current_atr * 2), 1.0)
                strength = max(strength, 0.3)

                if close[i] > open_[i]:  # 大陽線 → デマンドゾーン
                    if i > 0:
                        zone_low = min(open_[i], low[i - 1])
                        zone_high = max(open_[i], high[i - 1]) if i > 0 else open_[i]
                        zone_high = min(zone_high, open_[i] + current_atr * 0.3)
                        demand_zones.append(
                            (zone_low, zone_high, i, 0, strength)
                        )
                        if len(demand_zones) > max_zones:
                            demand_zones = demand_zones[-max_zones:]
                else:  # 大陰線 → サプライゾーン
                    if i > 0:
                        zone_high = max(open_[i], high[i - 1])
                        zone_low = min(open_[i], low[i - 1]) if i > 0 else open_[i]
                        zone_low = max(zone_low, open_[i] - current_atr * 0.3)
                        supply_zones.append(
                            (zone_low, zone_high, i, 0, strength)
                        )
                        if len(supply_zones) > max_zones:
                            supply_zones = supply_zones[-max_zones:]

            # 期限切れ・タッチ過多のゾーンを除去
            demand_zones = [
                z for z in demand_zones
                if (i - z[2]) < zone_expiry and z[3] < max_touches
            ]
            supply_zones = [
                z for z in supply_zones
                if (i - z[2]) < zone_expiry and z[3] < max_touches
            ]

            # === デマンドゾーン反発チェック ===
            best_demand_score = 0.0
            tolerance = current_atr * 0.2
            new_demand = []
            for z_low, z_high, z_bar, z_touches, z_strength in demand_zones:
                age = i - z_bar
                if age < 2:
                    new_demand.append((z_low, z_high, z_bar, z_touches, z_strength))
                    continue

                # 安値がゾーン上限+許容範囲内
                if low[i] <= z_high + tolerance and close[i] > z_low:
                    if close[i] > open_[i]:  # 陽線
                        freshness = 1.0 - (age / zone_expiry) * 0.5
                        touch_decay = max(1.0 - z_touches * 0.25, 0.2)
                        score = z_strength * freshness * touch_decay

                        # ゾーン下限テスト
                        if low[i] <= z_low:
                            score *= 1.2

                        best_demand_score = max(best_demand_score, score)
                        new_demand.append((z_low, z_high, z_bar, z_touches + 1, z_strength))
                    else:
                        new_demand.append((z_low, z_high, z_bar, z_touches, z_strength))
                else:
                    new_demand.append((z_low, z_high, z_bar, z_touches, z_strength))
            demand_zones = new_demand

            # === サプライゾーン反発チェック ===
            best_supply_score = 0.0
            new_supply = []
            for z_low, z_high, z_bar, z_touches, z_strength in supply_zones:
                age = i - z_bar
                if age < 2:
                    new_supply.append((z_low, z_high, z_bar, z_touches, z_strength))
                    continue

                if high[i] >= z_low - tolerance and close[i] < z_high:
                    if close[i] < open_[i]:  # 陰線
                        freshness = 1.0 - (age / zone_expiry) * 0.5
                        touch_decay = max(1.0 - z_touches * 0.25, 0.2)
                        score = z_strength * freshness * touch_decay

                        if high[i] >= z_high:
                            score *= 1.2

                        best_supply_score = max(best_supply_score, score)
                        new_supply.append((z_low, z_high, z_bar, z_touches + 1, z_strength))
                    else:
                        new_supply.append((z_low, z_high, z_bar, z_touches, z_strength))
                else:
                    new_supply.append((z_low, z_high, z_bar, z_touches, z_strength))
            supply_zones = new_supply

            # 最終判定
            if best_demand_score > best_supply_score and best_demand_score > 0:
                signals[i] = min(best_demand_score, 1.0)
            elif best_supply_score > best_demand_score and best_supply_score > 0:
                signals[i] = -min(best_supply_score, 1.0)

        return signals

    # =========================================================================
    # エンジン 6: SessionORB
    # =========================================================================

    def _compute_session_orb(self, df, ind, n) -> np.ndarray:
        """SessionORBEngine ロジック再現"""
        signals = np.zeros(n)

        # datetime列チェック
        if "datetime" not in df.columns:
            return signals

        close, high, low = ind["close"], ind["high"], ind["low"]
        atr, volume = ind["atr"], ind["volume"]

        # datetime配列をhour配列に変換
        try:
            dt_col = pd.to_datetime(df["datetime"])
            hours = dt_col.dt.hour.values
        except Exception:
            return signals

        # ORBレンジを追跡
        london_orb_high = None
        london_orb_low = None
        ny_orb_high = None
        ny_orb_low = None

        for i in range(60, n):
            if np.isnan(atr[i]) or atr[i] <= 0:
                continue

            h = hours[i]

            # ロンドンORBセット（10:00 EET）
            if h == self.london_open_hour:
                london_orb_high = high[i]
                london_orb_low = low[i]
            # NY ORBセット（16:00 EET）
            if h == self.ny_open_hour:
                ny_orb_high = high[i]
                ny_orb_low = low[i]

            # ロンドンORBチェック（11:00, 12:00）
            london_signal = 0.0
            if london_orb_high is not None and h in (self.london_open_hour + 1, self.london_open_hour + 2):
                orb_range = london_orb_high - london_orb_low
                if close[i] > london_orb_high:
                    london_signal = 0.4 + min((close[i] - london_orb_high) / atr[i] * 0.3, 0.5)
                    if orb_range < atr[i] * 0.5:
                        london_signal += 0.1
                elif close[i] < london_orb_low:
                    london_signal = -0.4 - min((london_orb_low - close[i]) / atr[i] * 0.3, 0.5)
                    if orb_range < atr[i] * 0.5:
                        london_signal -= 0.1

                # ボリューム確認
                if london_signal != 0.0:
                    vol_lookback = volume[max(0, i - 20):i]
                    avg_vol = np.mean(vol_lookback) if len(vol_lookback) > 0 else 1.0
                    if avg_vol > 0 and volume[i] < avg_vol * 1.1:
                        london_signal *= 0.7

            # NY ORBチェック（17:00, 18:00）
            ny_signal = 0.0
            if ny_orb_high is not None and h in (self.ny_open_hour + 1, self.ny_open_hour + 2):
                orb_range = ny_orb_high - ny_orb_low
                if close[i] > ny_orb_high:
                    ny_signal = 0.4 + min((close[i] - ny_orb_high) / atr[i] * 0.3, 0.5)
                    if orb_range < atr[i] * 0.5:
                        ny_signal += 0.1
                elif close[i] < ny_orb_low:
                    ny_signal = -0.4 - min((ny_orb_low - close[i]) / atr[i] * 0.3, 0.5)
                    if orb_range < atr[i] * 0.5:
                        ny_signal -= 0.1

                if ny_signal != 0.0:
                    vol_lookback = volume[max(0, i - 20):i]
                    avg_vol = np.mean(vol_lookback) if len(vol_lookback) > 0 else 1.0
                    if avg_vol > 0 and volume[i] < avg_vol * 1.1:
                        ny_signal *= 0.7

            # 絶対値が大きい方を採用
            if abs(london_signal) >= abs(ny_signal):
                signals[i] = np.clip(london_signal, -1.0, 1.0)
            else:
                signals[i] = np.clip(ny_signal, -1.0, 1.0)

        return signals

    # =========================================================================
    # エンジン 7: MarketStructure
    # =========================================================================

    def _compute_market_structure(self, df, ind, n) -> np.ndarray:
        """MarketStructureEngine ロジック再現"""
        signals = np.zeros(n)
        close, high, low = ind["close"], ind["high"], ind["low"]

        swing_tolerance = 3
        analysis_window = 60
        min_swings = 4

        for i in range(60, n):
            start = max(0, i - analysis_window)
            window_highs = high[start:i + 1]
            window_lows = low[start:i + 1]
            wn = len(window_highs)

            # スイングポイント検出
            swing_highs = []  # (index_in_window, value)
            swing_lows = []

            for j in range(swing_tolerance, wn - swing_tolerance):
                # スイング高値
                is_high = True
                for k in range(1, swing_tolerance + 1):
                    if window_highs[j] < window_highs[j - k] or window_highs[j] < window_highs[j + k]:
                        is_high = False
                        break
                if is_high:
                    swing_highs.append((j, window_highs[j]))

                # スイング安値
                is_low = True
                for k in range(1, swing_tolerance + 1):
                    if window_lows[j] > window_lows[j - k] or window_lows[j] > window_lows[j + k]:
                        is_low = False
                        break
                if is_low:
                    swing_lows.append((j, window_lows[j]))

            if len(swing_highs) < 2 or len(swing_lows) < 2:
                continue

            # 直近2つのスイングで構造判定
            sh1_val = swing_highs[-2][1]
            sh2_val = swing_highs[-1][1]
            sl1_val = swing_lows[-2][1]
            sl2_val = swing_lows[-1][1]

            hh = sh2_val > sh1_val  # Higher High
            lh = sh2_val < sh1_val  # Lower High
            hl = sl2_val > sl1_val  # Higher Low
            ll = sl2_val < sl1_val  # Lower Low

            if hh and hl:
                structure = "bullish"
            elif ll and lh:
                structure = "bearish"
            else:
                structure = "neutral"

            # BoS/CHoCH 検出
            recency_limit = wn - 1 - swing_tolerance

            if structure == "bearish":
                # CHoCH上方: 弱気→強気への構造崩壊
                latest_high_idx = swing_highs[-1][0]
                latest_high_val = swing_highs[-1][1]
                if latest_high_idx >= recency_limit:
                    if close[i] > latest_high_val:
                        signal = 0.8
                        if i > 0 and close[i - 1] > latest_high_val:
                            signal = 1.0
                        signals[i] = signal
                        continue

            elif structure == "bullish":
                # CHoCH下方: 強気→弱気への構造崩壊
                latest_low_idx = swing_lows[-1][0]
                latest_low_val = swing_lows[-1][1]
                if latest_low_idx >= recency_limit:
                    if close[i] < latest_low_val:
                        signal = -0.8
                        if i > 0 and close[i - 1] < latest_low_val:
                            signal = -1.0
                        signals[i] = signal
                        continue

            if structure == "bullish":
                # BoS上方: トレンド継続
                latest_high_val = swing_highs[-1][1]
                if close[i] > latest_high_val:
                    signals[i] = 0.5
            elif structure == "bearish":
                # BoS下方: トレンド継続
                latest_low_val = swing_lows[-1][1]
                if close[i] < latest_low_val:
                    signals[i] = -0.5
            elif structure == "neutral":
                # レンジブレイク
                if len(swing_highs) >= 3:
                    range_high = max(sh[1] for sh in swing_highs[-3:])
                    if close[i] > range_high:
                        signals[i] = 0.6
                if len(swing_lows) >= 3:
                    range_low = min(sl[1] for sl in swing_lows[-3:])
                    if close[i] < range_low:
                        signals[i] = -0.6

        return signals

    # =========================================================================
    # レジーム事前計算
    # =========================================================================

    def _compute_regimes(self, ind, n) -> Dict:
        """レジーム判定を全バーで事前計算"""
        from engine.regime import Regime, REGIME_GATES

        adx = ind["adx"]
        bb_width = ind["bb_width"]
        atr = ind["atr"]
        close = ind["close"]

        regime_names = np.full(n, "ranging", dtype=object)
        gates_tf = np.full(n, 0.5)
        gates_mr = np.full(n, 0.5)
        gates_st = np.full(n, 0.5)
        threshold_adj = np.zeros(n)

        atr_lookback = 100

        for i in range(60, n):
            if np.isnan(adx[i]) or np.isnan(atr[i]):
                continue

            current_adx = adx[i]

            # BB幅の中央値
            bw_start = max(0, i - atr_lookback)
            bw_slice = bb_width[bw_start:i + 1]
            bw_valid = bw_slice[~np.isnan(bw_slice)]
            bb_median = np.median(bw_valid) if len(bw_valid) > 0 else 0
            current_bw = bb_width[i] if not np.isnan(bb_width[i]) else 0

            # ATRパーセンタイル
            atr_start = max(0, i - atr_lookback)
            atr_slice = atr[atr_start:i + 1]
            atr_valid = atr_slice[~np.isnan(atr_slice)]
            if len(atr_valid) < 20:
                continue
            atr_pct = float(np.sum(atr_valid < atr[i]) / len(atr_valid) * 100)

            # スコアリング（regime.py と同じロジック）
            scores = {"trending": 0.0, "ranging": 0.0, "volatile": 0.0, "quiet": 0.0}

            if current_adx >= 25:
                strength = min((current_adx - 25) / 15, 1.0)
                scores["trending"] += 0.5 + 0.3 * strength
            elif current_adx <= 20:
                strength = min((20 - current_adx) / 10, 1.0)
                scores["ranging"] += 0.5 + 0.3 * strength
            else:
                scores["trending"] += 0.2
                scores["ranging"] += 0.2

            if bb_median > 0:
                bb_ratio = current_bw / bb_median
                if bb_ratio > 1.3:
                    scores["trending"] += 0.2
                    scores["volatile"] += 0.15
                elif bb_ratio < 0.7:
                    scores["ranging"] += 0.2
                    scores["quiet"] += 0.15

            if atr_pct >= 75:
                scores["volatile"] += 0.4
                if current_adx >= 25:
                    scores["trending"] += 0.15
            elif atr_pct <= 25:
                scores["quiet"] += 0.4
                if current_adx <= 20:
                    scores["quiet"] += 0.15

            best = max(scores, key=scores.get)

            regime_map = {
                "trending": Regime.TRENDING,
                "ranging": Regime.RANGING,
                "volatile": Regime.VOLATILE,
                "quiet": Regime.QUIET,
            }
            regime = regime_map[best]
            gates = REGIME_GATES[regime]

            regime_names[i] = best
            gates_tf[i] = gates["trend_follow"]
            gates_mr[i] = gates["mean_revert"]
            gates_st[i] = gates["structural"]

            if best == "volatile":
                threshold_adj[i] = 0.1
            elif best == "quiet":
                threshold_adj[i] = 0.05

        return {
            "regime_name": regime_names,
            "gates_tf": gates_tf,
            "gates_mr": gates_mr,
            "gates_st": gates_st,
            "threshold_adj": threshold_adj,
        }

    # =========================================================================
    # セッション事前計算
    # =========================================================================

    def _compute_sessions(self, df, n) -> np.ndarray:
        """セッションを全バーで事前計算"""
        sessions = np.full(n, "closed", dtype=object)

        if "datetime" not in df.columns:
            return sessions

        try:
            dt_col = pd.to_datetime(df["datetime"])
            hours = dt_col.dt.hour.values
            weekdays = dt_col.dt.weekday.values  # 0=Mon, 6=Sun
        except Exception:
            return sessions

        for i in range(n):
            h = hours[i]
            wd = weekdays[i]

            if wd >= 5:  # 週末
                sessions[i] = "closed"
            elif 2 <= h < 9:
                sessions[i] = "asia"
            elif 9 <= h < 10:
                sessions[i] = "london_prep"
            elif 10 <= h < 15:
                sessions[i] = "london"
            elif 15 <= h < 19:
                sessions[i] = "overlap"
            elif 19 <= h < 24:
                sessions[i] = "ny_late"
            else:
                sessions[i] = "closed"

        return sessions

    # =========================================================================
    # 合成シグナル計算
    # =========================================================================

    def _compute_composite(
        self,
        raw_signals: Dict[str, np.ndarray],
        regimes: Dict,
        sessions: np.ndarray,
        n: int,
    ) -> tuple:
        """レジームゲーティング + コンフルエンス付き合成シグナル"""
        from session.weights import (
            ENGINE_CATEGORIES, ENGINE_KEYS,
            get_session_weights,
        )
        from session.detector import Session

        CATEGORY_THRESHOLD = 0.15
        MIN_CONFLUENCE = 2

        composite_dir = np.zeros(n, dtype=int)
        composite_score = np.zeros(n)

        session_map = {
            "asia": Session.ASIA,
            "london_prep": Session.LONDON_PREP,
            "london": Session.LONDON,
            "overlap": Session.OVERLAP,
            "ny_late": Session.NY_LATE,
            "closed": Session.CLOSED,
        }

        for i in range(60, n):
            # セッション重み
            session_name = sessions[i]
            session_enum = session_map.get(session_name, Session.CLOSED)
            session_w = get_session_weights(session_enum)

            # レジームゲート
            gate_tf = regimes["gates_tf"][i]
            gate_mr = regimes["gates_mr"][i]
            gate_st = regimes["gates_st"][i]

            gate_map = {
                "trend_follow": gate_tf,
                "mean_revert": gate_mr,
                "structural": gate_st,
            }

            # 各エンジンの最終重み付けシグナル
            weighted = {}
            for eng in ENGINE_KEYS:
                raw = raw_signals[eng][i]
                cat = ENGINE_CATEGORIES[eng]
                gate = gate_map.get(cat, 0.5)
                sw = session_w.get(eng, 0.0)
                weighted[eng] = raw * sw * gate

            # カテゴリスコア
            cat_scores = {"trend_follow": 0.0, "mean_revert": 0.0, "structural": 0.0}
            for eng, val in weighted.items():
                cat = ENGINE_CATEGORIES[eng]
                cat_scores[cat] += val

            # コンフルエンス判定
            bullish_count = sum(1 for v in cat_scores.values() if v > CATEGORY_THRESHOLD)
            bearish_count = sum(1 for v in cat_scores.values() if v < -CATEGORY_THRESHOLD)

            composite = sum(weighted.values())
            entry_threshold = 0.6 + regimes["threshold_adj"][i]

            if bullish_count >= MIN_CONFLUENCE and composite > 0:
                if composite >= entry_threshold:
                    composite_dir[i] = 1
                    composite_score[i] = composite
            elif bearish_count >= MIN_CONFLUENCE and composite < 0:
                if abs(composite) >= entry_threshold:
                    composite_dir[i] = -1
                    composite_score[i] = abs(composite)

        return composite_dir, composite_score

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def _empty_precompute(self, n) -> Dict:
        return {
            "composite_direction": np.zeros(n, dtype=int),
            "composite_score": np.zeros(n),
            "regime": np.full(n, "ranging", dtype=object),
            "session": np.full(n, "closed", dtype=object),
            "raw_signals": {k: np.zeros(n) for k in [
                "trend", "mean_rev", "breakout", "momentum_div",
                "supply_demand", "session_orb", "market_structure",
            ]},
        }
