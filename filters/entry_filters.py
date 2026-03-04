"""
エントリーフィルターモジュール
期待値向上のための追加フィルター群

- ボラティリティフィルター: 超低ボラ時のスプレッド負けを防止
- スプレッド監視: 異常スプレッド時のエントリー遅延
- 曜日・時間帯フィルター: 過去データで負けやすい時間帯を回避
- アダプティブ閾値: 直近成績に応じてエントリー閾値を動的調整

設計方針:
  - エントリー数を激減させない（緩やかなフィルターのみ）
  - 明らかに不利な条件のみブロック
  - 全フィルターに bypass（無効化）オプション付き
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz

from config_manager import get
from database import get_trade_log
from logger_setup import get_logger

logger = get_logger("filters.entry")

JST = pytz.timezone("Asia/Tokyo")


# =====================================================
# 1. ボラティリティフィルター
# =====================================================

class VolatilityFilter:
    """
    ATRベースのボラティリティフィルター
    
    超低ボラティリティ環境ではスプレッドコストに対して
    利幅が取れないため、エントリーを見送る。
    
    ブロック条件: ATRが過去N本のP%タイル未満
    デフォルト: 過去100本の10パーセンタイル未満 → ブロック
    → 統計的に約10%の足のみブロック = エントリー数への影響は軽微
    """

    def __init__(self):
        self.lookback = get("filters.volatility_lookback", 100)
        self.min_percentile = get("filters.volatility_min_percentile", 10)
        self.enabled = get("filters.volatility_enabled", True)

    def check(self, df: pd.DataFrame, current_atr: float) -> dict:
        """
        ボラティリティが十分かチェック
        
        Args:
            df: H1足OHLCVデータ
            current_atr: 現在のATR(14)値
            
        Returns:
            dict: {"allowed": bool, "reason": str, "atr_percentile": float}
        """
        if not self.enabled:
            return {"allowed": True, "reason": "filter_disabled", "atr_percentile": 50.0}

        try:
            import pandas_ta as ta
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            
            if atr_series is None or len(atr_series) < self.lookback:
                return {"allowed": True, "reason": "data_insufficient", "atr_percentile": 50.0}

            recent_atr = atr_series.dropna().tail(self.lookback)
            if len(recent_atr) < 20:
                return {"allowed": True, "reason": "data_insufficient", "atr_percentile": 50.0}

            percentile = float(np.percentile(recent_atr, self.min_percentile))
            current_pct = float((recent_atr < current_atr).sum() / len(recent_atr) * 100)

            if current_atr < percentile:
                reason = (
                    f"低ボラティリティ: ATR={current_atr:.5f} < "
                    f"{self.min_percentile}パーセンタイル({percentile:.5f})"
                )
                logger.info(f"ボラティリティフィルター: {reason}")
                return {"allowed": False, "reason": reason, "atr_percentile": current_pct}

            return {"allowed": True, "reason": "OK", "atr_percentile": current_pct}

        except Exception as e:
            logger.error(f"ボラティリティフィルターエラー: {e}")
            return {"allowed": True, "reason": f"error: {e}", "atr_percentile": 50.0}


# =====================================================
# 2. スプレッド監視フィルター
# =====================================================

class SpreadFilter:
    """
    スプレッド異常監視フィルター
    
    流動性低下時（指標前後・市場開閉時等）にスプレッドが
    通常の2倍以上に拡大した場合、エントリーを次足に見送る。
    
    ブロック条件: 現在スプレッド > 通常スプレッド × max_ratio
    デフォルト: 通常の2.5倍超 → ブロック
    → 正常時はほぼブロックされない（年間数%の足のみ）
    """

    def __init__(self):
        self.max_ratio = get("filters.spread_max_ratio", 2.5)
        self.normal_spread_pips = get("filters.normal_spread_pips", 1.6)
        self.enabled = get("filters.spread_enabled", True)

    def check(self, current_spread: float, symbol: str = "EURUSD") -> dict:
        """
        スプレッドが許容範囲内かチェック
        
        Args:
            current_spread: 現在のスプレッド（price差分）
            symbol: 通貨ペア
            
        Returns:
            dict: {"allowed": bool, "reason": str, "spread_pips": float, "ratio": float}
        """
        if not self.enabled:
            return {"allowed": True, "reason": "filter_disabled", "spread_pips": 0, "ratio": 1.0}

        try:
            # EURUSDの場合 pip = 0.0001
            pip_size = 0.0001 if "JPY" not in symbol else 0.01
            spread_pips = current_spread / pip_size

            threshold_pips = self.normal_spread_pips * self.max_ratio
            ratio = spread_pips / self.normal_spread_pips if self.normal_spread_pips > 0 else 1.0

            if spread_pips > threshold_pips:
                reason = (
                    f"スプレッド異常: {spread_pips:.1f}pips > "
                    f"閾値{threshold_pips:.1f}pips (通常の{ratio:.1f}倍)"
                )
                logger.info(f"スプレッドフィルター: {reason}")
                return {
                    "allowed": False,
                    "reason": reason,
                    "spread_pips": spread_pips,
                    "ratio": ratio,
                }

            return {
                "allowed": True,
                "reason": "OK",
                "spread_pips": spread_pips,
                "ratio": ratio,
            }

        except Exception as e:
            logger.error(f"スプレッドフィルターエラー: {e}")
            return {"allowed": True, "reason": f"error: {e}", "spread_pips": 0, "ratio": 1.0}


# =====================================================
# 3. 曜日・時間帯パフォーマンスフィルター
# =====================================================

class TimePerformanceFilter:
    """
    曜日×時間帯の過去パフォーマンスに基づくフィルター
    
    過去のトレードDBから、特定の曜日・時間帯の勝率を算出し、
    「常に負けている」スロットのみをブロック対象とする。
    
    ブロック条件:
      - 該当スロットで最低N回以上のトレード実績がある（min_trades）
      - 勝率が min_win_rate 未満
    
    デフォルト: 最低10回以上 & 勝率30%未満 → ブロック
    → 非常に緩い条件なので、明確に負けるスロットのみ排除
    """

    def __init__(self):
        self.min_trades = get("filters.time_min_trades", 10)
        self.min_win_rate = get("filters.time_min_win_rate", 0.30)
        self.enabled = get("filters.time_performance_enabled", True)
        self._cache: Dict[str, dict] = {}
        self._cache_time: Optional[datetime] = None

    def check(self, current_time: Optional[datetime] = None) -> dict:
        """
        現在の曜日×時間帯がトレードに適しているかチェック
        
        Args:
            current_time: 判定時刻（Noneの場合は現在時刻JST）
            
        Returns:
            dict: {"allowed": bool, "reason": str, "win_rate": float, "trade_count": int}
        """
        if not self.enabled:
            return {"allowed": True, "reason": "filter_disabled", "win_rate": 0.5, "trade_count": 0}

        try:
            if current_time is None:
                current_time = datetime.now(JST)
            elif current_time.tzinfo is None:
                current_time = JST.localize(current_time)

            day_of_week = current_time.weekday()  # 0=月, 4=金
            hour = current_time.hour
            slot_key = f"{day_of_week}_{hour}"

            # キャッシュ更新（1日1回）
            stats = self._get_performance_stats()
            
            if slot_key not in stats:
                # データなし → ブロックしない
                return {"allowed": True, "reason": "no_data", "win_rate": 0.5, "trade_count": 0}

            slot = stats[slot_key]
            trade_count = slot["total"]
            win_rate = slot["win_rate"]

            if trade_count >= self.min_trades and win_rate < self.min_win_rate:
                day_names = ["月", "火", "水", "木", "金", "土", "日"]
                reason = (
                    f"低パフォーマンス時間帯: {day_names[day_of_week]}曜{hour}時 "
                    f"(勝率{win_rate*100:.0f}% / {trade_count}回)"
                )
                logger.info(f"時間帯フィルター: {reason}")
                return {
                    "allowed": False,
                    "reason": reason,
                    "win_rate": win_rate,
                    "trade_count": trade_count,
                }

            return {
                "allowed": True,
                "reason": "OK",
                "win_rate": win_rate,
                "trade_count": trade_count,
            }

        except Exception as e:
            logger.error(f"時間帯フィルターエラー: {e}")
            return {"allowed": True, "reason": f"error: {e}", "win_rate": 0.5, "trade_count": 0}

    def _get_performance_stats(self) -> Dict[str, dict]:
        """過去トレードから曜日×時間帯別の成績を集計"""
        # 1日1回キャッシュ更新
        now = datetime.now(JST)
        if self._cache_time and (now - self._cache_time).total_seconds() < 86400:
            return self._cache

        try:
            trades = get_trade_log(weeks=12)  # 過去12週分
            stats: Dict[str, dict] = {}

            for trade in trades:
                if trade.get("status") != "closed" or trade.get("pnl") is None:
                    continue

                entry_time_str = trade.get("entry_time", "")
                if not entry_time_str:
                    continue

                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                    if entry_time.tzinfo is None:
                        entry_time = JST.localize(entry_time)
                except (ValueError, TypeError):
                    continue

                day = entry_time.weekday()
                hour = entry_time.hour
                slot_key = f"{day}_{hour}"

                if slot_key not in stats:
                    stats[slot_key] = {"wins": 0, "total": 0, "pnl_sum": 0.0}

                stats[slot_key]["total"] += 1
                if trade["pnl"] > 0:
                    stats[slot_key]["wins"] += 1
                stats[slot_key]["pnl_sum"] += trade["pnl"]

            # 勝率を計算
            for key in stats:
                total = stats[key]["total"]
                stats[key]["win_rate"] = stats[key]["wins"] / total if total > 0 else 0.5

            self._cache = stats
            self._cache_time = now
            logger.info(f"時間帯パフォーマンスキャッシュ更新: {len(stats)}スロット")
            return stats

        except Exception as e:
            logger.error(f"パフォーマンス集計エラー: {e}")
            return self._cache or {}


# =====================================================
# 4. アダプティブ信頼度閾値
# =====================================================

class AdaptiveThreshold:
    """
    直近トレード成績に応じてエントリー閾値を動的調整
    
    - 好調期（勝率高い）→ 閾値を少し下げ、チャンスを取りに行く
    - 不調期（勝率低い）→ 閾値を少し上げ、自信のあるシグナルのみ
    
    調整範囲: base ± max_adjustment（デフォルト ±0.1）
    → 基準閾値0.6の場合、0.5～0.7の範囲で変動
    → エントリー数の大幅な減少は発生しない
    """

    def __init__(self):
        self.base_threshold = get("session.entry_threshold", 0.6)
        self.lookback_trades = get("filters.adaptive_lookback_trades", 20)
        self.max_adjustment = get("filters.adaptive_max_adjustment", 0.1)
        self.target_win_rate = get("filters.adaptive_target_win_rate", 0.50)
        self.enabled = get("filters.adaptive_enabled", True)

    def get_adjusted_threshold(self) -> Tuple[float, dict]:
        """
        直近成績に基づいて調整済みエントリー閾値を返す
        
        Returns:
            tuple: (adjusted_threshold, details_dict)
        """
        if not self.enabled:
            return self.base_threshold, {"adjustment": 0, "reason": "disabled"}

        try:
            trades = get_trade_log(weeks=4)
            closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]

            if len(closed) < 5:
                # データ不足 → 基準値をそのまま使用
                return self.base_threshold, {
                    "adjustment": 0,
                    "reason": "insufficient_data",
                    "trade_count": len(closed),
                }

            recent = closed[:self.lookback_trades]
            wins = sum(1 for t in recent if t["pnl"] > 0)
            win_rate = wins / len(recent)

            # 勝率と目標の差分に比例して閾値を調整
            # 勝率 > 目標 → adjustment < 0 → 閾値を下げる（積極的に）
            # 勝率 < 目標 → adjustment > 0 → 閾値を上げる（慎重に）
            deviation = self.target_win_rate - win_rate
            adjustment = deviation * self.max_adjustment * 2  # スケーリング

            # 上限下限クランプ
            adjustment = max(-self.max_adjustment, min(self.max_adjustment, adjustment))
            adjusted = self.base_threshold + adjustment

            # 極端な値を防止（最低0.3、最大0.9）
            adjusted = max(0.3, min(0.9, adjusted))

            details = {
                "adjustment": round(adjustment, 4),
                "base_threshold": self.base_threshold,
                "adjusted_threshold": round(adjusted, 4),
                "recent_win_rate": round(win_rate, 3),
                "trade_count": len(recent),
                "reason": "adaptive",
            }

            if abs(adjustment) > 0.01:
                logger.info(
                    f"アダプティブ閾値: {self.base_threshold:.2f} → {adjusted:.2f} "
                    f"(勝率{win_rate*100:.0f}%, 直近{len(recent)}件)"
                )

            return adjusted, details

        except Exception as e:
            logger.error(f"アダプティブ閾値エラー: {e}")
            return self.base_threshold, {"adjustment": 0, "reason": f"error: {e}"}


# =====================================================
# 統合フィルターマネージャー
# =====================================================

class EntryFilterManager:
    """
    全追加フィルターを統合管理
    main.py から一括チェック用
    """

    def __init__(self):
        self.volatility_filter = VolatilityFilter()
        self.spread_filter = SpreadFilter()
        self.time_filter = TimePerformanceFilter()
        self.adaptive_threshold = AdaptiveThreshold()

    def pre_entry_check(
        self,
        df: pd.DataFrame,
        current_atr: float,
        current_spread: float,
        symbol: str = "EURUSD",
    ) -> dict:
        """
        エントリー前の追加フィルター一括チェック
        
        Returns:
            dict: {
                "allowed": bool,
                "reason": str,
                "volatility": dict,
                "spread": dict,
                "time_performance": dict,
            }
        """
        result = {
            "allowed": True,
            "reason": "OK",
            "volatility": {},
            "spread": {},
            "time_performance": {},
        }

        # 1. ボラティリティチェック
        vol_check = self.volatility_filter.check(df, current_atr)
        result["volatility"] = vol_check
        if not vol_check["allowed"]:
            result["allowed"] = False
            result["reason"] = vol_check["reason"]
            return result

        # 2. スプレッドチェック
        spread_check = self.spread_filter.check(current_spread, symbol)
        result["spread"] = spread_check
        if not spread_check["allowed"]:
            result["allowed"] = False
            result["reason"] = spread_check["reason"]
            return result

        # 3. 時間帯パフォーマンスチェック
        time_check = self.time_filter.check()
        result["time_performance"] = time_check
        if not time_check["allowed"]:
            result["allowed"] = False
            result["reason"] = time_check["reason"]
            return result

        return result

    def get_adjusted_threshold(self) -> Tuple[float, dict]:
        """アダプティブ閾値を取得"""
        return self.adaptive_threshold.get_adjusted_threshold()
