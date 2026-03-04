"""
LLMフィルター統合管理モジュール
3つのフィルターを統合し、エントリー可否を判定
"""

import pandas as pd
from typing import Dict, List, Optional

from config_manager import get
from llm.news_sentiment import NewsSentimentFilter
from llm.event_risk import EventRiskFilter
from llm.regime_detector import RegimeDetector
from logger_setup import get_logger

logger = get_logger("llm.filter_manager")


class LLMFilterManager:
    """LLMフィルター統合管理"""

    def __init__(self):
        self.news_filter = NewsSentimentFilter()
        self.event_filter = EventRiskFilter()
        self.regime_detector = RegimeDetector()
        self._enabled = get("llm.enabled", False)

    def should_allow_entry(
        self,
        df: pd.DataFrame,
        news_list: Optional[List[str]] = None,
        direction: str = "BUY",
    ) -> dict:
        """
        全LLMフィルターを通してエントリー可否を判定
        
        Args:
            df: H1足OHLCVデータ
            news_list: 最新ニューステキストのリスト
            direction: "BUY" or "SELL"
            
        Returns:
            dict: {
                "allowed": bool,
                "reason": str,
                "event_risk": dict,
                "sentiment": dict,
                "regime": dict,
            }
        """
        result = {
            "allowed": True,
            "reason": "OK",
            "event_risk": {},
            "sentiment": {},
            "regime": {},
        }

        # 1. イベントリスクチェック（常時実行 - LLMの有無に関係なく）
        event_check = self.event_filter.check_blackout()
        result["event_risk"] = event_check

        if event_check["is_blackout"]:
            result["allowed"] = False
            result["reason"] = f"イベントブラックアウト: {event_check['event']}"
            logger.warning(f"エントリー拒否: {result['reason']}")
            return result

        if not self._enabled:
            # LLM無効時はイベントチェックのみでOK
            result["regime"] = self.regime_detector.detect(df, use_llm=False)
            return result

        # 2. ニュース感情チェック
        if news_list:
            sentiment = self.news_filter.analyze_batch(news_list)
            result["sentiment"] = sentiment

            if sentiment["should_block"]:
                # 強センチメントが方向と逆の場合はブロック
                if (direction == "BUY" and sentiment["score"] < -0.7) or \
                   (direction == "SELL" and sentiment["score"] > 0.7):
                    result["allowed"] = False
                    result["reason"] = (
                        f"強ネガセンチメント (score={sentiment['score']:.2f}) "
                        f"がエントリー方向 ({direction}) と矛盾"
                    )
                    logger.warning(f"エントリー拒否: {result['reason']}")
                    return result

        # 3. レジーム判定
        regime = self.regime_detector.detect(df, use_llm=self._enabled)
        result["regime"] = regime

        return result

    def update_events(self, events: List[dict]):
        """イベントカレンダーを更新"""
        self.event_filter.set_upcoming_events(events)

    def set_enabled(self, enabled: bool):
        """LLMフィルターの有効/無効"""
        self._enabled = enabled
        logger.info(f"LLMフィルター{'有効' if enabled else '無効'}に設定")
