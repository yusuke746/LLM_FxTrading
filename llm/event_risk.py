"""
イベントリスク判定フィルター
ECB会合・NFP・CPI等の重要経済指標を検知し、リスク高時は新規エントリーを停止

影響度別制御:
  ★★★★★ ECB金融政策, 米雇用統計(NFP): 発表前30分〜後30分エントリー停止
  ★★★★  米CPI・PCE: 発表前15分〜後15分エントリー停止
  ★★★★  FED議長発言: LLMセンチ判定→強センチ時は停止
  ★★★   米GDP・ISM, ユーロ圏CPI: LLMフィルターで判定
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz
from openai import OpenAI

from config_manager import get
from logger_setup import get_logger

logger = get_logger("llm.event_risk")

JST = pytz.timezone("Asia/Tokyo")

# 重要経済指標のデフォルト設定
HIGH_IMPACT_EVENTS = {
    "ECB_RATE_DECISION": {
        "name": "ECB金融政策発表",
        "impact": 5,
        "blackout_before_min": 30,
        "blackout_after_min": 30,
    },
    "NFP": {
        "name": "米雇用統計(NFP)",
        "impact": 5,
        "blackout_before_min": 30,
        "blackout_after_min": 30,
    },
    "US_CPI": {
        "name": "米CPI",
        "impact": 4,
        "blackout_before_min": 15,
        "blackout_after_min": 15,
    },
    "US_PCE": {
        "name": "米PCEデフレーター",
        "impact": 4,
        "blackout_before_min": 15,
        "blackout_after_min": 15,
    },
    "FED_CHAIR_SPEECH": {
        "name": "FED議長発言",
        "impact": 4,
        "blackout_before_min": 15,
        "blackout_after_min": 15,
    },
    "US_GDP": {
        "name": "米GDP",
        "impact": 3,
        "blackout_before_min": 10,
        "blackout_after_min": 10,
    },
    "US_ISM": {
        "name": "米ISM",
        "impact": 3,
        "blackout_before_min": 10,
        "blackout_after_min": 10,
    },
    "EUROZONE_CPI": {
        "name": "ユーロ圏CPI",
        "impact": 3,
        "blackout_before_min": 10,
        "blackout_after_min": 10,
    },
    "FOMC_MINUTES": {
        "name": "FOMC議事録",
        "impact": 4,
        "blackout_before_min": 15,
        "blackout_after_min": 15,
    },
}

EVENT_RISK_PROMPT = """あなたはFXマーケットのイベントリスク分析専門家です。
以下の経済イベント情報を分析し、EUR/USDトレードへのリスクを評価してください。

# 評価基準
- リスクレベル: "high" / "medium" / "low"
- エントリー推奨: true（エントリーOK）/ false（見送り推奨）
- ボラティリティ予測: 予想される値動きの大きさ（pips単位）

# 出力形式（JSONのみ）
{{
  "risk_level": "high"/"medium"/"low",
  "should_trade": true/false,
  "expected_volatility_pips": 数値,
  "reasoning": "1行の理由"
}}

# イベント情報
{event_info}
"""


class EventRiskFilter:
    """経済イベントリスク判定フィルター"""

    def __init__(self):
        self.client = None
        self.model = get("llm.model", "gpt-4o")
        self.upcoming_events: List[dict] = []

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def set_upcoming_events(self, events: List[dict]):
        """
        今後のイベントリストを設定
        
        Args:
            events: [{"type": "ECB_RATE_DECISION", "datetime": "2026-03-05T20:45:00+09:00", ...}]
        """
        self.upcoming_events = events
        logger.info(f"イベントカレンダー更新: {len(events)}件")

    def check_blackout(self, dt: Optional[datetime] = None) -> dict:
        """
        現在時刻がブラックアウト期間かチェック
        
        Args:
            dt: チェック対象時刻（Noneの場合は現在時刻）
            
        Returns:
            dict: {"is_blackout": bool, "event": str, "minutes_to_event": int}
        """
        if dt is None:
            dt = datetime.now(JST)
        elif dt.tzinfo is None:
            dt = JST.localize(dt)

        for event in self.upcoming_events:
            event_type = event.get("type", "")
            event_config = HIGH_IMPACT_EVENTS.get(event_type, {})

            if not event_config:
                continue

            event_dt_str = event.get("datetime", "")
            try:
                event_dt = datetime.fromisoformat(event_dt_str)
                if event_dt.tzinfo is None:
                    event_dt = JST.localize(event_dt)
            except (ValueError, TypeError):
                continue

            before_min = event_config.get("blackout_before_min", 30)
            after_min = event_config.get("blackout_after_min", 30)

            blackout_start = event_dt - timedelta(minutes=before_min)
            blackout_end = event_dt + timedelta(minutes=after_min)

            if blackout_start <= dt <= blackout_end:
                minutes_to = int((event_dt - dt).total_seconds() / 60)
                logger.warning(
                    f"ブラックアウト期間: {event_config['name']} "
                    f"(イベントまで{minutes_to}分)"
                )
                return {
                    "is_blackout": True,
                    "event": event_config["name"],
                    "event_type": event_type,
                    "impact": event_config["impact"],
                    "minutes_to_event": minutes_to,
                    "blackout_end": blackout_end.isoformat(),
                }

        return {
            "is_blackout": False,
            "event": None,
            "minutes_to_event": None,
        }

    def assess_risk_with_llm(self, event_info: str) -> dict:
        """
        LLMでイベントリスクを詳細評価（★★★以下のイベント用）
        
        Args:
            event_info: イベントの詳細情報
            
        Returns:
            dict: リスク評価結果
        """
        if not get("llm.enabled", False):
            return {"risk_level": "low", "should_trade": True, "reasoning": "LLM disabled"}

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "FXイベントリスク分析の専門家として、JSONのみで回答してください。"},
                    {"role": "user", "content": EVENT_RISK_PROMPT.format(event_info=event_info)},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"LLMイベントリスク評価: {result.get('risk_level')} - {result.get('reasoning', '')}")
            return result

        except Exception as e:
            logger.error(f"LLMイベントリスク評価エラー: {e}")
            return {"risk_level": "medium", "should_trade": True, "reasoning": f"Error: {e}"}

    def get_next_events(self, hours: int = 24) -> List[dict]:
        """今後N時間以内のイベントを取得"""
        now = datetime.now(JST)
        cutoff = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self.upcoming_events:
            try:
                event_dt = datetime.fromisoformat(event.get("datetime", ""))
                if event_dt.tzinfo is None:
                    event_dt = JST.localize(event_dt)
                if now <= event_dt <= cutoff:
                    event_config = HIGH_IMPACT_EVENTS.get(event.get("type", ""), {})
                    upcoming.append({
                        **event,
                        "name": event_config.get("name", event.get("type", "Unknown")),
                        "impact": event_config.get("impact", 1),
                    })
            except (ValueError, TypeError):
                continue

        return sorted(upcoming, key=lambda x: x.get("datetime", ""))
