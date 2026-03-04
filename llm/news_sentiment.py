"""
ニュース感情分析フィルター
OpenAI API (GPT-4o) を使用してニュースの感情スコアを算出

- スコア: -1.0（極めてネガティブ）〜 +1.0（極めてポジティブ）
- 強センチメント (|score| > 0.7) の場合のみ操作を制限
- EUR/USD に影響するニュースを対象
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from config_manager import get
from logger_setup import get_logger

logger = get_logger("llm.news_sentiment")

SENTIMENT_PROMPT = """あなたはFX市場のニュース感情分析の専門家です。
以下のニュースを分析し、EUR/USD（ユーロ/ドル）に対する影響を評価してください。

# 評価基準
- EUR（ユーロ）にポジティブ or USD（ドル）にネガティブ → プラススコア（EURが上がる方向）
- EUR（ユーロ）にネガティブ or USD（ドル）にポジティブ → マイナススコア（EURが下がる方向）
- 影響なし or 中立 → 0に近いスコア

# 出力形式（JSONのみ）
{{
  "score": -1.0〜1.0の数値,
  "confidence": 0.0〜1.0の信頼度,
  "summary": "1行の要約",
  "impact_duration": "short"/"medium"/"long"
}}

# ニュース
{news_text}
"""


class NewsSentimentFilter:
    """ニュース感情分析フィルター"""

    def __init__(self):
        self.client = None
        self.model = get("llm.model", "gpt-4o")
        self.threshold = get("llm.sentiment_threshold", 0.7)
        self._cache: Dict[str, dict] = {}

    def _get_client(self) -> OpenAI:
        """OpenAIクライアントの遅延初期化（config.yaml の api_key を使用）"""
        if self.client is None:
            api_key = get("llm.api_key", "")
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                # 環境変数 OPENAI_API_KEY にフォールバック
                self.client = OpenAI()
        return self.client

    def analyze(self, news_text: str) -> dict:
        """
        ニューステキストの感情分析
        
        Args:
            news_text: ニューステキスト
            
        Returns:
            dict: {"score": float, "confidence": float, "summary": str, "should_block": bool}
        """
        if not get("llm.enabled", False):
            return self._neutral_result()

        # キャッシュチェック（同一ニュースの重複分析を防止）
        cache_key = news_text[:100]
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたはFXニュース分析の専門家です。JSONのみで回答してください。"},
                    {"role": "user", "content": SENTIMENT_PROMPT.format(news_text=news_text)},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            score = float(result.get("score", 0.0))
            confidence = float(result.get("confidence", 0.5))

            output = {
                "score": max(-1.0, min(1.0, score)),
                "confidence": max(0.0, min(1.0, confidence)),
                "summary": result.get("summary", ""),
                "impact_duration": result.get("impact_duration", "short"),
                "should_block": abs(score) >= self.threshold and confidence >= 0.6,
                "raw_response": result,
            }

            self._cache[cache_key] = output
            logger.info(f"ニュース感情スコア: {score:.2f} (信頼度: {confidence:.2f})")
            return output

        except Exception as e:
            logger.error(f"ニュース感情分析エラー: {e}")
            return self._neutral_result()

    def analyze_batch(self, news_list: List[str]) -> dict:
        """
        複数ニュースの一括分析。加重平均スコアを返す
        
        Args:
            news_list: ニューステキストのリスト
            
        Returns:
            dict: 加重平均スコアと個別結果
        """
        if not news_list:
            return self._neutral_result()

        results = [self.analyze(news) for news in news_list]
        
        # 信頼度で重み付けした平均スコア
        total_weight = sum(r["confidence"] for r in results)
        if total_weight == 0:
            avg_score = 0.0
        else:
            avg_score = sum(r["score"] * r["confidence"] for r in results) / total_weight

        return {
            "score": round(avg_score, 3),
            "confidence": round(total_weight / len(results), 3),
            "should_block": abs(avg_score) >= self.threshold,
            "individual_results": results,
            "count": len(results),
        }

    def _neutral_result(self) -> dict:
        """中立結果（LLM無効時 or エラー時）"""
        return {
            "score": 0.0,
            "confidence": 0.0,
            "summary": "",
            "should_block": False,
        }

    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
