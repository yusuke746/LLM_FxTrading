"""
レジーム判定フィルター
市場レジーム（トレンド中/レンジング/高ボラティリティ）を分類し、
セッション判定と組み合わせて有効なエンジンのみ起動

【期待値向上】ATRパーセンタイルランキング:
  ATRの現在値が過去100本中の何パーセンタイルかで
  ボラティリティレジームをテクニカルに事前フィルタリング
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from openai import OpenAI

from config_manager import get
from logger_setup import get_logger

logger = get_logger("llm.regime")


class MarketRegime:
    """市場レジームの種別"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


REGIME_PROMPT = """あなたはFXマーケットのレジーム分析専門家です。
以下の市場データを分析し、現在の市場レジーム（状態）を判定してください。

# レジーム種別
- "trending_up": 明確な上昇トレンド
- "trending_down": 明確な下降トレンド
- "ranging": レンジ相場（方向感なし）
- "high_volatility": 高ボラティリティ（方向不明だが大きく動いている）
- "low_volatility": 低ボラティリティ（動きが少ない）

# 出力形式（JSONのみ）
{{
  "regime": "レジーム種別",
  "confidence": 0.0〜1.0,
  "recommended_engines": ["trend", "mean_rev", "breakout"]のうち有効なもの,
  "reasoning": "1行の理由"
}}

# 市場データ（EUR/USD H1）
{market_data}
"""


class RegimeDetector:
    """市場レジーム検出器（テクニカル + LLM補助）"""

    def __init__(self):
        self.client = None
        self.model = get("llm.model", "gpt-4o")

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def detect(self, df: pd.DataFrame, use_llm: bool = False) -> dict:
        """
        市場レジームを判定
        
        Args:
            df: H1足 OHLCVデータ（100本以上推奨）
            use_llm: LLMを使用するかどうか（デフォルト: テクニカルのみ）
            
        Returns:
            dict: {"regime": str, "confidence": float, "recommended_engines": list}
        """
        # テクニカルベースのレジーム判定（常に実行）
        tech_regime = self._detect_technical(df)

        if use_llm and get("llm.enabled", False):
            llm_regime = self._detect_with_llm(df)
            # テクニカルとLLMが一致する場合は信頼度UP
            if llm_regime["regime"] == tech_regime["regime"]:
                tech_regime["confidence"] = min(tech_regime["confidence"] + 0.15, 1.0)
            else:
                # 不一致の場合はテクニカルを優先（LLM優先禁止原則）
                logger.info(
                    f"レジーム判定不一致: テクニカル={tech_regime['regime']}, "
                    f"LLM={llm_regime['regime']} → テクニカル優先"
                )

        return tech_regime

    def _detect_technical(self, df: pd.DataFrame) -> dict:
        """
        テクニカル指標ベースのレジーム判定
        ADX + ATRパーセンタイル + BBバンド幅 を複合判定
        """
        if len(df) < 100:
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0,
                "recommended_engines": ["trend", "mean_rev", "breakout"],
                "volatility_percentile": 0.5,
            }

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # ADX
            adx_result = ta.adx(high, low, close, length=14)
            current_adx = adx_result["ADX_14"].iloc[-1]
            plus_di = adx_result["DMP_14"].iloc[-1]
            minus_di = adx_result["DMN_14"].iloc[-1]

            # ATR & ATRパーセンタイル
            atr = ta.atr(high, low, close, length=14)
            current_atr = atr.iloc[-1]
            atr_percentile = self._calc_atr_percentile(atr)

            # BB バンド幅
            bb = ta.bbands(close, length=20, std=2.0)
            bb_width = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / bb["BBM_20_2.0"]
            current_bb_width = bb_width.iloc[-1]
            avg_bb_width = bb_width.iloc[-50:].mean()

            # EMA方向性
            ema_20 = ta.ema(close, length=20)
            ema_50 = ta.ema(close, length=50)

            if pd.isna(current_adx) or pd.isna(current_atr):
                return {
                    "regime": MarketRegime.UNKNOWN,
                    "confidence": 0.0,
                    "recommended_engines": ["trend", "mean_rev", "breakout"],
                    "volatility_percentile": 0.5,
                }

            # レジーム判定ロジック
            regime = MarketRegime.RANGING
            confidence = 0.5
            recommended = ["trend", "mean_rev", "breakout"]

            # 高ボラティリティ判定
            if atr_percentile > 0.85:
                regime = MarketRegime.HIGH_VOLATILITY
                confidence = 0.7 + (atr_percentile - 0.85) * 2
                recommended = ["breakout", "trend"]
            # 低ボラティリティ判定
            elif atr_percentile < 0.15:
                regime = MarketRegime.LOW_VOLATILITY
                confidence = 0.7 + (0.15 - atr_percentile) * 2
                recommended = ["mean_rev"]
            # トレンド判定（ADX高 + EMA方向一致）
            elif current_adx >= 25:
                if plus_di > minus_di and ema_20.iloc[-1] > ema_50.iloc[-1]:
                    regime = MarketRegime.TRENDING_UP
                    confidence = min(current_adx / 50, 1.0)
                    recommended = ["trend", "breakout"]
                elif minus_di > plus_di and ema_20.iloc[-1] < ema_50.iloc[-1]:
                    regime = MarketRegime.TRENDING_DOWN
                    confidence = min(current_adx / 50, 1.0)
                    recommended = ["trend", "breakout"]
                else:
                    regime = MarketRegime.RANGING
                    confidence = 0.5
            # レンジ判定（ADX低 + BBバンド幅狭）
            elif current_adx < 20 and current_bb_width < avg_bb_width:
                regime = MarketRegime.RANGING
                confidence = 0.6 + (20 - current_adx) / 40
                recommended = ["mean_rev"]

            result = {
                "regime": regime,
                "confidence": round(min(confidence, 1.0), 3),
                "recommended_engines": recommended,
                "volatility_percentile": round(atr_percentile, 3),
                "adx": round(current_adx, 2),
                "atr": round(current_atr, 6),
                "bb_width": round(current_bb_width, 4) if not pd.isna(current_bb_width) else None,
            }

            logger.info(f"レジーム判定: {regime} (信頼度: {confidence:.2f}, ATR%ile: {atr_percentile:.2f})")
            return result

        except Exception as e:
            logger.error(f"テクニカルレジーム判定エラー: {e}")
            return {
                "regime": MarketRegime.UNKNOWN,
                "confidence": 0.0,
                "recommended_engines": ["trend", "mean_rev", "breakout"],
                "volatility_percentile": 0.5,
            }

    def _calc_atr_percentile(self, atr: pd.Series, lookback: int = 100) -> float:
        """
        ATRのパーセンタイルランキングを算出
        現在のATRが過去N本中の何パーセンタイルかを返す
        """
        atr_clean = atr.dropna()
        if len(atr_clean) < lookback:
            lookback = len(atr_clean)

        if lookback < 10:
            return 0.5

        recent_atr = atr_clean.iloc[-lookback:]
        current_atr = atr_clean.iloc[-1]

        rank = (recent_atr < current_atr).sum()
        percentile = rank / len(recent_atr)

        return percentile

    def _detect_with_llm(self, df: pd.DataFrame) -> dict:
        """LLM補助によるレジーム判定"""
        try:
            # 直近20本のサマリーデータを作成
            recent = df.tail(20)
            market_data = (
                f"直近20本H1足:\n"
                f"  始値範囲: {recent['open'].min():.5f} ~ {recent['open'].max():.5f}\n"
                f"  高値範囲: {recent['high'].min():.5f} ~ {recent['high'].max():.5f}\n"
                f"  安値範囲: {recent['low'].min():.5f} ~ {recent['low'].max():.5f}\n"
                f"  終値範囲: {recent['close'].min():.5f} ~ {recent['close'].max():.5f}\n"
                f"  方向: {('上昇' if recent['close'].iloc[-1] > recent['close'].iloc[0] else '下降')}\n"
                f"  変動幅: {(recent['close'].iloc[-1] - recent['close'].iloc[0]):.5f}\n"
            )

            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "マーケットレジーム分析の専門家。JSONのみで回答。"},
                    {"role": "user", "content": REGIME_PROMPT.format(market_data=market_data)},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "regime": result.get("regime", MarketRegime.UNKNOWN),
                "confidence": float(result.get("confidence", 0.5)),
                "recommended_engines": result.get("recommended_engines", []),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error(f"LLMレジーム判定エラー: {e}")
            return {"regime": MarketRegime.UNKNOWN, "confidence": 0.0, "recommended_engines": []}
