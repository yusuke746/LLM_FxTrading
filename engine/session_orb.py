"""
セッションORB（オープニングレンジ・ブレイクアウト）エンジン（EUR/USD H1専用チューニング）

パラメータ:
  - ロンドン開場ORB: 10:00 サーバー時間 の1本目H1レンジ
  - NY開場ORB: 16:00 サーバー時間 の1本目H1レンジ
  - ブレイクアウト判定: レンジの高値/安値を超えた方向
  - ATR確認: スクイーズ後のORBほど信頼度UP

シグナル:
  +1.0: ORBレンジを上方ブレイク（ボリューム & ATR確認）
  -1.0: ORBレンジを下方ブレイク（ボリューム & ATR確認）
   0.0: ORBレンジ内 or 対象セッション外

戦略本質:
  ロンドン/NY開場の最初の1時間は機関の方向感が出やすい。
  そのレンジをブレイクした方向にモメンタムが続く傾向がある。

時刻基準: MT5サーバー時間 (EET/EEST)
  OHLCVのdatetime列はサーバー時間。
  London/NYのDSTとサーバーDSTが連動するため、
  セッション時間は季節を問わず固定。
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple

from logger_setup import get_logger

logger = get_logger("engine.session_orb")

# デフォルトパラメータ
ATR_PERIOD = 14
# ORBセッション時間（サーバー時間, H1足のhour）
LONDON_ORB_HOUR = 10     # ロンドン開場 10:00 サーバー時間
NY_ORB_HOUR = 16         # NY開場 16:00 サーバー時間
VOLUME_MULTIPLIER = 1.1  # ORBブレイク確認のボリューム閾値


class SessionORBEngine:
    """セッション・オープニングレンジ・ブレイクアウトエンジン"""

    def __init__(
        self,
        atr_period: int = ATR_PERIOD,
        volume_multiplier: float = VOLUME_MULTIPLIER,
    ):
        self.atr_period = atr_period
        self.volume_multiplier = volume_multiplier

    def get_signal(self, df: pd.DataFrame) -> float:
        """
        セッションORBシグナルを算出

        Args:
            df: H1足 OHLCVデータ（datetime列はMT5サーバー時間）

        Returns:
            float: -1.0 ~ +1.0 のシグナルスコア
        """
        if len(df) < self.atr_period + 10:
            return 0.0

        try:
            # datetimeの処理
            if "datetime" not in df.columns:
                return 0.0

            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            open_p = df["open"].values

            # ATR算出
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            current_atr = atr_series.iloc[-1]
            if pd.isna(current_atr):
                return 0.0

            # 現在のバーの時刻を取得（サーバー時間）
            current_bar = df.iloc[-1]
            current_hour = self._get_server_hour(current_bar["datetime"])
            if current_hour is None:
                return 0.0

            # ロンドンORBチェック（11:00-12:00 サーバー時間 = ORB直後2本）
            london_signal = self._check_orb_breakout(
                df, LONDON_ORB_HOUR, current_hour, current_atr
            )

            # NY ORBチェック（17:00-18:00 サーバー時間 = ORB直後2本）
            ny_signal = self._check_orb_breakout(
                df, NY_ORB_HOUR, current_hour, current_atr
            )

            # より強いシグナルを採用
            if abs(london_signal) >= abs(ny_signal):
                return round(london_signal, 4)
            else:
                return round(ny_signal, 4)

        except Exception as e:
            logger.error(f"セッションORB算出エラー: {e}", exc_info=True)
            return 0.0

    def _get_server_hour(self, dt) -> Optional[int]:
        """
        datetime からサーバー時間のhourを取得
        OHLCVのdatetimeは既にサーバー時間(EET/EEST)に変換済み
        """
        try:
            if hasattr(dt, "hour"):
                return dt.hour
            return None
        except Exception:
            return None

    def _check_orb_breakout(
        self,
        df: pd.DataFrame,
        orb_hour: int,
        current_jst_hour: int,
        current_atr: float,
    ) -> float:
        """
        特定セッションのORBブレイクアウトをチェック

        Args:
            df: OHLCVデータ
            orb_hour: ORB対象の時間（サーバー時間）
            current_jst_hour: 現在のサーバー時間
            current_atr: 現在のATR値

        Returns:
            float: シグナルスコア
        """
        # ORBの次の2本のみがエントリー候補
        # (orb_hour+1, orb_hour+2)
        valid_hours = [(orb_hour + 1) % 24, (orb_hour + 2) % 24]
        if current_jst_hour not in valid_hours:
            return 0.0

        # ORB足（セッション開始の最初の1本）を見つける
        orb_bar = None
        for i in range(len(df) - 1, max(len(df) - 10, 0), -1):
            bar_hour = self._get_server_hour(df.iloc[i]["datetime"])
            if bar_hour == orb_hour:
                orb_bar = df.iloc[i]
                break

        if orb_bar is None:
            return 0.0

        orb_high = orb_bar["high"]
        orb_low = orb_bar["low"]
        orb_range = orb_high - orb_low

        if orb_range <= 0:
            return 0.0

        current_close = df["close"].iloc[-1]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]

        signal = 0.0

        # 上方ブレイクアウト
        if current_close > orb_high:
            breakout_distance = (current_close - orb_high) / current_atr
            signal = min(0.4 + breakout_distance * 0.3, 0.9)

            # ORBレンジが小さい（スクイーズ）ほど信頼度UP
            if orb_range < current_atr * 0.5:
                signal += 0.1  # スクイーズ後のブレイクアウト

        # 下方ブレイクアウト
        elif current_close < orb_low:
            breakout_distance = (orb_low - current_close) / current_atr
            signal = -min(0.4 + breakout_distance * 0.3, 0.9)

            if orb_range < current_atr * 0.5:
                signal -= 0.1

        # ボリューム確認
        if abs(signal) > 0 and "tick_volume" in df.columns:
            vol = df["tick_volume"].values
            avg_vol = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
            current_vol = vol[-1]
            if avg_vol > 0 and current_vol < avg_vol * self.volume_multiplier:
                signal *= 0.7  # ボリューム不足で減衰

        return min(max(signal, -1.0), 1.0)

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """現在のORB情報を取得（ダッシュボード表示用）"""
        try:
            current_bar = df.iloc[-1]
            current_hour = self._get_server_hour(current_bar["datetime"])

            return {
                "current_server_hour": current_hour,
                "signal_score": self.get_signal(df),
            }
        except Exception:
            return {}
