"""
MT5発注・データ取得モジュール
XMTrading × MetaTrader5 Python API

- 成行注文 + SL/TP一括発注
- リトライエラーハンドリング（最大3回）
- OHLCVデータ取得（H1足リアルタイム＆履歴）
- ポジション管理（取得・クローズ・変更）

【期待値向上】スリッページ保護:
  最大許容スリッページを設定し、不利な約定を防止
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from config_manager import get
from logger_setup import get_logger

logger = get_logger("execution.mt5")

JST = pytz.timezone("Asia/Tokyo")
UTC = pytz.UTC

# MT5定数（MetaTrader5インポート前のフォールバック）
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 2
TRADE_ACTION_REMOVE = 3
ORDER_FILLING_IOC = 1
ORDER_FILLING_FOK = 2
ORDER_TIME_GTC = 0

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 秒
MAX_SLIPPAGE_POINTS = 30  # 最大許容スリッページ（ポイント）


class MT5Executor:
    """MetaTrader5 API実行モジュール"""

    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.mt5 = None
        self._initialized = False

    def initialize(self) -> bool:
        """MT5を初期化して接続（config.yamlの認証情報を使用）"""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5

            # config.yaml から MT5 認証情報を取得
            mt5_login = get("meta.mt5_login", 0)
            mt5_password = get("meta.mt5_password", "")
            mt5_server = get("meta.mt5_server", "")

            if mt5_login and mt5_password and mt5_server:
                if not mt5.initialize(
                    login=int(mt5_login),
                    password=str(mt5_password),
                    server=str(mt5_server),
                ):
                    logger.error(f"MT5初期化失敗(認証付き): {mt5.last_error()}")
                    return False
            else:
                # 認証情報未設定の場合、既にログイン済みのMT5に接続
                if not mt5.initialize():
                    logger.error(f"MT5初期化失敗: {mt5.last_error()}")
                    return False

            # シンボル情報を確認
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"シンボル {self.symbol} が見つかりません")
                mt5.shutdown()
                return False

            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"シンボル {self.symbol} の選択に失敗")
                    mt5.shutdown()
                    return False

            account = mt5.account_info()
            if account:
                logger.info(
                    f"MT5接続成功: アカウント={account.login} "
                    f"サーバー={account.server} 残高={account.balance}"
                )

            self._initialized = True
            return True

        except ImportError:
            logger.error("MetaTrader5パッケージが未インストール")
            return False
        except Exception as e:
            logger.error(f"MT5初期化例外: {e}")
            return False

    def shutdown(self):
        """MT5をシャットダウン"""
        if self.mt5 and self._initialized:
            self.mt5.shutdown()
            self._initialized = False
            logger.info("MT5シャットダウン完了")

    def get_ohlcv(
        self,
        timeframe: str = "H1",
        bars: int = 200,
        start_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        OHLCV データを取得
        
        Args:
            timeframe: タイムフレーム（"H1", "H4", "D1"等）
            bars: 取得本数
            start_time: 開始時刻（Noneの場合は直近N本）
            
        Returns:
            pd.DataFrame: OHLCV + tick_volume データ
        """
        if not self._check_initialized():
            return None

        try:
            tf_map = {
                "M1": self.mt5.TIMEFRAME_M1,
                "M5": self.mt5.TIMEFRAME_M5,
                "M15": self.mt5.TIMEFRAME_M15,
                "M30": self.mt5.TIMEFRAME_M30,
                "H1": self.mt5.TIMEFRAME_H1,
                "H4": self.mt5.TIMEFRAME_H4,
                "D1": self.mt5.TIMEFRAME_D1,
                "W1": self.mt5.TIMEFRAME_W1,
            }

            mt5_tf = tf_map.get(timeframe.upper())
            if mt5_tf is None:
                logger.error(f"無効なタイムフレーム: {timeframe}")
                return None

            if start_time:
                rates = self.mt5.copy_rates_from(self.symbol, mt5_tf, start_time, bars)
            else:
                rates = self.mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, bars)

            if rates is None or len(rates) == 0:
                logger.warning(f"OHLCVデータ取得失敗: {self.mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={
                "time": "datetime",
                "real_volume": "volume",
            })

            # H1足ならtick_volumeも含める
            return df[["datetime", "open", "high", "low", "close", "tick_volume"]].copy()

        except Exception as e:
            logger.error(f"OHLCVデータ取得例外: {e}")
            return None

    def get_current_price(self) -> Optional[Dict[str, float]]:
        """
        現在のBid/Ask価格を取得
        
        Returns:
            dict: {"bid": float, "ask": float, "spread": float}
        """
        if not self._check_initialized():
            return None

        try:
            tick = self.mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None

            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": round(tick.ask - tick.bid, 5),
                "time": datetime.fromtimestamp(tick.time, tz=UTC),
            }
        except Exception as e:
            logger.error(f"価格取得例外: {e}")
            return None

    def execute_order(
        self,
        direction: str,
        lot_size: float,
        sl: float,
        tp: float,
        comment: str = "fx_bot",
    ) -> Optional[dict]:
        """
        成行注文を発注（SL/TP一括）
        
        Args:
            direction: "BUY" or "SELL"
            lot_size: ロットサイズ
            sl: ストップロス価格
            tp: テイクプロフィット価格
            comment: 注文コメント
            
        Returns:
            dict: 約定結果（ticket, price, etc.）
        """
        if not self._check_initialized():
            return None

        price_info = self.get_current_price()
        if price_info is None:
            logger.error("価格取得失敗 → 発注断念")
            return None

        if direction == "BUY":
            order_type = self.mt5.ORDER_TYPE_BUY
            price = price_info["ask"]
        else:
            order_type = self.mt5.ORDER_TYPE_SELL
            price = price_info["bid"]

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": MAX_SLIPPAGE_POINTS,
            "magic": 202603,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        # リトライロジック
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self.mt5.order_send(request)

                if result is None:
                    logger.error(f"発注失敗 (attempt {attempt}): result is None")
                    time.sleep(RETRY_DELAY)
                    continue

                if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"✅ 約定成功: {direction} {lot_size}lots @ {result.price:.5f} "
                        f"ticket={result.order} SL={sl:.5f} TP={tp:.5f}"
                    )
                    return {
                        "ticket": result.order,
                        "price": result.price,
                        "lot_size": lot_size,
                        "direction": direction,
                        "sl": sl,
                        "tp": tp,
                        "spread": price_info["spread"],
                    }
                else:
                    logger.warning(
                        f"発注エラー (attempt {attempt}): "
                        f"retcode={result.retcode} comment={result.comment}"
                    )

                    # リトライ前に価格を更新
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                        new_price = self.get_current_price()
                        if new_price:
                            request["price"] = new_price["ask"] if direction == "BUY" else new_price["bid"]

            except Exception as e:
                logger.error(f"発注例外 (attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        logger.error(f"❌ 発注失敗: {MAX_RETRIES}回リトライ後も失敗")
        return None

    def close_position(self, ticket: int, reason: str = "manual") -> bool:
        """
        ポジションを全クローズ
        
        Args:
            ticket: ポジションチケット
            reason: クローズ理由
            
        Returns:
            bool: 成功/失敗
        """
        if not self._check_initialized():
            return False

        try:
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                logger.warning(f"ポジションが見つかりません: ticket={ticket}")
                return False

            pos = position[0]
            price_info = self.get_current_price()
            if price_info is None:
                return False

            if pos.type == self.mt5.ORDER_TYPE_BUY:
                close_type = self.mt5.ORDER_TYPE_SELL
                price = price_info["bid"]
            else:
                close_type = self.mt5.ORDER_TYPE_BUY
                price = price_info["ask"]

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": MAX_SLIPPAGE_POINTS,
                "magic": 202603,
                "comment": f"close_{reason}",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }

            for attempt in range(1, MAX_RETRIES + 1):
                result = self.mt5.order_send(request)
                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ クローズ成功: ticket={ticket} reason={reason}")
                    return True
                logger.warning(f"クローズ失敗 (attempt {attempt})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

            return False

        except Exception as e:
            logger.error(f"クローズ例外: {e}")
            return False

    def partial_close(self, ticket: int, lot_size: float) -> bool:
        """
        ポジションの部分クローズ
        
        Args:
            ticket: ポジションチケット
            lot_size: クローズするロットサイズ
        """
        if not self._check_initialized():
            return False

        try:
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                return False

            pos = position[0]
            price_info = self.get_current_price()
            if price_info is None:
                return False

            if pos.type == self.mt5.ORDER_TYPE_BUY:
                close_type = self.mt5.ORDER_TYPE_SELL
                price = price_info["bid"]
            else:
                close_type = self.mt5.ORDER_TYPE_BUY
                price = price_info["ask"]

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": lot_size,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": MAX_SLIPPAGE_POINTS,
                "magic": 202603,
                "comment": "partial_close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }

            result = self.mt5.order_send(request)
            if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ 部分クローズ成功: ticket={ticket} {lot_size}lots")
                return True

            logger.error(f"部分クローズ失敗: ticket={ticket}")
            return False

        except Exception as e:
            logger.error(f"部分クローズ例外: {e}")
            return False

    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """
        ポジションのSL/TPを変更
        
        Args:
            ticket: ポジションチケット
            sl: 新しいSL（Noneの場合は変更なし）
            tp: 新しいTP（Noneの場合は変更なし）
        """
        if not self._check_initialized():
            return False

        try:
            position = self.mt5.positions_get(ticket=ticket)
            if not position:
                return False

            pos = position[0]

            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": sl if sl is not None else pos.sl,
                "tp": tp if tp is not None else pos.tp,
            }

            result = self.mt5.order_send(request)
            if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                logger.debug(f"SL/TP変更成功: ticket={ticket} SL={sl} TP={tp}")
                return True

            logger.warning(f"SL/TP変更失敗: ticket={ticket}")
            return False

        except Exception as e:
            logger.error(f"SL/TP変更例外: {e}")
            return False

    def get_positions(self, symbol: Optional[str] = None) -> list:
        """
        オープンポジションを取得
        
        Args:
            symbol: 通貨ペア（Noneの場合は全ポジション）
        """
        if not self._check_initialized():
            return []

        try:
            if symbol:
                positions = self.mt5.positions_get(symbol=symbol)
            else:
                positions = self.mt5.positions_get()

            return list(positions) if positions else []

        except Exception as e:
            logger.error(f"ポジション取得例外: {e}")
            return []

    def get_account_info(self) -> Optional[dict]:
        """口座情報を取得"""
        if not self._check_initialized():
            return None

        try:
            account = self.mt5.account_info()
            if account is None:
                return None

            return {
                "login": account.login,
                "balance": account.balance,
                "equity": account.equity,
                "margin": account.margin,
                "free_margin": account.margin_free,
                "profit": account.profit,
                "leverage": account.leverage,
                "server": account.server,
            }
        except Exception as e:
            logger.error(f"口座情報取得例外: {e}")
            return None

    def _check_initialized(self) -> bool:
        """MT5初期化チェック"""
        if not self._initialized:
            logger.error("MT5が初期化されていません。initialize()を先に呼んでください。")
            return False
        return True
