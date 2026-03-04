"""
両建て防止ロジック
反対方向のポジションが存在する場合は自動クローズしてから新規エントリー

状態遷移表:
  なし + BUY → 新規BUY
  なし + SELL → 新規SELL
  BUY保有 + BUY → 上限未満なら追加BUY
  BUY保有 + SELL → BUYを全クローズ → SELLエントリー
  SELL保有 + SELL → 上限未満なら追加SELL
  SELL保有 + BUY → SELLを全クローズ → BUYエントリー
"""

from typing import List, Optional, Callable

from config_manager import get
from logger_setup import get_logger

logger = get_logger("position.no_hedge")


class NoHedgeController:
    """両建て防止コントローラー"""

    def __init__(self, executor=None):
        """
        Args:
            executor: MT5執行インスタンス（close_position, get_positions メソッドを持つ）
        """
        self.executor = executor
        self.max_positions = get("risk.max_positions", 3)

    def pre_entry_check(
        self,
        symbol: str,
        direction: str,
    ) -> dict:
        """
        エントリー前の両建てチェック＆反対ポジションクローズ
        
        Args:
            symbol: 通貨ペア（例: "EURUSD"）
            direction: "BUY" or "SELL"
            
        Returns:
            dict: {
                "can_entry": bool,
                "closed_positions": list,
                "reason": str,
            }
        """
        if self.executor is None:
            logger.warning("MT5 executorが未設定")
            return {"can_entry": False, "closed_positions": [], "reason": "executor未設定"}

        positions = self.executor.get_positions(symbol)
        closed = []

        # 反対方向のポジションをクローズ
        for pos in positions:
            pos_direction = self._get_position_direction(pos)
            if pos_direction and pos_direction != direction:
                try:
                    ticket = self._get_ticket(pos)
                    success = self.executor.close_position(ticket, reason="reverse_signal")
                    if success:
                        closed.append(ticket)
                        logger.info(
                            f"反対ポジション自動クローズ: ticket={ticket} "
                            f"({pos_direction} → {direction})"
                        )
                    else:
                        logger.error(f"ポジションクローズ失敗: ticket={ticket}")
                        return {
                            "can_entry": False,
                            "closed_positions": closed,
                            "reason": f"反対ポジションクローズ失敗: {ticket}",
                        }
                except Exception as e:
                    logger.error(f"ポジションクローズ例外: {e}")
                    return {
                        "can_entry": False,
                        "closed_positions": closed,
                        "reason": f"クローズ例外: {e}",
                    }

        # 同方向ポジションの上限チェック
        same_dir_positions = self.executor.get_positions(symbol)
        same_dir_count = sum(
            1 for p in same_dir_positions
            if self._get_position_direction(p) == direction
        )

        if same_dir_count >= self.max_positions:
            logger.info(
                f"同方向ポジション上限到達: {same_dir_count}/{self.max_positions}"
            )
            return {
                "can_entry": False,
                "closed_positions": closed,
                "reason": f"同方向ポジション上限 ({same_dir_count}/{self.max_positions})",
            }

        return {
            "can_entry": True,
            "closed_positions": closed,
            "reason": "OK",
        }

    def _get_position_direction(self, position) -> Optional[str]:
        """ポジションオブジェクトからBUY/SELLを取得"""
        if hasattr(position, "type"):
            # MT5のポジションオブジェクト
            import MetaTrader5 as mt5
            return "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        elif isinstance(position, dict):
            return position.get("direction", position.get("type", None))
        return None

    def _get_ticket(self, position) -> int:
        """ポジションオブジェクトからチケット番号を取得"""
        if hasattr(position, "ticket"):
            return position.ticket
        elif isinstance(position, dict):
            return position.get("ticket", 0)
        return 0
