"""
Discord通知モジュール
Webhook URLを使ってDiscordにアラートを送信
"""

import json
from typing import Optional

import requests

from config_manager import get
from logger_setup import get_logger

logger = get_logger("notifier")


class DiscordNotifier:
    """Discord Webhook通知"""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or get("discord.webhook_url", "")
        self.enabled = get("discord.enabled", False) and bool(self.webhook_url)

    def send(self, message: str, username: str = "FX Bot") -> bool:
        """
        Discordにメッセージを送信
        
        Args:
            message: 送信メッセージ（Markdown対応）
            username: 表示名
        """
        if not self.enabled:
            logger.debug("Discord通知は無効です")
            return False

        try:
            # Discordの2000文字制限
            if len(message) > 1900:
                message = message[:1900] + "\n... (省略)"

            payload = {
                "content": message,
                "username": username,
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )

            if response.status_code in (200, 204):
                logger.debug("Discord通知送信成功")
                return True
            else:
                logger.warning(f"Discord通知失敗: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Discord通知例外: {e}")
            return False

    def send_trade_alert(self, trade_info: dict) -> bool:
        """トレードアラートを送信"""
        direction_emoji = "🟢" if trade_info.get("direction") == "BUY" else "🔴"
        msg = (
            f"{direction_emoji} **{trade_info.get('direction', 'N/A')}** "
            f"{trade_info.get('lot_size', 0)} lots @ {trade_info.get('price', 0):.5f}\n"
            f"SL: {trade_info.get('sl', 0):.5f} | TP: {trade_info.get('tp', 0):.5f}\n"
            f"スコア: {trade_info.get('signal_score', 0):.3f} | "
            f"セッション: {trade_info.get('session', 'N/A')}"
        )
        return self.send(msg)

    def send_close_alert(self, close_info: dict) -> bool:
        """クローズアラートを送信"""
        pnl = close_info.get("pnl", 0)
        emoji = "💰" if pnl > 0 else "💸"
        msg = (
            f"{emoji} **ポジションクローズ** ticket={close_info.get('ticket', 'N/A')}\n"
            f"PnL: {pnl:+.2f} | 理由: {close_info.get('reason', 'N/A')}"
        )
        return self.send(msg)

    def send_error_alert(self, error_msg: str) -> bool:
        """エラーアラートを送信"""
        return self.send(f"🚨 **エラー**: {error_msg}")
