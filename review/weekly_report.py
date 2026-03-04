"""
週次LLMレポート生成モジュール
OpenAI API (GPT-4o) が過去1週間のトレードログを分析し、
改善仮説・代替戦略案を生成

【期待値向上】レポートに以下を追加:
  - セッション別の勝率分析（どのセッションが最も有効か）
  - エンジン別の貢献度分析
  - 来週の注目イベント一覧
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from config_manager import get
from database import get_trade_log
from logger_setup import get_logger

logger = get_logger("review.weekly_report")

WEEKLY_REPORT_PROMPT = """あなたはプロのFXトレーディングアナリストです。
以下の1週間のトレードログを分析し、包括的なレポートを生成してください。

# 分析項目
1. **パフォーマンスサマリー**: 勝率、PF、総損益、最大DD
2. **セッション別分析**: アジア/ロンドン/NY/重複のそれぞれの勝率と平均PnL
3. **エンジン別分析**: トレンド/逆張り/ブレイクアウトの貢献度
4. **改善仮説**: データに基づいた具体的な改善提案（3つ以上）
5. **リスク評価**: 来週の注意点

# 出力形式（JSON）
{{
  "summary": {{
    "total_trades": 数値,
    "win_rate": 数値,
    "profit_factor": 数値,
    "total_pnl": 数値,
    "max_drawdown": 数値
  }},
  "session_analysis": {{
    "asia": {{"trades": 数値, "win_rate": 数値, "avg_pnl": 数値}},
    "london": {{"trades": 数値, "win_rate": 数値, "avg_pnl": 数値}},
    "overlap": {{"trades": 数値, "win_rate": 数値, "avg_pnl": 数値}},
    "ny_late": {{"trades": 数値, "win_rate": 数値, "avg_pnl": 数値}}
  }},
  "improvements": ["改善提案1", "改善提案2", "改善提案3"],
  "risk_warnings": ["注意点1", "注意点2"],
  "overall_assessment": "総合評価テキスト"
}}

# トレードログ
{trade_log}
"""


class WeeklyReportGenerator:
    """週次LLMレポート生成器"""

    def __init__(self):
        self.client = None
        self.model = get("llm.model", "gpt-4o")

    def _get_client(self) -> OpenAI:
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def generate(self, weeks: int = 1) -> dict:
        """
        週次レポートを生成
        
        Args:
            weeks: 分析対象の週数
            
        Returns:
            dict: レポート内容（LLM分析 + 統計データ）
        """
        # トレードログ取得
        trades = get_trade_log(weeks=weeks)

        if not trades:
            logger.info("トレードログなし → レポート生成スキップ")
            return {
                "generated": False,
                "reason": "No trades in the period",
            }

        # 基本統計（LLM不使用）
        stats = self._calculate_stats(trades)

        # LLMレポート（有効な場合のみ）
        llm_report = {}
        if get("llm.enabled", False):
            llm_report = self._generate_llm_report(trades)

        report = {
            "generated": True,
            "period_weeks": weeks,
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "llm_analysis": llm_report,
        }

        logger.info(f"週次レポート生成完了: {stats['total_trades']}件のトレードを分析")
        return report

    def _calculate_stats(self, trades: List[dict]) -> dict:
        """基本統計の算出（LLM不使用）"""
        closed_trades = [t for t in trades if t.get("status") == "closed"]

        if not closed_trades:
            return {"total_trades": 0}

        pnls = [t.get("pnl", 0) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # セッション別集計
        session_stats = {}
        for session in ["asia", "london", "overlap", "ny_late"]:
            session_trades = [t for t in closed_trades if t.get("session") == session]
            if session_trades:
                s_pnls = [t.get("pnl", 0) for t in session_trades]
                s_wins = [p for p in s_pnls if p > 0]
                session_stats[session] = {
                    "trades": len(session_trades),
                    "win_rate": len(s_wins) / len(session_trades) if session_trades else 0,
                    "avg_pnl": sum(s_pnls) / len(s_pnls) if s_pnls else 0,
                    "total_pnl": sum(s_pnls),
                }

        return {
            "total_trades": len(closed_trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "session_stats": session_stats,
        }

    def _generate_llm_report(self, trades: List[dict]) -> dict:
        """LLMによるレポート生成"""
        try:
            # トレードログを要約して送信（トークン節約）
            trade_summary = []
            for t in trades[:50]:  # 最大50件
                trade_summary.append({
                    "direction": t.get("direction"),
                    "pnl": t.get("pnl"),
                    "pnl_pips": t.get("pnl_pips"),
                    "session": t.get("session"),
                    "close_reason": t.get("close_reason"),
                    "signal_score": t.get("signal_score"),
                })

            trade_log_text = json.dumps(trade_summary, indent=2, ensure_ascii=False)

            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "FXトレーディングアナリスト。JSONのみで回答。"},
                    {"role": "user", "content": WEEKLY_REPORT_PROMPT.format(trade_log=trade_log_text)},
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            logger.info("LLMレポート生成完了")
            return result

        except Exception as e:
            logger.error(f"LLMレポート生成エラー: {e}")
            return {"error": str(e)}

    def format_discord_message(self, report: dict) -> str:
        """レポートをDiscord向けにフォーマット"""
        if not report.get("generated"):
            return "📊 週次レポート: トレードなし"

        stats = report.get("statistics", {})
        llm = report.get("llm_analysis", {})

        msg = (
            f"📊 **週次パフォーマンスレポート**\n"
            f"```\n"
            f"トレード数: {stats.get('total_trades', 0)}\n"
            f"勝率:       {stats.get('win_rate', 0)*100:.1f}%\n"
            f"PF:         {stats.get('profit_factor', 0):.2f}\n"
            f"総損益:     {stats.get('total_pnl', 0):.2f}\n"
            f"平均損益:   {stats.get('avg_pnl', 0):.2f}\n"
            f"最高益:     {stats.get('best_trade', 0):.2f}\n"
            f"最大損:     {stats.get('worst_trade', 0):.2f}\n"
            f"```\n"
        )

        # セッション別
        session_stats = stats.get("session_stats", {})
        if session_stats:
            msg += "\n**セッション別実績:**\n```\n"
            for s, data in session_stats.items():
                msg += f"{s:10s}: {data['trades']}件 勝率{data['win_rate']*100:.0f}% PnL={data['total_pnl']:.0f}\n"
            msg += "```\n"

        # LLM改善提案
        improvements = llm.get("improvements", [])
        if improvements:
            msg += "\n**💡 改善提案:**\n"
            for i, imp in enumerate(improvements, 1):
                msg += f"{i}. {imp}\n"

        return msg
