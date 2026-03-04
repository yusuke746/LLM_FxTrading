"""
エンジン IC（Information Coefficient）追跡モジュール

各エンジンの「先見性」を定量化:
  IC = corr(signal[t], forward_return[t+1..t+N])

ICが高いエンジンは「当たっている」= 重みを上げてよい
ICがゼロ/負のエンジンは「ノイズ or 逆指標」= 重みを下げるべき

用途:
  - ダッシュボードでの表示（どのエンジンが貢献しているか可視化）
  - 将来の動的重み調整の基盤
  - 「このシステムにエッジがあるか？」の客観的回答

計算方法:
  1. ローリングウィンドウ（直近50トレード）
  2. 各エンジンのシグナル値 vs 実際のPnL方向の相関（Spearman rank相関）
  3. IC > 0.05 なら有益、IC > 0.10 なら優秀、IC < 0 なら逆指標

注意:
  - 母集団が少ないうちは統計的に無意味なので closed_trades >= 30 から開始
  - ICの安定性も重要: IC_mean / IC_std (ICIR = Information Coefficient Information Ratio)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os
from datetime import datetime

from logger_setup import get_logger

logger = get_logger("monitoring.ic_tracker")


class ICTracker:
    """エンジン別 Information Coefficient 追跡器"""

    ENGINE_KEYS = [
        "trend", "mean_rev", "breakout",
        "momentum_div", "supply_demand",
        "session_orb", "market_structure",
    ]
    CATEGORY_KEYS = ["trend_follow", "mean_revert", "structural"]

    def __init__(
        self,
        window_size: int = 50,
        save_path: str = "data/ic_history.json",
    ):
        self.window_size = window_size
        self.save_path = save_path

        # ローリングウィンドウ: 各トレードの記録
        # (engine_signals, category_scores, regime, pnl, direction)
        self._trade_records: deque = deque(maxlen=window_size * 2)

        # 計算済みIC
        self._engine_ics: Dict[str, float] = {k: 0.0 for k in self.ENGINE_KEYS}
        self._category_ics: Dict[str, float] = {k: 0.0 for k in self.CATEGORY_KEYS}
        self._regime_accuracy: Dict[str, Dict] = {}

        # 履歴（時系列IC推移）
        self._ic_history: List[Dict] = []

        self._load()

    def record_trade(
        self,
        engine_signals: Dict[str, float],
        category_scores: Dict[str, float],
        regime: str,
        trade_direction: str,
        pnl: float,
        session: str = "",
    ):
        """
        トレード結果を記録

        Args:
            engine_signals: 各エンジンの生シグナル（エントリー時点）
            category_scores: カテゴリ別スコア（エントリー時点）
            regime: エントリー時のレジーム
            trade_direction: "BUY" or "SELL"
            pnl: 実現PnL（円）
            session: セッション名
        """
        # PnLを方向付きリターンに変換
        # BUYでプラスPnL = 正、SELLでプラスPnL = 正（シグナルが正しかった）
        signed_return = pnl  # そのままでOK（BUY/SELLに関わらずpnl>0=成功）

        record = {
            "engine_signals": {k: engine_signals.get(k, 0.0) for k in self.ENGINE_KEYS},
            "category_scores": {k: category_scores.get(k, 0.0) for k in self.CATEGORY_KEYS},
            "regime": regime,
            "direction": trade_direction,
            "pnl": pnl,
            "signed_return": signed_return,
            "session": session,
            "timestamp": datetime.now().isoformat(),
        }
        self._trade_records.append(record)

        # IC再計算
        if len(self._trade_records) >= 30:
            self._recalculate_ics()
            self._save()

    def _recalculate_ics(self):
        """ローリングウィンドウでICを再計算"""
        records = list(self._trade_records)[-self.window_size:]
        n = len(records)
        if n < 30:
            return

        returns = np.array([r["signed_return"] for r in records])

        # --- エンジン別IC ---
        for eng in self.ENGINE_KEYS:
            signals = np.array([
                abs(r["engine_signals"][eng]) * (1 if r["direction"] == "BUY" else -1)
                * np.sign(r["engine_signals"][eng])
                for r in records
            ])
            # 実質: シグナルの絶対方向 × トレード方向一致度 を PnL と相関
            # シンプルに: |signal| が大きいトレードほど pnl が大きいか？
            signal_abs = np.array([abs(r["engine_signals"][eng]) for r in records])
            if np.std(signal_abs) > 0 and np.std(returns) > 0:
                # Spearman rank相関
                ic = self._spearman_corr(signal_abs, np.sign(returns) * signal_abs)
                # より直感的: シグナル方向とPnL方向の一致率
                directional_ic = self._directional_ic(records, eng)
                self._engine_ics[eng] = round(directional_ic, 4)
            else:
                self._engine_ics[eng] = 0.0

        # --- カテゴリ別IC ---
        for cat in self.CATEGORY_KEYS:
            cat_signals = np.array([abs(r["category_scores"][cat]) for r in records])
            if np.std(cat_signals) > 0 and np.std(returns) > 0:
                directional_ic = self._directional_cat_ic(records, cat)
                self._category_ics[cat] = round(directional_ic, 4)
            else:
                self._category_ics[cat] = 0.0

        # --- レジーム別精度 ---
        regime_groups = {}
        for r in records:
            reg = r["regime"]
            if reg not in regime_groups:
                regime_groups[reg] = []
            regime_groups[reg].append(r["pnl"])

        self._regime_accuracy = {}
        for reg, pnls in regime_groups.items():
            wins = sum(1 for p in pnls if p > 0)
            total = len(pnls)
            self._regime_accuracy[reg] = {
                "win_rate": round(wins / total, 3) if total > 0 else 0,
                "avg_pnl": round(np.mean(pnls), 2),
                "count": total,
            }

        # ICヒストリーに追加
        self._ic_history.append({
            "timestamp": datetime.now().isoformat(),
            "n_trades": n,
            "engine_ics": self._engine_ics.copy(),
            "category_ics": self._category_ics.copy(),
            "regime_accuracy": self._regime_accuracy.copy(),
        })

        logger.info(
            f"IC更新 (n={n}): "
            + " | ".join(f"{eng}={ic:.3f}" for eng, ic in self._engine_ics.items())
        )

    def _directional_ic(self, records: List[Dict], engine: str) -> float:
        """
        方向一致IC: エンジンが正しい方向を当てている割合 - 0.5

        IC > 0: 予測力あり
        IC = 0: ランダム
        IC < 0: 逆指標
        """
        correct = 0
        total = 0

        for r in records:
            sig = r["engine_signals"][engine]
            if abs(sig) < 0.05:  # シグナルなし = 計算外
                continue

            pnl = r["pnl"]
            direction = r["direction"]

            # シグナル方向とトレード方向が一致 + PnL > 0 → 正解
            # シグナル方向とトレード方向が一致 + PnL < 0 → 不正解
            if direction == "BUY":
                if sig > 0 and pnl > 0:
                    correct += 1
                elif sig > 0 and pnl <= 0:
                    pass  # incorrect
                # sig < 0 は BUYトレードでは逆方向なので含めない
            else:  # SELL
                if sig < 0 and pnl > 0:
                    correct += 1
                elif sig < 0 and pnl <= 0:
                    pass

            total += 1

        if total < 10:
            return 0.0
        return (correct / total) - 0.5  # 0.5を引いてICとする

    def _directional_cat_ic(self, records: List[Dict], category: str) -> float:
        """カテゴリ版 方向一致IC"""
        correct = 0
        total = 0

        for r in records:
            score = r["category_scores"][category]
            if abs(score) < 0.1:
                continue

            pnl = r["pnl"]
            direction = r["direction"]

            if direction == "BUY":
                if score > 0 and pnl > 0:
                    correct += 1
            else:
                if score < 0 and pnl > 0:
                    correct += 1

            total += 1

        if total < 10:
            return 0.0
        return (correct / total) - 0.5

    def _spearman_corr(self, x: np.ndarray, y: np.ndarray) -> float:
        """Spearman順位相関"""
        n = len(x)
        if n < 10:
            return 0.0
        rank_x = np.argsort(np.argsort(x)).astype(float)
        rank_y = np.argsort(np.argsort(y)).astype(float)
        d = rank_x - rank_y
        rho = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
        return rho

    # === 外部API ===

    def get_engine_ics(self) -> Dict[str, float]:
        """各エンジンのIC値を返す"""
        return self._engine_ics.copy()

    def get_category_ics(self) -> Dict[str, float]:
        """各カテゴリのIC値を返す"""
        return self._category_ics.copy()

    def get_regime_accuracy(self) -> Dict[str, Dict]:
        """レジーム別の精度を返す"""
        return self._regime_accuracy.copy()

    def get_summary(self) -> Dict:
        """IC追跡のサマリーを返す"""
        n = len(self._trade_records)
        return {
            "total_tracked_trades": n,
            "min_required": 30,
            "is_significant": n >= 30,
            "engine_ics": self._engine_ics,
            "category_ics": self._category_ics,
            "regime_accuracy": self._regime_accuracy,
            "best_engine": max(self._engine_ics, key=self._engine_ics.get)
                if any(v != 0 for v in self._engine_ics.values()) else "N/A",
            "worst_engine": min(self._engine_ics, key=self._engine_ics.get)
                if any(v != 0 for v in self._engine_ics.values()) else "N/A",
        }

    def format_report(self) -> str:
        """IC追跡レポートをフォーマットして返す"""
        summary = self.get_summary()
        lines = [
            "═══ Engine IC Report ═══",
            f"追跡トレード数: {summary['total_tracked_trades']} / {self.window_size}",
        ]

        if not summary["is_significant"]:
            lines.append(f"⚠️ 統計的有意性には {30 - summary['total_tracked_trades']} トレード不足")
            return "\n".join(lines)

        lines.append("\n--- エンジン別 IC (>0=有益, >0.05=優秀, <0=逆指標) ---")
        for eng, ic in sorted(self._engine_ics.items(), key=lambda x: -x[1]):
            bar = "█" * int(max(0, ic + 0.5) * 20)
            status = "✓" if ic > 0.05 else ("─" if ic > -0.02 else "✗")
            lines.append(f"  {status} {eng:20s}: {ic:+.4f}  {bar}")

        lines.append("\n--- カテゴリ別 IC ---")
        for cat, ic in sorted(self._category_ics.items(), key=lambda x: -x[1]):
            status = "✓" if ic > 0.05 else ("─" if ic > -0.02 else "✗")
            lines.append(f"  {status} {cat:15s}: {ic:+.4f}")

        if self._regime_accuracy:
            lines.append("\n--- レジーム別精度 ---")
            for reg, stats in self._regime_accuracy.items():
                lines.append(
                    f"  {reg:10s}: WR={stats['win_rate']:.1%} "
                    f"AvgPnL=¥{stats['avg_pnl']:,.0f} "
                    f"(n={stats['count']})"
                )

        return "\n".join(lines)

    # === 永続化 ===

    def _save(self):
        """IC情報をファイルに保存"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            data = {
                "records": [
                    {k: v for k, v in r.items()} for r in self._trade_records
                ],
                "engine_ics": self._engine_ics,
                "category_ics": self._category_ics,
                "regime_accuracy": self._regime_accuracy,
                "ic_history": self._ic_history[-100:],  # 最新100件のみ
            }
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"IC保存エラー: {e}")

    def _load(self):
        """IC情報をファイルから読み込み"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for r in data.get("records", []):
                    self._trade_records.append(r)
                self._engine_ics = data.get("engine_ics", self._engine_ics)
                self._category_ics = data.get("category_ics", self._category_ics)
                self._regime_accuracy = data.get("regime_accuracy", {})
                self._ic_history = data.get("ic_history", [])

                logger.info(f"IC履歴ロード: {len(self._trade_records)}トレード")
        except Exception as e:
            logger.warning(f"IC履歴ロードエラー: {e}")
