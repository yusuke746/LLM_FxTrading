"""
バックテスト実行モジュール
グリッドサーチ用にパラメータセットを受けてバックテストを実行

EUR/USD H1足専用。セッション別エンジン重み + レジームゲーティング対応。

【v2変更点】
  旧: _get_simple_signal() でライブとは別の簡易ロジック → 最適化が無意味
  新: SignalPrecomputer で7エンジンのロジックを忠実に再現し
      レジームゲーティング + コンフルエンス判定を含む合成シグナルを使用
      グリッドサーチはSL/TP/BE/Trailのみ変化 → シグナルは1回計算で再利用
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

from logger_setup import get_logger

logger = get_logger("optimizer.backtest")


class BacktestRunner:
    """ベクトル化バックテスト実行器"""

    def __init__(self):
        self.point = 0.00001   # EUR/USD 1ポイント
        self.pip = 0.0001      # 1pip
        # JPY口座用: 1ロット1pip = $10 × USDJPYレート
        # バックテストでは固定レートを使用
        self.usdjpy_rate = 150.0  # バックテスト用固定レート
        self.pip_value = 10.0 * self.usdjpy_rate  # 1ロット1pip = ¥1,500
        self.max_lot = 30.0    # ロット上限
        self._precomputed = None  # シグナル事前計算結果キャッシュ

    def precompute_signals(self, df: pd.DataFrame) -> None:
        """
        シグナルを事前計算してキャッシュ
        グリッドサーチ前に1回呼び出す。以降の run() はキャッシュを使用。
        """
        from optimizer.signal_precomputer import SignalPrecomputer
        precomputer = SignalPrecomputer()
        self._precomputed = precomputer.precompute(df)
        logger.info(
            f"シグナル事前計算完了: {len(df)}本 | "
            f"BUYシグナル={np.sum(self._precomputed['composite_direction'] == 1)} | "
            f"SELLシグナル={np.sum(self._precomputed['composite_direction'] == -1)}"
        )

    def run(
        self,
        df: pd.DataFrame,
        params: Dict[str, float],
        initial_balance: float = 1_000_000,  # 100万円想定
        risk_per_trade: float = 0.01,
        precomputed: Optional[Dict] = None,
    ) -> dict:
        """
        バックテストを実行
        
        Args:
            df: H1足 OHLCVデータ
            params: {
                "sl_multiplier": float,
                "be_trigger": float,
                "partial_trigger": float,
                "tp_multiplier": float,
                "trail_multiplier": float,
            }
            initial_balance: 初期残高
            risk_per_trade: 1トレードあたりのリスク比率
            precomputed: 事前計算済みシグナル（Noneの場合はその場で計算）
            
        Returns:
            dict: バックテスト結果
        """
        if len(df) < 100:
            return self._empty_result()

        sl_mult = params["sl_multiplier"]
        be_trigger = params["be_trigger"]
        partial_trigger = params["partial_trigger"]
        tp_mult = params["tp_multiplier"]
        trail_mult = params["trail_multiplier"]

        # テクニカル指標の算出（ATR はポジション管理にも必要）
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
        atr = atr_series.values

        # === シグナル取得 ===
        signals = precomputed or self._precomputed
        if signals is None:
            # 事前計算されていない場合はその場で計算
            from optimizer.signal_precomputer import SignalPrecomputer
            precomputer = SignalPrecomputer()
            signals = precomputer.precompute(df)

        signal_dir = signals["composite_direction"]
        signal_score = signals["composite_score"]

        # バックテスト実行
        balance = initial_balance
        peak_balance = initial_balance
        trades = []
        in_trade = False
        trade_direction = None
        entry_price = 0.0
        trade_sl = 0.0
        trade_tp = 0.0
        trade_atr = 0.0
        lot_size = 0.0
        be_set = False
        partial_done = False
        remaining_lot = 0.0
        trail_sl = 0.0

        for i in range(60, len(df)):
            if np.isnan(atr[i]):
                continue

            current_atr = atr[i]

            if not in_trade:
                # === プリコンピュートされた合成シグナルを使用 ===
                if i < len(signal_dir) and signal_dir[i] != 0:
                    direction = "BUY" if signal_dir[i] > 0 else "SELL"
                    sl_distance = current_atr * sl_mult
                    tp_distance = current_atr * tp_mult
                    sl_pips = sl_distance / self.pip

                    # ロットサイズ計算（JPY口座対応）
                    risk_amount = balance * risk_per_trade  # JPY
                    lot = risk_amount / (sl_pips * self.pip_value)  # pip_valueはJPY
                    lot = max(0.01, min(round(int(lot * 100) / 100, 2), self.max_lot))

                    entry_price = close[i]
                    if direction == "BUY":
                        trade_sl = entry_price - sl_distance
                        trade_tp = entry_price + tp_distance
                    else:
                        trade_sl = entry_price + sl_distance
                        trade_tp = entry_price - tp_distance

                    trade_direction = direction
                    trade_atr = current_atr
                    lot_size = lot
                    remaining_lot = lot
                    in_trade = True
                    be_set = False
                    partial_done = False
                    trail_sl = trade_sl
                    entry_bar = i

            else:
                # ポジション管理（変更なし）
                cur_price = close[i]
                cur_high = high[i]
                cur_low = low[i]

                if trade_direction == "BUY":
                    unrealized = cur_price - entry_price
                    # SLチェック
                    if cur_low <= trail_sl:
                        pnl_pips = (trail_sl - entry_price) / self.pip
                        pnl = pnl_pips * remaining_lot * self.pip_value
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "BUY", "bars": i - entry_bar, "bar_idx": i})
                        in_trade = False
                        continue
                    # TPチェック
                    if cur_high >= trade_tp:
                        pnl_pips = (trade_tp - entry_price) / self.pip
                        pnl = pnl_pips * remaining_lot * self.pip_value
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "BUY", "bars": i - entry_bar, "bar_idx": i})
                        in_trade = False
                        continue
                else:
                    unrealized = entry_price - cur_price
                    if cur_high >= trail_sl:
                        pnl_pips = (entry_price - trail_sl) / self.pip
                        pnl = pnl_pips * remaining_lot * self.pip_value
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "SELL", "bars": i - entry_bar, "bar_idx": i})
                        in_trade = False
                        continue
                    if cur_low <= trade_tp:
                        pnl_pips = (entry_price - trade_tp) / self.pip
                        pnl = pnl_pips * remaining_lot * self.pip_value
                        balance += pnl
                        trades.append({"pnl": pnl, "direction": "SELL", "bars": i - entry_bar, "bar_idx": i})
                        in_trade = False
                        continue

                # STEP1: BE設定
                if not be_set and unrealized >= trade_atr * be_trigger:
                    if trade_direction == "BUY":
                        trail_sl = entry_price + self.pip * 2  # 建値 + 少し上
                    else:
                        trail_sl = entry_price - self.pip * 2
                    be_set = True

                # STEP2: 部分利確
                if be_set and not partial_done and unrealized >= trade_atr * partial_trigger:
                    partial_lot = round(remaining_lot * 0.5, 2)
                    partial_pnl_pips = unrealized / self.pip
                    partial_pnl = partial_pnl_pips * partial_lot * self.pip_value
                    balance += partial_pnl
                    remaining_lot = round(remaining_lot - partial_lot, 2)
                    remaining_lot = max(remaining_lot, 0.01)
                    partial_done = True

                    # STEP3: TP再設定
                    if trade_direction == "BUY":
                        trade_tp = entry_price + trade_atr * tp_mult
                    else:
                        trade_tp = entry_price - trade_atr * tp_mult

                # STEP4: トレーリング
                if partial_done:
                    trail_distance = trade_atr * trail_mult
                    if trade_direction == "BUY":
                        new_trail = cur_price - trail_distance
                        if new_trail > trail_sl:
                            trail_sl = new_trail
                    else:
                        new_trail = cur_price + trail_distance
                        if new_trail < trail_sl:
                            trail_sl = new_trail

                peak_balance = max(peak_balance, balance)

        # 結果集計
        return self._calculate_metrics(trades, initial_balance, balance, peak_balance)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        params: Dict[str, float],
        n_splits: int = 3,
        is_ratio: float = 0.7,  # In-Sample比率
    ) -> dict:
        """
        ウォークフォワード検証（アンカード方式）
        
        データを n_splits 期間に分割し、毎回 IS(前半) で最適化した
        パラメータを OOS(後半) で検証して安定性を評価する。
        
        n_splits=3 の場合:
          期間1: IS=[0-70%], OOS=[70-100%]   (1/3 スライス)
          期間2: IS=[0-70%], OOS=[70-100%]   (2/3 スライス)
          期間3: IS=[0-70%], OOS=[70-100%]   (全体)
        """
        from optimizer.signal_precomputer import SignalPrecomputer

        total_bars = len(df)
        
        # 最低でも全体を IS/OOS に分割して比較
        is_end = int(total_bars * is_ratio)
        
        if is_end < 200 or (total_bars - is_end) < 100:
            # データ不足 → 全体でバックテストのみ
            full_result = self.run(df, params)
            full_result["stability_score"] = 0.5  # 不明
            full_result["period_results"] = [full_result]
            return full_result

        # In-Sample / Out-of-Sample 分割
        # 各分割に対してシグナルを再計算
        precomputer = SignalPrecomputer()

        is_df = df.iloc[:is_end].reset_index(drop=True)
        oos_df = df.iloc[is_end:].reset_index(drop=True)
        
        is_signals = precomputer.precompute(is_df)
        oos_signals = precomputer.precompute(oos_df)
        
        is_result = self.run(is_df, params, precomputed=is_signals)
        oos_result = self.run(oos_df, params, precomputed=oos_signals)
        
        results = [is_result, oos_result]
        
        # 追加分割（データ十分なら3分割でOOS検証を増やす）
        if n_splits >= 3 and total_bars > 600:
            split_size = total_bars // n_splits
            for i in range(n_splits):
                start = i * split_size
                end = min(start + split_size, total_bars)
                split_df = df.iloc[start:end].reset_index(drop=True)
                if len(split_df) >= 100:
                    split_signals = precomputer.precompute(split_df)
                    result = self.run(split_df, params, precomputed=split_signals)
                    if result["total_trades"] >= 3:
                        results.append(result)

        # 有効な結果のみフィルター
        valid_results = [r for r in results if r["total_trades"] >= 3]
        if not valid_results:
            return self._empty_result()

        # 各期間の結果を集約
        sharpes = [r["sharpe_ratio"] for r in valid_results]
        avg_sharpe = np.mean(sharpes)
        max_dd = max(r["max_dd"] for r in valid_results)
        avg_pf = np.mean([r["profit_factor"] for r in valid_results])
        avg_winrate = np.mean([r["win_rate"] for r in valid_results])
        total_trades = sum(r["total_trades"] for r in valid_results)

        # 安定性スコア: OOS Sharpe / IS Sharpe（1.0に近いほど安定、過学習なし）
        is_sharpe = is_result["sharpe_ratio"]
        oos_sharpe = oos_result["sharpe_ratio"]
        
        if is_sharpe > 0 and oos_sharpe >= 0:
            # OOS/IS 比率（1.0 = 完全安定、0.5以上なら良好）
            stability_score = min(oos_sharpe / is_sharpe, 1.0)
        elif is_sharpe > 0 and oos_sharpe < 0:
            # OOSで損失 → 過学習
            stability_score = 0.0
        elif is_sharpe <= 0:
            # ISですでに不良
            stability_score = 0.0
        else:
            stability_score = 0.5

        # CVも加味（Sharpe変動係数が小さいほど安定）
        if len(sharpes) >= 3:
            sharpe_cv = np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-10)
            cv_penalty = max(0, 1 - sharpe_cv)
            stability_score = stability_score * 0.7 + cv_penalty * 0.3

        return {
            "sharpe_ratio": round(avg_sharpe, 4),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(avg_pf, 4),
            "win_rate": round(avg_winrate, 4),
            "total_trades": total_trades,
            "stability_score": round(stability_score, 4),
            "is_sharpe": round(is_sharpe, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "period_results": valid_results,
        }

    def _calculate_metrics(
        self,
        trades: list,
        initial_balance: float,
        final_balance: float,
        peak_balance: float,
    ) -> dict:
        """バックテスト結果のメトリクス計算"""
        if not trades:
            return self._empty_result()

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        # Sharpe Ratio（日次PnLベース: H1=24本で1日）
        bars_per_day = 24  # H1 timeframe
        daily_pnl = {}
        for t in trades:
            day_key = t.get("bar_idx", 0) // bars_per_day
            daily_pnl[day_key] = daily_pnl.get(day_key, 0) + t["pnl"]
        
        if len(daily_pnl) >= 2:
            daily_returns = np.array(list(daily_pnl.values()))
            daily_std = daily_returns.std()
            if daily_std > 0:
                sharpe = (daily_returns.mean() / daily_std) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max Drawdown
        cumulative = np.cumsum(pnls)
        balance_curve = initial_balance + cumulative
        peak = np.maximum.accumulate(balance_curve)
        drawdown = (peak - balance_curve) / peak * 100
        max_dd = drawdown.max() if len(drawdown) > 0 else 0

        # 連敗
        max_consecutive_losses = 0
        current_streak = 0
        for p in pnls:
            if p <= 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        return {
            "sharpe_ratio": round(sharpe, 4),
            "max_dd": round(max_dd, 2),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(np.mean(pnls), 2),
            "max_consecutive_losses": max_consecutive_losses,
            "final_balance": round(final_balance, 2),
            "return_pct": round((final_balance / initial_balance - 1) * 100, 2),
        }

    def _empty_result(self) -> dict:
        return {
            "sharpe_ratio": -999,
            "max_dd": 100,
            "profit_factor": 0,
            "win_rate": 0,
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0,
            "avg_pnl": 0,
            "max_consecutive_losses": 0,
            "final_balance": 0,
            "return_pct": -100,
        }
