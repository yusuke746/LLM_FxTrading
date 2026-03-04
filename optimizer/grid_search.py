"""
グリッドサーチ最適化モジュール
ATR倍率パラメータの全組み合わせをバックテストで検証

探索空間:
  SL_MULTIPLIER:    [1.0, 1.2, 1.5, 1.8, 2.0]
  BE_TRIGGER:       [0.8, 1.0, 1.2, 1.5]
  PARTIAL_TRIGGER:  [1.0, 1.2, 1.5, 1.8]
  TP_MULTIPLIER:    [2.0, 2.5, 3.0, 3.5]
  TRAIL_MULTIPLIER: [0.8, 1.0, 1.2, 1.5]

安全制約:
  - TP/SL比 ≥ 1.5 (RR比保証)
  - 部分利確トリガー > BEトリガー
  - Max DD ≤ 20%

評価指標（優先順）:
  1. Sharpe Ratio（最重視）
  2. Max Drawdown（上限20%で失格）
  3. Profit Factor

【期待値向上】
  - 並列処理 (multiprocessing) で高速化
  - Sharpe × (1 - DD/100) の複合スコアで過学習を抑制
"""

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from optimizer.backtest_runner import BacktestRunner
from logger_setup import get_logger

logger = get_logger("optimizer.grid_search")

# 探索空間
DEFAULT_PARAM_GRID = {
    "sl_multiplier":    [1.0, 1.2, 1.5, 1.8, 2.0],
    "be_trigger":       [0.8, 1.0, 1.2, 1.5],
    "partial_trigger":  [1.0, 1.2, 1.5, 1.8],
    "tp_multiplier":    [2.0, 2.5, 3.0, 3.5],
    "trail_multiplier": [0.8, 1.0, 1.2, 1.5],
}


def _run_single_backtest(args: Tuple) -> Tuple[dict, dict]:
    """並列実行用のラッパー関数（トップレベルで定義）"""
    params, df_dict, initial_balance, precomputed = args
    # DataFrameを再構築（pickleのため）
    df = pd.DataFrame(df_dict)
    runner = BacktestRunner()
    result = runner.run(df, params, initial_balance=initial_balance, precomputed=precomputed)
    return params, result


class GridSearchOptimizer:
    """グリッドサーチ最適化"""

    def __init__(
        self,
        param_grid: Optional[Dict[str, list]] = None,
        max_dd_limit: float = 20.0,
        min_rr_ratio: float = 1.5,
        max_workers: Optional[int] = None,
    ):
        self.param_grid = param_grid or DEFAULT_PARAM_GRID
        self.max_dd_limit = max_dd_limit
        self.min_rr_ratio = min_rr_ratio
        self.max_workers = max_workers

    def optimize(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000,
        top_n: int = 5,
    ) -> dict:
        """
        グリッドサーチを実行して最適パラメータを返す
        
        Args:
            df: H1足 OHLCVデータ（8週分以上推奨）
            initial_balance: 初期残高
            top_n: 上位N件の結果を返す
            
        Returns:
            dict: {
                "best_params": dict,
                "best_score": float,
                "best_result": dict,
                "top_results": list,
                "total_combinations": int,
                "valid_combinations": int,
            }
        """
        # パラメータの全組み合わせを生成
        all_combinations = self._generate_valid_combinations()
        total_combinations = len(list(itertools.product(*self.param_grid.values())))
        valid_combinations = len(all_combinations)

        logger.info(
            f"グリッドサーチ開始: 全{total_combinations}組み合わせ中 "
            f"{valid_combinations}パターンが有効"
        )

        results = []
        df_dict = df.to_dict(orient="list")

        # シグナル事前計算（1回のみ、全コンボで再利用）
        from optimizer.signal_precomputer import SignalPrecomputer
        precomputer = SignalPrecomputer()
        precomputed = precomputer.precompute(df)
        logger.info(
            f"シグナル事前計算完了: BUY={sum(precomputed['composite_direction'] == 1)} "
            f"SELL={sum(precomputed['composite_direction'] == -1)}"
        )

        # 並列実行
        if self.max_workers != 1 and valid_combinations > 10:
            try:
                args_list = [
                    (params, df_dict, initial_balance, precomputed)
                    for params in all_combinations
                ]

                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(_run_single_backtest, args): args[0]
                        for args in args_list
                    }

                    for future in as_completed(futures):
                        try:
                            params, result = future.result()
                            if self._is_valid_result(result):
                                score = self._calculate_composite_score(result)
                                results.append((params, result, score))
                        except Exception as e:
                            logger.debug(f"並列バックテスト例外: {e}")

            except Exception as e:
                logger.warning(f"並列処理失敗、逐次実行に切替: {e}")
                results = self._run_sequential(all_combinations, df, initial_balance, precomputed)
        else:
            results = self._run_sequential(all_combinations, df, initial_balance, precomputed)

        if not results:
            logger.warning("有効な結果がありません")
            return {
                "best_params": {},
                "best_score": -999,
                "best_result": {},
                "top_results": [],
                "total_combinations": total_combinations,
                "valid_combinations": valid_combinations,
            }

        # スコアでソート
        results.sort(key=lambda x: x[2], reverse=True)
        top_results = results[:top_n]

        best_params, best_result, best_score = results[0]

        logger.info(
            f"グリッドサーチ完了: 最適パラメータ={best_params} "
            f"Sharpe={best_result['sharpe_ratio']:.3f} "
            f"MaxDD={best_result['max_dd']:.1f}% "
            f"PF={best_result['profit_factor']:.2f}"
        )

        return {
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "best_result": best_result,
            "top_results": [
                {"params": p, "result": r, "score": round(s, 4)}
                for p, r, s in top_results
            ],
            "total_combinations": total_combinations,
            "valid_combinations": valid_combinations,
        }

    def _generate_valid_combinations(self) -> List[dict]:
        """安全制約を満たすパラメータ組み合わせのみを生成"""
        valid = []
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        for combo in itertools.product(*values):
            p = dict(zip(keys, combo))

            # 安全制約1: TP/SL比 ≥ 1.5
            if p["tp_multiplier"] / p["sl_multiplier"] < self.min_rr_ratio:
                continue

            # 安全制約2: 部分利確トリガー > BEトリガー
            if p["partial_trigger"] <= p["be_trigger"]:
                continue

            valid.append(p)

        return valid

    def _run_sequential(
        self,
        combinations: List[dict],
        df: pd.DataFrame,
        initial_balance: float,
        precomputed: Optional[Dict] = None,
    ) -> List[Tuple[dict, dict, float]]:
        """逐次バックテスト実行"""
        runner = BacktestRunner()
        results = []

        for i, params in enumerate(combinations):
            result = runner.run(df, params, initial_balance=initial_balance, precomputed=precomputed)

            if self._is_valid_result(result):
                score = self._calculate_composite_score(result)
                results.append((params, result, score))

            if (i + 1) % 100 == 0:
                logger.info(f"進捗: {i+1}/{len(combinations)}")

        return results

    def _is_valid_result(self, result: dict) -> bool:
        """結果が有効か（DD上限チェック）"""
        if result["max_dd"] > self.max_dd_limit:
            return False
        if result["total_trades"] < 10:  # 最低トレード数
            return False
        return True

    def _calculate_composite_score(self, result: dict) -> float:
        """
        複合スコアの算出
        Sharpe × (1 - DD/100) で過学習を抑制
        """
        sharpe = result["sharpe_ratio"]
        dd_penalty = 1 - result["max_dd"] / 100
        pf_bonus = min(result["profit_factor"] / 3, 0.5)  # PFボーナス（上限0.5）

        return sharpe * dd_penalty + pf_bonus
