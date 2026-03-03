"""
Hyperparameter Optimization with Optuna
Finds optimal strategy parameters with constraints.
"""

import optuna
import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Callable, List, Tuple
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visual_logger import visual

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StrategyOptimizer:
    """Optuna-based strategy parameter optimizer."""
    
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_func: Callable,
                 n_trials: int = 100,
                 max_drawdown_limit: float = 0.20,
                 min_trades: int = 50):
        """
        Initialize optimizer.
        
        Args:
            data: Historical OHLCV data
            strategy_func: Function(data, **params) -> (entries, exits)
            n_trials: Number of optimization trials
            max_drawdown_limit: Maximum allowed drawdown (20%)
            min_trades: Minimum number of trades required
        """
        self.data = data
        self.strategy_func = strategy_func
        self.n_trials = n_trials
        self.max_drawdown_limit = max_drawdown_limit
        self.min_trades = min_trades
        
        self.best_params = None
        self.best_value = None
        self.study = None
        
    def optimize(self, 
                 param_space: Dict,
                 objective: str = 'sharpe',
                 n_jobs: int = 1) -> Dict:
        """
        Run optimization.
        
        Args:
            param_space: Dictionary defining parameter ranges
                        e.g., {'rsi_period': (10, 20), 'macd_fast': (8, 14)}
            objective: 'sharpe', 'return', 'profit_factor', 'calmar'
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary with best parameters
        """
        self.objective = objective
        self.param_space = param_space
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # Run optimization with visual progress
        visual.print_optimization_header()
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        
        # Custom callback for visual updates
        def callback(study, trial):
            if trial.number % 10 == 0 or trial.number == self.n_trials - 1:
                visual.print_optimization_progress(
                    trial=trial.number + 1,
                    total=self.n_trials,
                    best_value=study.best_value if study.best_value else 0
                )
        
        self.study.optimize(
            self._objective_function,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
            callbacks=[callback]
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        # Visual completion
        visual.print_optimization_complete(self.best_params, self.best_value)
        
        return self.best_params
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Sample parameters
        params = self._sample_params(trial)
        
        try:
            # Generate signals
            entries, exits = self.strategy_func(self.data, **params)
            
            # Skip if no signals
            if not entries.any() or not exits.any():
                return -999
            
            # Run backtest
            portfolio = vbt.Portfolio.from_signals(
                close=self.data['close'],
                entries=entries,
                exits=exits,
                init_cash=10000,
                fees=0.001,
                freq='1h'
            )
            
            # Get metrics
            stats = portfolio.stats()
            
            sharpe = stats.get('Sharpe Ratio', -999)
            total_return = stats.get('Total Return [%]', -999) / 100
            max_drawdown = stats.get('Max Drawdown [%]', 100) / 100
            profit_factor = self._calculate_profit_factor(portfolio)
            total_trades = stats.get('Total Trades', 0)
            calmar = stats.get('Calmar Ratio', -999)
            
            # Constraints - penalize if violated
            penalty = 0
            
            if max_drawdown > self.max_drawdown_limit:
                penalty += (max_drawdown - self.max_drawdown_limit) * 10
                
            if total_trades < self.min_trades:
                penalty += (self.min_trades - total_trades) / self.min_trades
            
            # Objective selection
            if self.objective == 'sharpe':
                value = sharpe
            elif self.objective == 'return':
                value = total_return
            elif self.objective == 'profit_factor':
                value = profit_factor
            elif self.objective == 'calmar':
                value = calmar
            else:
                value = sharpe
            
            return value - penalty
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -999
    
    def _sample_params(self, trial: optuna.Trial) -> Dict:
        """Sample parameters from defined space."""
        params = {}
        
        for name, config in self.param_space.items():
            if isinstance(config, tuple):
                if len(config) == 2 and isinstance(config[0], int):
                    # Integer range
                    params[name] = trial.suggest_int(name, config[0], config[1])
                elif len(config) == 2 and isinstance(config[0], float):
                    # Float range
                    params[name] = trial.suggest_float(name, config[0], config[1])
                else:
                    # Categorical
                    params[name] = trial.suggest_categorical(name, config)
            elif isinstance(config, list):
                # Categorical
                params[name] = trial.suggest_categorical(name, config)
                
        return params
    
    def _calculate_profit_factor(self, portfolio: vbt.Portfolio) -> float:
        """Calculate profit factor."""
        trades = portfolio.trades
        if hasattr(trades, 'returns'):
            returns = trades.returns
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            return gains / losses if losses > 0 else float('inf')
        return 0
    
    def get_optimization_report(self) -> pd.DataFrame:
        """Get report of all trials."""
        if self.study is None:
            raise ValueError("Run optimize() first")
            
        trials_df = self.study.trials_dataframe()
        return trials_df
    
    def plot_optimization_results(self):
        """Plot optimization results."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(
            self.study, ax=axes[0, 0]
        )
        
        # Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(
            self.study, ax=axes[0, 1]
        )
        
        # Slice plot
        optuna.visualization.matplotlib.plot_slice(
            self.study, ax=axes[1, 0]
        )
        
        # Contour plot
        optuna.visualization.matplotlib.plot_contour(
            self.study, ax=axes[1, 1]
        )
        
        plt.tight_layout()
        plt.show()


class MultiObjectiveOptimizer:
    """Multi-objective optimization for balancing return and risk."""
    
    def __init__(self, data: pd.DataFrame, strategy_func: Callable):
        self.data = data
        self.strategy_func = strategy_func
        
    def optimize(self,
                 param_space: Dict,
                 n_trials: int = 100) -> List[Dict]:
        """
        Optimize for multiple objectives simultaneously.
        
        Objectives:
        1. Maximize Sharpe ratio
        2. Minimize Max Drawdown
        """
        def objective(trial):
            params = self._sample_params(trial, param_space)
            
            try:
                entries, exits = self.strategy_func(self.data, **params)
                
                portfolio = vbt.Portfolio.from_signals(
                    close=self.data['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=10000,
                    fees=0.001
                )
                
                stats = portfolio.stats()
                sharpe = stats.get('Sharpe Ratio', 0)
                max_dd = stats.get('Max Drawdown [%]', 100) / 100
                
                return sharpe, max_dd
                
            except Exception:
                return 0, 1
        
        study = optuna.create_study(
            directions=['maximize', 'minimize']
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get Pareto front
        pareto_front = study.best_trials
        
        results = []
        for trial in pareto_front:
            results.append({
                'params': trial.params,
                'sharpe': trial.values[0],
                'max_drawdown': trial.values[1]
            })
        
        return results
    
    def _sample_params(self, trial: optuna.Trial, param_space: Dict) -> Dict:
        """Sample parameters."""
        params = {}
        
        for name, config in param_space.items():
            if isinstance(config, tuple) and len(config) == 2:
                if isinstance(config[0], int):
                    params[name] = trial.suggest_int(name, config[0], config[1])
                else:
                    params[name] = trial.suggest_float(name, config[0], config[1])
                    
        return params


def create_strategy_param_space(strategy_type: str = 'quant') -> Dict:
    """
    Create default parameter space for strategies.
    
    Args:
        strategy_type: 'quant', 'rsi', 'macd', 'combo'
    """
    spaces = {
        'quant': {
            'rsi_period': (10, 20),
            'rsi_overbought': (65, 80),
            'rsi_oversold': (20, 35),
            'macd_fast': (8, 14),
            'macd_slow': (21, 30),
            'macd_signal': (7, 12),
            'ema_short': (8, 15),
            'ema_long': (18, 30),
            'atr_multiplier_sl': (1.5, 3.0),
            'atr_multiplier_tp': (2.0, 4.0),
        },
        'rsi': {
            'period': (10, 20),
            'overbought': (65, 80),
            'oversold': (20, 35),
            'exit_threshold': (45, 55),
        },
        'macd': {
            'fast': (8, 14),
            'slow': (21, 30),
            'signal': (7, 12),
            'histogram_threshold': (0.5, 2.0),
        },
        'combo': {
            'rsi_weight': (0.2, 0.4),
            'macd_weight': (0.2, 0.4),
            'ma_weight': (0.2, 0.4),
            'rsi_period': (10, 20),
            'macd_fast': (8, 14),
            'macd_slow': (21, 30),
        }
    }
    
    return spaces.get(strategy_type, spaces['quant'])


if __name__ == '__main__':
    # Example usage
    print("Strategy Optimizer Example")
    print("=" * 60)
    
    # Sample data
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='1h')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Simple RSI strategy
    def rsi_strategy(data, rsi_period=14, overbought=70, oversold=30):
        # Pandas RSI implementation
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        entries = rsi < oversold
        exits = rsi > overbought
        return entries, exits
    
    # Optimize
    optimizer = StrategyOptimizer(
        data=data,
        strategy_func=rsi_strategy,
        n_trials=50
    )
    
    param_space = {
        'rsi_period': (10, 20),
        'overbought': (65, 75),
        'oversold': (25, 35)
    }
    
    best_params = optimizer.optimize(param_space, objective='sharpe')
    
    print(f"\nBest Parameters: {best_params}")
    
    # Get report
    report = optimizer.get_optimization_report()
    print(f"\nTop 5 Trials:")
    print(report.nlargest(5, 'value')[['params', 'value']])
