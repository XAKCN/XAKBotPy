"""
Backtesting Engine using VectorBT
Fast vectorized backtesting with comprehensive metrics.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Callable, Optional, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_name: str
    symbol: str
    timeframe: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    avg_trade_duration: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'profit_factor': self.profit_factor,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'avg_trade_duration': self.avg_trade_duration,
            'total_trades': self.total_trades,
        }
    
    def print_summary(self):
        """Print formatted results using visual_logger."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils.visual_logger import visual
        
        results = {
            'total_return': self.total_return * 100,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate * 100,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'profit_factor': self.profit_factor,
        }
        
        visual.print_backtest_results(results)


class BacktestEngine:
    """VectorBT-based backtesting engine."""
    
    def __init__(self, 
                 initial_cash: float = 10000.0,
                 fees: float = 0.001,
                 slippage: float = 0.0005,
                 freq: str = '1h'):
        """
        Initialize backtest engine.
        
        Args:
            initial_cash: Starting capital
            fees: Trading fees (0.001 = 0.1%)
            slippage: Slippage per trade (0.0005 = 0.05%)
            freq: Data frequency for annualization
        """
        self.initial_cash = initial_cash
        self.fees = fees
        self.slippage = slippage
        self.freq = freq
        
    def run_backtest(self,
                     data: pd.DataFrame,
                     entries: pd.Series,
                     exits: pd.Series,
                     short_entries: Optional[pd.Series] = None,
                     short_exits: Optional[pd.Series] = None,
                     strategy_name: str = "Strategy",
                     symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run vectorized backtest.
        
        Args:
            data: DataFrame with OHLCV data
            entries: Boolean Series for entry signals
            exits: Boolean Series for exit signals
            short_entries: Boolean Series for short entry signals (optional)
            short_exits: Boolean Series for short exit signals (optional)
            strategy_name: Name of the strategy
            symbol: Trading symbol
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        logger.info(f"Running backtest for {strategy_name} on {symbol}")
        
        # Run portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq
        )
        
        # Extract metrics
        stats = portfolio.stats()
        
        # Calculate additional metrics
        result = self._create_result(
            portfolio=portfolio,
            stats=stats,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=self.freq
        )
        
        return result
    
    def run_with_position_size(self,
                               data: pd.DataFrame,
                               entries: pd.Series,
                               exits: pd.Series,
                               size: pd.Series,
                               strategy_name: str = "Strategy",
                               symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run backtest with dynamic position sizing.
        
        Args:
            data: DataFrame with OHLCV data
            entries: Boolean Series for entry signals
            exits: Boolean Series for exit signals
            size: Position size as fraction of equity (0-1)
            strategy_name: Name of the strategy
            symbol: Trading symbol
        """
        portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            size=size,
            init_cash=self.initial_cash,
            fees=self.fees,
            slippage=self.slippage,
            freq=self.freq
        )
        
        stats = portfolio.stats()
        
        return self._create_result(
            portfolio=portfolio,
            stats=stats,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=self.freq
        )
    
    def run_with_stops(self,
                       data: pd.DataFrame,
                       entries: pd.Series,
                       exits: pd.Series,
                       sl_stop: Optional[pd.Series] = None,
                       tp_stop: Optional[pd.Series] = None,
                       tsl_stop: Optional[pd.Series] = None,
                       strategy_name: str = "Strategy",
                       symbol: str = "BTCUSDT") -> BacktestResult:
        """
        Run backtest with stop losses and take profits.
        
        Args:
            data: DataFrame with OHLCV data
            entries: Boolean Series for entry signals
            exits: Boolean Series for exit signals
            sl_stop: Stop loss as fraction (e.g., 0.02 for 2%)
            tp_stop: Take profit as fraction
            tsl_stop: Trailing stop as fraction
        """
        signal_kwargs = {
            'close': data['close'],
            'entries': entries,
            'exits': exits,
            'init_cash': self.initial_cash,
            'fees': self.fees,
            'slippage': self.slippage,
            'freq': self.freq,
        }
        if sl_stop is not None:
            signal_kwargs['sl_stop'] = sl_stop
        if tp_stop is not None:
            signal_kwargs['tp_stop'] = tp_stop
        if tsl_stop is not None:
            signal_kwargs['tsl_stop'] = tsl_stop

        try:
            portfolio = vbt.Portfolio.from_signals(**signal_kwargs)
        except TypeError as error:
            # Older VectorBT builds may not support tsl_stop.
            if 'tsl_stop' in signal_kwargs:
                logger.warning("VectorBT does not support tsl_stop in this environment. Retrying without it.")
                signal_kwargs.pop('tsl_stop', None)
                portfolio = vbt.Portfolio.from_signals(**signal_kwargs)
            else:
                raise error
        
        stats = portfolio.stats()
        
        return self._create_result(
            portfolio=portfolio,
            stats=stats,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=self.freq
        )
    
    def _create_result(self,
                       portfolio: vbt.Portfolio,
                       stats: pd.Series,
                       strategy_name: str,
                       symbol: str,
                       timeframe: str) -> BacktestResult:
        """Create BacktestResult from portfolio."""
        
        # Get trades
        trades = portfolio.trades
        
        # Calculate metrics
        total_return = stats.get('Total Return [%]', 0) / 100
        sharpe_ratio = stats.get('Sharpe Ratio', 0)
        sortino_ratio = stats.get('Sortino Ratio', 0)
        calmar_ratio = stats.get('Calmar Ratio', 0)
        profit_factor = self._calculate_profit_factor(trades)
        win_rate = stats.get('Win Rate [%]', 0) / 100
        max_drawdown = stats.get('Max Drawdown [%]', 0) / 100
        avg_trade_duration = stats.get('Avg Winning Trade Duration', pd.Timedelta(0)).total_seconds() / 3600
        total_trades = stats.get('Total Trades', 0)
        
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            avg_trade_duration=avg_trade_duration,
            total_trades=total_trades,
            equity_curve=portfolio.value(),
            trades=trades.records if hasattr(trades, 'records') else pd.DataFrame(),
            metrics=stats.to_dict()
        )
    
    def _calculate_profit_factor(self, trades) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            if hasattr(trades, 'returns'):
                returns = trades.returns.values
                gains = returns[returns > 0].sum()
                losses = abs(returns[returns < 0].sum())
                
                if losses == 0:
                    return float('inf') if gains > 0 else 0
                return gains / losses
        except Exception:
            pass
        return 0
    
    def walk_forward_optimization(self,
                                   data: pd.DataFrame,
                                   strategy_func: Callable,
                                   param_grid: Dict,
                                   train_size: float = 0.7,
                                   n_splits: int = 5) -> Dict:
        """
        Perform walk-forward optimization.
        
        Args:
            data: Full dataset
            strategy_func: Function that takes data and params, returns (entries, exits)
            param_grid: Dictionary of parameter ranges
            train_size: Fraction of data for training
            n_splits: Number of walk-forward splits
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        results = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(data)):
            logger.info(f"Walk-Forward Fold {fold + 1}/{n_splits}")
            
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Optimize on training data
            best_params = self._optimize_params(
                train_data, 
                strategy_func, 
                param_grid
            )
            
            # Test on out-of-sample data
            entries, exits = strategy_func(test_data, **best_params)
            
            result = self.run_backtest(
                test_data,
                entries,
                exits,
                strategy_name=f"WFO_Fold_{fold + 1}"
            )
            
            results.append({
                'fold': fold + 1,
                'params': best_params,
                'result': result
            })
        
        # Aggregate results
        avg_sharpe = np.mean([r['result'].sharpe_ratio for r in results])
        avg_return = np.mean([r['result'].total_return for r in results])
        
        logger.info(f"Walk-Forward Average Sharpe: {avg_sharpe:.2f}")
        logger.info(f"Walk-Forward Average Return: {avg_return:.2%}")
        
        return {
            'folds': results,
            'avg_sharpe': avg_sharpe,
            'avg_return': avg_return
        }
    
    def _optimize_params(self,
                         data: pd.DataFrame,
                         strategy_func: Callable,
                         param_grid: Dict) -> Dict:
        """Simple grid search for parameter optimization."""
        import itertools
        
        best_sharpe = -np.inf
        best_params = {}
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            try:
                entries, exits = strategy_func(data, **params)
                
                # Quick backtest
                pf = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=self.initial_cash,
                    fees=self.fees,
                    freq=self.freq
                )
                
                sharpe = pf.sharpe_ratio()
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
                continue
        
        logger.info(f"Best params: {best_params} (Sharpe: {best_sharpe:.2f})")
        return best_params


class StrategyWrapper:
    """Wrapper to convert strategy classes to VectorBT signals."""
    
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate entry/exit signals from strategy.
        
        Returns:
            (entries, exits) as boolean Series
        """
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        
        # Iterate through data (simulating real-time)
        for i in range(len(data)):
            if i < 50:  # Warm-up period for indicators
                continue
                
            current_data = data.iloc[:i+1]
            
            # Get strategy signal
            result = self.strategy.analyze(current_data)
            decision = result.get('decision', 'HOLD')
            
            # Generate signal
            if decision == 'BUY':
                entries.iloc[i] = True
            elif decision == 'SELL':
                exits.iloc[i] = True
        
        return entries, exits


if __name__ == '__main__':
    # Example usage
    print("Backtest Engine Example")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='1h')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Simple strategy: Buy when price > SMA(20)
    sma = data['close'].rolling(20).mean()
    entries = data['close'] > sma
    exits = data['close'] < sma
    
    # Run backtest
    engine = BacktestEngine(initial_cash=10000, fees=0.001)
    result = engine.run_backtest(data, entries, exits, strategy_name="SMA_Cross")
    
    result.print_summary()
