"""
Adaptive Position Sizing
Kelly Criterion, volatility-based sizing, and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Position sizing result."""
    size: float  # Position size in base currency
    size_pct: float  # Position size as % of equity
    risk_amount: float  # Risk amount in quote currency
    stop_loss_price: float  # Stop loss price level
    take_profit_price: float  # Take profit price level
    leverage: float  # Recommended leverage (1.0 for spot)


class PositionSizer:
    """Calculate adaptive position sizes."""
    
    def __init__(self, 
                 max_risk_per_trade: float = 0.02,
                 max_position_size: float = 0.20,
                 default_leverage: float = 1.0):
        """
        Initialize position sizer.
        
        Args:
            max_risk_per_trade: Maximum risk per trade (2% default)
            max_position_size: Maximum position size (20% default)
            default_leverage: Default leverage (1.0 = spot)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.default_leverage = default_leverage
        
    def calculate_position_size(self,
                                 equity: float,
                                 entry_price: float,
                                 stop_loss_price: float,
                                 take_profit_price: float,
                                 atr: Optional[float] = None,
                                 volatility_regime: str = 'NORMAL') -> PositionSize:
        """
        Calculate position size based on risk.
        
        Args:
            equity: Total equity in quote currency
            entry_price: Planned entry price
            stop_loss_price: Stop loss price level
            take_profit_price: Take profit price level
            atr: Average True Range (optional)
            volatility_regime: 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
            
        Returns:
            PositionSize object
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            logger.warning("Stop loss too close to entry")
            risk_per_unit = entry_price * 0.01  # Default 1%
        
        # Adjust risk based on volatility regime
        risk_multiplier = self._get_risk_multiplier(volatility_regime)
        adjusted_risk_pct = self.max_risk_per_trade * risk_multiplier
        
        # Risk amount
        risk_amount = equity * adjusted_risk_pct
        
        # Position size in base currency
        position_size = risk_amount / risk_per_unit
        
        # Position value
        position_value = position_size * entry_price
        
        # Check max position size constraint
        max_position_value = equity * self.max_position_size
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            position_value = max_position_value
            risk_amount = position_size * risk_per_unit
        
        # Position size as % of equity
        position_pct = position_value / equity
        
        return PositionSize(
            size=position_size,
            size_pct=position_pct,
            risk_amount=risk_amount,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            leverage=self.default_leverage
        )
    
    def kelly_criterion(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float,
                       fraction: float = 0.25) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - [(1 - W) / R]
        W = win rate
        R = win/loss ratio
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            fraction: Fractional Kelly (0.25 = quarter Kelly)
            
        Returns:
            Recommended position size as % of equity
        """
        if avg_loss == 0:
            return 0
        
        R = avg_win / avg_loss
        kelly_pct = win_rate - ((1 - win_rate) / R)
        
        # Apply fractional Kelly and cap at max risk
        position_pct = max(0, min(kelly_pct * fraction, self.max_risk_per_trade))
        
        return position_pct
    
    def dynamic_position_size(self,
                             base_size: float,
                             atr: float,
                             atr_mean: float,
                             volatility_regime: str = 'NORMAL') -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            base_size: Base position size
            atr: Current ATR
            atr_mean: Historical mean ATR
            volatility_regime: Current volatility regime
            
        Returns:
            Adjusted position size
        """
        if atr_mean == 0:
            return base_size
        
        volatility_ratio = atr / atr_mean
        
        # Size reduction in high volatility
        if volatility_ratio > 2.0:
            return base_size * 0.5
        elif volatility_ratio > 1.5:
            return base_size * 0.75
        elif volatility_regime == 'HIGH':
            return base_size * 0.8
        elif volatility_regime == 'EXTREME':
            return base_size * 0.5
        elif volatility_regime == 'LOW':
            return base_size * 1.2
        
        return base_size
    
    def _get_risk_multiplier(self, regime: str) -> float:
        """Get risk multiplier for volatility regime."""
        multipliers = {
            'LOW': 1.2,
            'NORMAL': 1.0,
            'HIGH': 0.8,
            'EXTREME': 0.5
        }
        return multipliers.get(regime, 1.0)


class DynamicStops:
    """Calculate dynamic stop losses and take profits."""
    
    def __init__(self, 
                 base_sl_mult: float = 1.0,
                 base_tp_mult: float = 2.0,
                 use_regime_multipliers: bool = False):
        """
        Initialize dynamic stops.
        
        Args:
            base_sl_mult: Base ATR multiplier for stop loss
            base_tp_mult: Base ATR multiplier for take profit
            use_regime_multipliers: If True, adjust multipliers by regime
        """
        self.base_sl_mult = base_sl_mult
        self.base_tp_mult = base_tp_mult
        self.use_regime_multipliers = use_regime_multipliers
        
    def calculate_stops(self,
                       entry_price: float,
                       atr: float,
                       direction: str,
                       regime: str = 'NORMAL',
                       fixed_rr: Optional[bool] = None) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            direction: 'LONG' or 'SHORT'
            regime: Market regime
            fixed_rr: If True, force fixed ATR multipliers (ignores regime)
            
        Returns:
            (stop_loss_price, take_profit_price)
        """
        use_fixed = not self.use_regime_multipliers if fixed_rr is None else fixed_rr
        if use_fixed:
            sl_mult, tp_mult = self.base_sl_mult, self.base_tp_mult
        else:
            sl_mult, tp_mult = self._get_multipliers(regime)
        
        # Calculate distances
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        
        if direction == 'LONG':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return stop_loss, take_profit

    def calculate_atr_stop_fractions(
        self,
        close: pd.Series,
        atr: pd.Series,
        sl_mult: Optional[float] = None,
        tp_mult: Optional[float] = None,
        min_stop: float = 0.0001
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Convert ATR distances to VectorBT stop fractions.

        Args:
            close: Close price series
            atr: ATR series
            sl_mult: Stop loss ATR multiplier (defaults to base_sl_mult)
            tp_mult: Take profit ATR multiplier (defaults to base_tp_mult)
            min_stop: Minimum stop fraction to avoid zero/invalid values

        Returns:
            (sl_stop, tp_stop) as positive fractions
        """
        sl_multiplier = self.base_sl_mult if sl_mult is None else sl_mult
        tp_multiplier = self.base_tp_mult if tp_mult is None else tp_mult

        valid_close = close.replace(0, np.nan)
        sl_stop = (atr * sl_multiplier / valid_close).replace([np.inf, -np.inf], np.nan)
        tp_stop = (atr * tp_multiplier / valid_close).replace([np.inf, -np.inf], np.nan)

        sl_stop = sl_stop.clip(lower=min_stop, upper=0.95).fillna(min_stop)
        tp_stop = tp_stop.clip(lower=min_stop, upper=5.0).fillna(min_stop)
        return sl_stop, tp_stop
    
    def calculate_trailing_stop(self,
                               entry_price: float,
                               current_price: float,
                               highest_price: float,
                               atr: float,
                               activation_pct: float = 0.015,
                               trail_mult: float = 1.0) -> Optional[float]:
        """
        Calculate trailing stop level.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            highest_price: Highest price since entry
            atr: Average True Range
            activation_pct: Profit % needed to activate trailing stop
            trail_mult: ATR multiplier for trailing distance
            
        Returns:
            Trailing stop price or None if not activated
        """
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct < activation_pct:
            return None  # Not activated yet
        
        # Trail at 1 ATR from highest price
        trail_distance = atr * trail_mult
        trailing_stop = highest_price - trail_distance
        
        return trailing_stop
    
    def chandelier_exit(self,
                       high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       period: int = 22,
                       multiplier: float = 3.0) -> pd.Series:
        """
        Calculate Chandelier Exit levels.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period
            multiplier: ATR multiplier
            
        Returns:
            Series with exit levels
        """
        from utils.enhanced_indicators import EnhancedIndicators
        
        # Create temporary dataframe for ATR calculation
        temp_df = pd.DataFrame({'high': high, 'low': low, 'close': close})
        atr = EnhancedIndicators.atr(temp_df, period)
        highest_high = high.rolling(window=period).max()
        
        chandelier = highest_high - (multiplier * atr)
        
        return chandelier
    
    def _get_multipliers(self, regime: str) -> Tuple[float, float]:
        """Get stop loss and take profit multipliers for regime."""
        multipliers = {
            'TRENDING_UP': (2.0, 4.0),
            'TRENDING_DOWN': (2.0, 4.0),
            'RANGING': (1.5, 2.5),
            'HIGH_VOLATILITY': (3.0, 5.0),
            'LOW_VOLATILITY': (1.5, 2.5),
            'NORMAL': (self.base_sl_mult, self.base_tp_mult)
        }
        return multipliers.get(regime, (self.base_sl_mult, self.base_tp_mult))


class CircuitBreaker:
    """Risk management circuit breakers."""
    
    def __init__(self,
                 max_daily_loss: float = 0.05,
                 max_consecutive_losses: int = 3,
                 max_drawdown: float = 0.15):
        """
        Initialize circuit breaker.
        
        Args:
            max_daily_loss: Maximum daily loss % before pausing
            max_consecutive_losses: Max consecutive losing trades
            max_drawdown: Maximum drawdown % before pausing
        """
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown = max_drawdown
        
        # State tracking
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.peak_equity = 0
        self.current_equity = 0
        self.is_paused = False
        self.pause_reason = None
        
    def update(self, trade_pnl: float, current_equity: float):
        """
        Update circuit breaker state with new trade.
        
        Args:
            trade_pnl: P&L from last trade
            current_equity: Current equity value
        """
        self.current_equity = current_equity
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Update daily P&L
        self.daily_pnl += trade_pnl
        
        # Update consecutive losses
        if trade_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check circuit breakers
        self._check_circuit_breakers()
    
    def _check_circuit_breakers(self):
        """Check if any circuit breaker should trigger."""
        # Check daily loss
        daily_loss_pct = abs(self.daily_pnl) / self.peak_equity if self.peak_equity > 0 else 0
        
        if daily_loss_pct >= self.max_daily_loss:
            self.is_paused = True
            self.pause_reason = f"Daily loss limit reached: {daily_loss_pct:.2%}"
            logger.warning(f"CIRCUIT BREAKER: {self.pause_reason}")
            return
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_paused = True
            self.pause_reason = f"{self.consecutive_losses} consecutive losses"
            logger.warning(f"CIRCUIT BREAKER: {self.pause_reason}")
            return
        
        # Check drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        if drawdown >= self.max_drawdown:
            self.is_paused = True
            self.pause_reason = f"Max drawdown reached: {drawdown:.2%}"
            logger.warning(f"CIRCUIT BREAKER: {self.pause_reason}")
            return
    
    def check_can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            (can_trade, reason)
        """
        if self.is_paused:
            return False, self.pause_reason
        return True, "OK"
    
    def reset_daily(self):
        """Reset daily stats (call at start of new day)."""
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.is_paused = False
        self.pause_reason = None
        
    def reset_all(self):
        """Reset all stats."""
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.peak_equity = self.current_equity
        self.is_paused = False
        self.pause_reason = None


if __name__ == '__main__':
    print("Position Sizing Test")
    print("=" * 60)
    
    # Test position sizer
    sizer = PositionSizer(max_risk_per_trade=0.02, max_position_size=0.20)
    
    equity = 10000
    entry = 50000
    stop = 49000
    tp = 52000
    atr = 500
    
    pos = sizer.calculate_position_size(
        equity=equity,
        entry_price=entry,
        stop_loss_price=stop,
        take_profit_price=tp,
        atr=atr,
        volatility_regime='NORMAL'
    )
    
    print(f"Entry: ${entry:,.2f}")
    print(f"Stop: ${stop:,.2f}")
    print(f"Take Profit: ${tp:,.2f}")
    print(f"\nPosition Size: {pos.size:.6f} BTC")
    print(f"Position Value: ${pos.size * entry:,.2f}")
    print(f"Position %: {pos.size_pct:.2%}")
    print(f"Risk Amount: ${pos.risk_amount:,.2f}")
    print(f"Risk/Reward: 1:{abs(tp - entry) / abs(entry - stop):.1f}")
    
    # Test Kelly Criterion
    kelly_size = sizer.kelly_criterion(
        win_rate=0.55,
        avg_win=0.03,
        avg_loss=0.02
    )
    print(f"\nKelly Position Size: {kelly_size:.2%}")
    
    # Test dynamic stops
    stops = DynamicStops()
    sl, tp = stops.calculate_stops(entry, atr, 'LONG', 'TRENDING_UP')
    print(f"\nDynamic Stops (Trending):")
    print(f"  Stop Loss: ${sl:,.2f}")
    print(f"  Take Profit: ${tp:,.2f}")
    
    # Test circuit breaker
    cb = CircuitBreaker()
    
    # Simulate trades
    trades = [100, -50, -100, -200, 50]  # 3 consecutive losses
    
    for pnl in trades:
        equity += pnl
        cb.update(pnl, equity)
        can_trade, reason = cb.check_can_trade()
        print(f"\nTrade P&L: ${pnl}, Equity: ${equity}")
        print(f"Can Trade: {can_trade}, Reason: {reason}")
