"""
XAKCN Trading Bot - Configuration
All settings can be overridden via environment variables.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _to_bool(name: str, default: bool) -> bool:
    """Parse boolean environment value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _to_int(name: str, default: int) -> int:
    """Parse integer environment value."""
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default
    return int(value)


def _to_float(name: str, default: float) -> float:
    """Parse float environment value."""
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default
    return float(value)


def _to_fraction(name: str, default: float) -> float:
    """
    Parse percentage/fraction values to fraction form.
    Accepts either 0.02 (fraction) or 2 (percentage).
    """
    value = os.getenv(name)
    if value is None or value.strip() == '':
        return default

    parsed = float(value)
    if parsed > 1.0:
        parsed /= 100.0
    return parsed


def _to_optional_str(name: str) -> Optional[str]:
    """Parse optional string environment value."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass
class BotConfig:
    """Main bot configuration."""
    
    # Trading symbol (Binance spot format, example: BTCUSDT)
    STOCK_CODE: str = os.getenv('STOCK_CODE', 'BTC')
    OPERATION_CODE: str = os.getenv('OPERATION_CODE', 'BTCUSDT')
    
    # Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
    CANDLE_PERIOD: str = os.getenv('CANDLE_PERIOD', '1h')
    
    # Quantity per trade (base asset amount)
    TRADED_QUANTITY: float = _to_float('TRADED_QUANTITY', 0.001)
    
    # Strategy to use (quant is the unified strategy)
    STRATEGY: str = os.getenv('STRATEGY', 'quant')
    
    # Cycle interval (seconds)
    CYCLE_INTERVAL: int = _to_int('CYCLE_INTERVAL', 60)
    
    # Test mode (simulate without executing)
    TEST_MODE: bool = _to_bool('TEST_MODE', True)
    
    # Minimum balance to trade
    MIN_BALANCE: float = _to_float('MIN_BALANCE', 100.0)

    # Account display currency
    ACCOUNT_CURRENCY: str = os.getenv('ACCOUNT_CURRENCY', 'USDT')

    # Binance spot API credentials
    BINANCE_API_KEY: Optional[str] = _to_optional_str('BINANCE_API_KEY')
    BINANCE_SECRET_KEY: Optional[str] = _to_optional_str('BINANCE_SECRET_KEY')
    BINANCE_TESTNET: bool = _to_bool('BINANCE_TESTNET', True)
    BINANCE_RECV_WINDOW: int = _to_int('BINANCE_RECV_WINDOW', 10000)
    
    def validate(self) -> bool:
        """Validate required settings."""
        if not self.OPERATION_CODE:
            raise ValueError("Trading symbol is required! Set OPERATION_CODE in .env")
        return True


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    
    # Moving Average
    MA_FAST_WINDOW: int = _to_int('MA_FAST_WINDOW', 10)
    MA_SLOW_WINDOW: int = _to_int('MA_SLOW_WINDOW', 20)
    
    # RSI
    RSI_PERIOD: int = _to_int('RSI_PERIOD', 14)
    RSI_OVERBOUGHT: float = _to_float('RSI_OVERBOUGHT', 70)
    RSI_OVERSOLD: float = _to_float('RSI_OVERSOLD', 30)
    
    # TSI
    TSI_FAST_PERIOD: int = _to_int('TSI_FAST_PERIOD', 13)
    TSI_SLOW_PERIOD: int = _to_int('TSI_SLOW_PERIOD', 25)
    TSI_SIGNAL_PERIOD: int = _to_int('TSI_SIGNAL_PERIOD', 13)
    
    # Stop Loss and Take Profit (%)
    STOP_LOSS_PERCENT: float = _to_float('STOP_LOSS_PERCENT', 3.0)
    TAKE_PROFIT_PERCENT: float = _to_float('TAKE_PROFIT_PERCENT', 6.0)
    
    # Quant Strategy Thresholds
    QUANT_BUY_THRESHOLD: float = _to_float('QUANT_BUY_THRESHOLD', 0.55)
    QUANT_SELL_THRESHOLD: float = _to_float('QUANT_SELL_THRESHOLD', -0.55)
    
    # Trailing Stop
    TRAILING_ACTIVATION: float = _to_float('TRAILING_ACTIVATION', 1.5)
    TRAILING_DISTANCE: float = _to_float('TRAILING_DISTANCE', 1.0)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Max risk per trade (%)
    MAX_RISK_PER_TRADE: float = _to_fraction('MAX_RISK_PER_TRADE', 0.02)
    
    # Max position size (%)
    MAX_POSITION_SIZE: float = _to_fraction('MAX_POSITION_SIZE', 0.20)
    
    # Max daily loss (%)
    MAX_DAILY_LOSS: float = _to_fraction('MAX_DAILY_LOSS', 0.05)
    
    # Max drawdown (%)
    MAX_DRAWDOWN: float = _to_fraction('MAX_DRAWDOWN', 0.15)
    
    # Max consecutive losses
    MAX_CONSECUTIVE_LOSSES: int = _to_int('MAX_CONSECUTIVE_LOSSES', 3)


# Global instances
config = BotConfig()
strategy_config = StrategyConfig()
risk_config = RiskConfig()
