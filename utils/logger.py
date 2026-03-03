"""
Trading Bot Logging System - Professional Visual Formatting.
No emojis, no accented characters.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any


class TradingLogger:
    """Custom logger for trading operations - PROFESSIONAL."""
    
    # Box drawing characters (ASCII only)
    BOX = {
        'h': '-', 'v': '|', 'tl': '+', 'tr': '+', 
        'bl': '+', 'br': '+', 'ml': '+', 'mr': '+',
        'tm': '+', 'bm': '+', 'mm': '+',
        'd_h': '=', 'd_v': '|', 'd_tl': '+', 'd_tr': '+',
        'd_bl': '+', 'd_br': '+', 'd_ml': '+', 'd_mr': '+'
    }
    
    # Symbols (NO EMOJIS)
    SYMBOLS = {
        'buy': '[B]', 'sell': '[S]', 'hold': '[H]',
        'up': '/\\', 'down': '\\/', 'check': 'OK', 'cross': 'XX',
    }
    
    def __init__(self, name: str = "TradingBot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Create logs directory
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(log_dir, 'trading_bot.log'),
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def _pad(self, text: str, width: int) -> str:
        """Pad text."""
        if len(text) >= width:
            return text[:width]
        return text + ' ' * (width - len(text))
    
    # =============================================================================
    # CYCLE LOGS
    # =============================================================================
    
    def log_cycle_start(self, symbol: str, balance: float, position: str = "NONE"):
        """Log cycle start."""
        b = self.BOX
        w = 80
        
        pos_indicator = f"[{position}]" if position else "[NONE]"
        
        line1 = "  [NEW CYCLE]"
        line1_pad = ' ' * (w - 2 - len(line1))
        
        line2 = f"  Asset:    {symbol:<12} | Position: {pos_indicator:<15}"
        line2_pad = ' ' * (w - 2 - len(line2))
        
        line3 = f"  Balance:  $ {balance:>12,.2f}"
        line3_pad = ' ' * (w - 2 - len(line3))
        
        message = f"""
{b['d_tl']}{b['d_h'] * (w-2)}{b['d_tr']}
{b['d_v']}{line1}{line1_pad}{b['d_v']}
{b['ml']}{b['h'] * (w-2)}{b['mr']}
{b['v']}{line2}{line2_pad}{b['v']}
{b['v']}{line3}{line3_pad}{b['v']}
{b['d_bl']}{b['d_h'] * (w-2)}{b['d_br']}"""
        self.info(message)
    
    # =============================================================================
    # ORDER LOGS
    # =============================================================================
    
    def log_order(self, order: Dict[str, Any], test_mode: bool = True):
        """Log executed order."""
        b = self.BOX
        w = 80
        s = self.SYMBOLS
        
        try:
            side = order.get('side', 'N/A')
            symbol = order.get('symbol', 'N/A')
            qty = float(order.get('executedQty', 0) or order.get('origQty', 0))
            total = float(order.get('cummulativeQuoteQty', 0))
            price = total / qty if qty > 0 else 0
            
            is_buy = side.upper() == 'BUY'
            symbol_icon = s['buy'] if is_buy else s['sell']
            action = "BUY" if is_buy else "SELL"
            mode_str = "TEST" if test_mode else "LIVE"
            
            lines = [
                "",
                f"{b['d_tl']}{b['d_h'] * (w-2)}{b['d_tr']}",
                f"{b['d_v']}  {symbol_icon} {mode_str} ORDER - {action}{' ' * (55 - len(action))}{b['d_v']}",
                f"{b['ml']}{b['h'] * (w-2)}{b['mr']}",
                f"{b['v']}  Asset:      {symbol:<20} Status: {order.get('status', 'FILLED')}{b['v']}",
                f"{b['v']}  Quantity:   {qty:>15.6f}{b['v']}",
                f"{b['v']}  Price:      $ {price:>15.2f}{b['v']}",
                f"{b['v']}  Total:      $ {total:>15.2f}{b['v']}",
                f"{b['d_bl']}{b['d_h'] * (w-2)}{b['d_br']}",
            ]
            
            self.info('\n'.join(lines))
            
        except Exception as e:
            self.error(f"Error logging order: {e}")
    
    def log_position_opened(self, signal: str, price: float, size: float, 
                           stop_loss: float, take_profit: float):
        """Log position opened."""
        b = self.BOX
        w = 80
        s = self.SYMBOLS
        
        is_buy = 'BUY' in signal
        icon = s['buy'] if is_buy else s['sell']
        action = 'LONG' if is_buy else 'SHORT'
        
        sl_pct = abs((stop_loss / price - 1) * 100)
        tp_pct = abs((take_profit / price - 1) * 100)
        
        lines = [
            "",
            f"{b['d_tl']}{b['d_h'] * (w-2)}{b['d_tr']}",
            f"{b['d_v']}  {icon} POSITION {action} OPENED{' ' * (56 - len(action))}{b['d_v']}",
            f"{b['ml']}{b['h'] * (w-2)}{b['mr']}",
            f"{b['v']}  Entry Price:    $ {price:>15.2f}{b['v']}",
            f"{b['v']}  Quantity:       {size:>15.6f}{b['v']}",
            f"{b['v']}  Stop Loss:      $ {stop_loss:>15.2f}  ({sl_pct:.1f}%){b['v']}",
            f"{b['v']}  Take Profit:    $ {take_profit:>15.2f}  ({tp_pct:.1f}%){b['v']}",
            f"{b['d_bl']}{b['d_h'] * (w-2)}{b['d_br']}",
        ]
        
        self.info('\n'.join(lines))
    
    def log_position_closed(self, pnl: float, pnl_pct: float, reason: str):
        """Log position closed."""
        b = self.BOX
        w = 80
        s = self.SYMBOLS
        
        is_profit = pnl >= 0
        icon = s['up'] if is_profit else s['down']
        result_str = "PROFIT" if is_profit else "LOSS"
        
        lines = [
            "",
            f"{b['d_tl']}{b['d_h'] * (w-2)}{b['d_tr']}",
            f"{b['d_v']}  {icon} POSITION CLOSED - {result_str}{' ' * (51 - len(result_str))}{b['d_v']}",
            f"{b['ml']}{b['h'] * (w-2)}{b['mr']}",
            f"{b['v']}  Reason:         {reason}{b['v']}",
            f"{b['v']}  P&L:            $ {pnl:>15.2f}  ({pnl_pct:+.2f}%){b['v']}",
            f"{b['d_bl']}{b['d_h'] * (w-2)}{b['d_br']}",
        ]
        
        self.info('\n'.join(lines))
    
    # =============================================================================
    # ERROR LOGS
    # =============================================================================
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error."""
        ctx = f" in {context}" if context else ""
        self.logger.error(f"[ERROR]{ctx}: {str(error)}")
    
    def log_warning(self, message: str):
        """Log warning."""
        self.logger.warning(f"[WARNING] {message}")
    
    def log_info(self, message: str):
        """Log info."""
        self.logger.info(f"[INFO] {message}")
    
    def log_success(self, message: str):
        """Log success."""
        self.logger.info(f"[OK] {message}")


# Global instance
logger = TradingLogger()
