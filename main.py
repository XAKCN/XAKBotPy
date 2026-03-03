#!/usr/bin/env python3
"""
XAKCN Trading Bot - Unified Edition
Complete quantitative trading system with ML, ensemble scoring, and adaptive risk management.

Usage:
    python main.py --mode demo          # Demo mode with all strategies
    python main.py --mode trade         # Live trading (test mode default)
    python main.py --mode train         # Train ML model
    python main.py --mode backtest      # Run backtest
    python main.py --mode optimize      # Optimize parameters

XAKCN LLC - 2026
"""

import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from config.settings import config, risk_config
from exchange.binance_spot_client import BinanceSpotTrader, HAS_BINANCE
from utils.visual_logger import visual
from utils.enhanced_indicators import EnhancedIndicators
from backtest.ensemble_scoring import EnsembleScorer, EnsembleWeights
from ml.feature_engineering import FeatureEngineer
from ml.model_training import XGBoostTrainer, ModelInference
from filters.regime_detection import SimpleRegimeDetector, VolatilityRegime
from filters.position_sizing import PositionSizer, DynamicStops, CircuitBreaker


@dataclass
class TradingConfig:
    """Trading configuration."""
    symbol: str = config.OPERATION_CODE
    timeframe: str = config.CANDLE_PERIOD
    initial_capital: float = 10000.0
    max_risk_per_trade: float = risk_config.MAX_RISK_PER_TRADE
    test_mode: bool = config.TEST_MODE
    use_ml: bool = True
    use_ensemble: bool = True
    cycle_interval: int = config.CYCLE_INTERVAL
    max_cycles: int = 0
    optimize_trials: int = 50
    trend_filter: bool = False
    trend_ema_fast: int = 20
    trend_ema_slow: int = 50
    trend_adx_threshold: float = 22.0
    volume_spike_filter: bool = False
    volume_spike_window: int = 20
    volume_spike_multiplier: float = 1.2
    fixed_atr_rr: bool = True
    ignore_exit_signals: bool = True
    atr_sl_mult: float = 1.0
    atr_tp_mult: float = 2.0
    atr_period: int = 14
    binance_api_key: Optional[str] = config.BINANCE_API_KEY
    binance_secret_key: Optional[str] = config.BINANCE_SECRET_KEY
    binance_testnet: bool = config.BINANCE_TESTNET
    binance_recv_window: int = config.BINANCE_RECV_WINDOW


class UnifiedTradingBot:
    """
    Unified trading bot combining all features.
    """
    
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg
        self.running = False
        self.cycle_count = 0
        
        # Components
        self.exchange: Optional[BinanceSpotTrader] = None
        self.scorer: Optional[EnsembleScorer] = None
        self.ml_model: Optional[ModelInference] = None
        self.regime_detector = SimpleRegimeDetector()
        self.volatility_regime = VolatilityRegime()
        self.position_sizer = PositionSizer(cfg.max_risk_per_trade)
        self.dynamic_stops = DynamicStops()
        self.circuit_breaker = CircuitBreaker()
        self.feature_engineer = FeatureEngineer()
        self.ei = EnhancedIndicators()
        
        # State
        self.current_regime = 'UNKNOWN'
        self.equity = cfg.initial_capital
        self.position = None
        self.trades = []
        self.virtual_quote_balance = float(cfg.initial_capital)
        self.virtual_base_balance = 0.0
        
        # Data cache for incremental updates
        self._data_cache: Optional[pd.DataFrame] = None
        self._first_fetch = True
        self._last_data_signature: Optional[Tuple[int, str, float, float]] = None
        self._stale_data_cycles = 0
        self._stale_fallback_done = False
        self._active_testnet = bool(cfg.binance_testnet)

    def _update_virtual_equity(self, current_price: Optional[float] = None):
        """Update simulated equity for test/demo wallet."""
        if current_price is None or current_price <= 0:
            self.equity = float(self.virtual_quote_balance)
            return
        self.equity = float(self.virtual_quote_balance + (self.virtual_base_balance * current_price))

    def _log_virtual_wallet(self):
        """Log virtual wallet balances."""
        logger.info(
            "[TEST] Carteira demo | USDT: %.2f | %s: %.6f | Equity: %.2f",
            self.virtual_quote_balance,
            self.cfg.symbol.replace("USDT", ""),
            self.virtual_base_balance,
            self.equity
        )

    def _compute_market_stats(self, data: Optional[pd.DataFrame]) -> Dict[str, float | str]:
        """
        Compute market variation and range aligned to selected timeframe candle.
        - VAR timeframe: close[-1] vs close[-2]
        - MAX/MIN timeframe: high/low of latest candle
        """
        label = str(self.cfg.timeframe).upper()
        stats: Dict[str, float | str] = {
            "variation_label": label,
            "change_pct": 0.0,
            "high": 0.0,
            "low": 0.0,
        }

        if data is None or data.empty:
            return stats

        close_last = float(data["close"].iloc[-1])
        if len(data) >= 2:
            close_prev = float(data["close"].iloc[-2])
            if close_prev > 0:
                stats["change_pct"] = ((close_last / close_prev) - 1.0) * 100.0

        if "high" in data.columns and "low" in data.columns:
            high_last = float(data["high"].iloc[-1])
            low_last = float(data["low"].iloc[-1])
            stats["high"] = max(high_last, low_last)
            stats["low"] = min(high_last, low_last)
        else:
            stats["high"] = close_last
            stats["low"] = close_last

        return stats

    def _connect_exchange(self, testnet: Optional[bool] = None) -> bool:
        """Create/recreate Binance spot client using selected endpoint."""
        target_testnet = self._active_testnet if testnet is None else bool(testnet)
        self.exchange = BinanceSpotTrader(
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            api_key=self.cfg.binance_api_key,
            api_secret=self.cfg.binance_secret_key,
            testnet=target_testnet,
            recv_window=self.cfg.binance_recv_window
        )
        if not self.exchange.initialized:
            return False
        self._active_testnet = target_testnet
        logger.info(
            "Fonte de dados Binance: %s",
            "testnet" if target_testnet else "live"
        )
        return True

    def _build_data_signature(self, data: pd.DataFrame) -> Tuple[int, str, float, float]:
        """Build compact signature to detect repeated market snapshots."""
        if data is None or data.empty:
            return (0, "", 0.0, 0.0)
        last_idx = data.index[-1]
        last_time = pd.Timestamp(last_idx).isoformat()
        last_close = float(data["close"].iloc[-1])
        last_volume = float(data["volume"].iloc[-1])
        return (len(data), last_time, round(last_close, 8), round(last_volume, 8))

    def _register_data_snapshot(self, data: pd.DataFrame):
        """Track data freshness by comparing latest candle signature."""
        signature = self._build_data_signature(data)
        if signature == self._last_data_signature:
            self._stale_data_cycles += 1
        else:
            self._stale_data_cycles = 0
            self._last_data_signature = signature

        if data is None or data.empty:
            return

        last_time = data.index[-1]
        last_close = float(data["close"].iloc[-1])
        logger.info(
            "Dados ciclo | candle=%s | close=%.2f | repeticoes=%s",
            pd.Timestamp(last_time).strftime("%Y-%m-%d %H:%M:%S"),
            last_close,
            self._stale_data_cycles
        )

    def _maybe_fallback_to_live_data(self) -> bool:
        """
        In TEST_MODE, testnet candles can become stale for many cycles.
        Fallback to live public endpoint to keep autopilot reactive.
        """
        if self._stale_fallback_done:
            return False
        if not self.cfg.test_mode:
            return False
        if not self._active_testnet:
            return False
        if self._stale_data_cycles < 3:
            return False

        logger.warning(
            "Dados estagnados por %s ciclos no testnet. "
            "Alternando para endpoint live para manter atualizacao por ciclo.",
            self._stale_data_cycles
        )
        if not self._connect_exchange(testnet=False):
            logger.error("Falha ao alternar para endpoint live. Mantendo testnet.")
            return False

        self._stale_fallback_done = True
        self._last_data_signature = None
        self._stale_data_cycles = 0
        return True
        
    def initialize(self):
        """Initialize all components."""
        logger.info("=" * 70)
        logger.info("XAKCN TRADING BOT - UNIFIED")
        logger.info("=" * 70)
        logger.info(f"Ativo: {self.cfg.symbol}")
        logger.info(f"Timeframe: {self.cfg.timeframe}")
        logger.info(f"Modo: {'TESTE' if self.cfg.test_mode else 'LIVE'}")
        logger.info(f"ML: {'Ativo' if self.cfg.use_ml else 'Inativo'}")
        logger.info(f"Ensemble: {'Ativo' if self.cfg.use_ensemble else 'Inativo'}")
        logger.info("=" * 70)
        
        # Initialize Binance Spot connection
        try:
            if not HAS_BINANCE:
                raise ImportError("python-binance package not installed")

            if self._connect_exchange(testnet=self.cfg.binance_testnet):
                logger.info("[OK] Binance Spot connected")
                if not self.cfg.test_mode:
                    account = self.exchange.get_account_info() if self.exchange else None
                    if account:
                        self.equity = account['equity']
                        logger.info(f"[OK] Account: {account['balance']:.2f} {account['currency']}")
                else:
                    self._update_virtual_equity()
                    logger.info("[OK] Carteira demo iniciada com saldo ficticio")
                    self._log_virtual_wallet()
            else:
                logger.error("[X] Binance Spot connection failed")
                if not self.cfg.test_mode:
                    logger.info("Falling back to test mode")
                    self.cfg.test_mode = True
        except Exception as e:
            logger.error(f"Failed to connect Binance Spot: {e}")
            if not self.cfg.test_mode:
                logger.info("Falling back to test mode")
                self.cfg.test_mode = True
        
        # Initialize ensemble scorer
        if self.cfg.use_ensemble:
            self.scorer = EnsembleScorer()
            logger.info("[OK] Ensemble scorer initialized")
        
        # Load ML model
        if self.cfg.use_ml:
            self._load_ml_model()
        
        logger.info("[OK] Bot inicializado com sucesso")
        logger.info("=" * 70)
        
    def _load_ml_model(self):
        """Load ML model if available."""
        model_path = Path('ml/models/xgboost_latest.pkl')
        
        if model_path.exists():
            try:
                self.ml_model = ModelInference(str(model_path))
                logger.info(f"[OK] ML model loaded")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
                self.cfg.use_ml = False
        else:
            logger.warning("No ML model found. Train with: python main.py --mode train")
            self.cfg.use_ml = False
    
    def fetch_data(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch market data from Binance Spot."""
        if not HAS_BINANCE:
            logger.error("python-binance not available. Install with: pip install python-binance")
            return None

        try:
            if self.exchange is None:
                logger.info("Connecting to Binance Spot...")
                if not self._connect_exchange(testnet=self.cfg.binance_testnet):
                    logger.error("Failed to initialize Binance Spot client.")
                    return None

            if not self.cfg.test_mode:
                account_info = self.exchange.get_account_info()
                if account_info:
                    self.equity = account_info['equity']
                    logger.info(
                        "Account Balance: $%.2f | Equity: $%.2f",
                        account_info['balance'],
                        account_info['equity']
                    )

            data = self.exchange.get_market_data(limit=limit)
            if data is None:
                return None

            self._register_data_snapshot(data)

            if self._maybe_fallback_to_live_data():
                if self.exchange is None:
                    return data
                refreshed = self.exchange.get_market_data(limit=limit)
                if refreshed is not None:
                    data = refreshed
                    self._register_data_snapshot(data)

            if data is not None:
                logger.info(f"[OK] Binance spot data: {len(data)} candles")
            return data
        except Exception as e:
            logger.error(f"Error fetching Binance spot data: {e}")
            return None
    
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Comprehensive market analysis."""
        result = {
            'timestamp': datetime.now(),
            'price': data['close'].iloc[-1],
            'regime': 'UNKNOWN',
            'signal': 'HOLD',
            'confidence': 'LOW',
            'score': 0.0,
            'details': {}
        }
        
        # 1. Regime Detection
        self.current_regime = self.regime_detector.detect_regime(data)
        result['regime'] = self.current_regime
        
        # 2. Volatility Assessment
        vol_ratio = self.volatility_regime.get_volatility_ratio(data)
        vol_regime = self.volatility_regime.get_regime(data)
        result['volatility_ratio'] = vol_ratio
        result['volatility_regime'] = vol_regime
        
        # 3. Ensemble Scoring
        if self.cfg.use_ensemble and self.scorer:
            decision, confidence, score, components = self.scorer.calculate_ensemble_score(data)
            confluence, confirmations = self.scorer.check_confluence(data)
            
            result['ensemble'] = {
                'decision': decision,
                'confidence': confidence,
                'score': score,
                'components': components,
                'confluence': confluence,
                'confirmations': confirmations
            }
        
        # 4. ML Prediction
        if self.cfg.use_ml and self.ml_model:
            try:
                features = self.feature_engineer.create_features(data)
                latest_features = features.iloc[[-1]]
                
                pred, prob = self.ml_model.predict(latest_features)
                
                result['ml'] = {
                    'prediction': pred,
                    'probability': prob,
                    'confidence': abs(prob - 0.5) * 2
                }
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # 5. Combine Signals
        result['signal'], result['confidence'], result['score'] = self._combine_signals(result)
        
        return result
    
    def _combine_signals(self, analysis: Dict) -> Tuple[str, str, float]:
        """Combine TA and ML signals."""
        ensemble_score = float(analysis.get('ensemble', {}).get('score', 0.0))
        ml_probability = float(analysis.get('ml', {}).get('probability', 0.5))

        # Convert ML probability [0,1] to signed score [-1,1].
        ml_score = (ml_probability - 0.5) * 2.0

        # Keep final score in signed space so both BUY and SELL thresholds work.
        if self.cfg.use_ml and 'ml' in analysis:
            combined_score = 0.6 * ensemble_score + 0.4 * ml_score
        else:
            combined_score = ensemble_score

        if combined_score > 0.75:
            return 'STRONG_BUY', 'HIGH', combined_score
        elif combined_score > 0.55:
            return 'BUY', 'MEDIUM', combined_score
        elif combined_score > 0.30:
            return 'WEAK_BUY', 'LOW', combined_score
        elif combined_score < -0.75:
            return 'STRONG_SELL', 'HIGH', combined_score
        elif combined_score < -0.55:
            return 'SELL', 'MEDIUM', combined_score
        elif combined_score < -0.30:
            return 'WEAK_SELL', 'LOW', combined_score
        else:
            return 'HOLD', 'NEUTRAL', combined_score
    
    def execute_signal(self, analysis: Dict, data: pd.DataFrame) -> bool:
        """Execute trading signal with tiered position sizing."""
        signal = analysis['signal']
        
        # Skip HOLD and NEUTRAL
        if signal in ['HOLD', 'NEUTRAL']:
            return False
        
        # Check circuit breaker
        can_trade, reason = self.circuit_breaker.check_can_trade()
        if not can_trade:
            logger.warning(f"Circuit breaker: {reason}")
            return False
        
        # Use provided data (no extra fetch)
        if data is None or len(data) < 50:
            logger.warning("Insufficient data for signal execution")
            return False
        
        current_price = analysis['price']
        atr = self.ei.atr(data).iloc[-1]
        
        # Spot mode supports BUY entries and SELL exits (no synthetic shorting).
        direction = 'LONG' if 'BUY' in signal else 'LONG'
        order_side = 'BUY' if 'BUY' in signal else 'SELL'
        
        # Tiered position sizing based on signal strength
        size_multipliers = {
            'STRONG_BUY': 1.0,   # 100% of calculated size
            'BUY': 0.75,          # 75% of calculated size
            'WEAK_BUY': 0.50,     # 50% of calculated size
            'STRONG_SELL': 1.0,
            'SELL': 0.75,
            'WEAK_SELL': 0.50
        }
        size_mult = size_multipliers.get(signal, 0.5)
        
        stop_loss, take_profit = self.dynamic_stops.calculate_stops(
            entry_price=current_price,
            atr=atr,
            direction=direction,
            regime=self.current_regime
        )

        if order_side == 'BUY':
            position = self.position_sizer.calculate_position_size(
                equity=self.equity,
                entry_price=current_price,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                atr=atr,
                volatility_regime=analysis.get('volatility_regime', 'NORMAL')
            )
            adjusted_size = position.size * size_mult
        else:
            # For spot SELL, use current base-asset balance as available exit size.
            base_free = 0.0
            if self.cfg.test_mode:
                base_free = self.virtual_base_balance
            elif self.exchange is not None:
                base_free = self.exchange.get_base_asset_free()
            adjusted_size = base_free * size_mult if base_free > 0 else 0.0
        
        # Execute trade
        if self.cfg.test_mode:
            if current_price <= 0:
                logger.error("[TEST] Preco invalido para simulacao")
                return False

            executed_size = adjusted_size
            if order_side == 'BUY':
                max_affordable = self.virtual_quote_balance / current_price
                executed_size = min(adjusted_size, max_affordable)
                if executed_size <= 0:
                    logger.info("[TEST] BUY ignorado: saldo USDT insuficiente na carteira demo")
                    return False
                notional = executed_size * current_price
                self.virtual_quote_balance = max(0.0, self.virtual_quote_balance - notional)
                self.virtual_base_balance += executed_size
            else:
                executed_size = min(adjusted_size, self.virtual_base_balance)
                if executed_size <= 0:
                    logger.info("[TEST] SELL ignorado: sem saldo do ativo base na carteira demo")
                    return False
                notional = executed_size * current_price
                self.virtual_base_balance = max(0.0, self.virtual_base_balance - executed_size)
                self.virtual_quote_balance += notional

            self._update_virtual_equity(current_price)

            logger.info(
                f"[TEST] {signal} {executed_size:.6f} @ ${current_price:,.2f} (size: {size_mult:.0%})"
            )
            logger.info(f"       SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
            self._log_virtual_wallet()

            # Record trade
            self.trades.append({
                'time': datetime.now(),
                'signal': signal,
                'price': current_price,
                'size': executed_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': size_mult,
                'quote_balance': self.virtual_quote_balance,
                'base_balance': self.virtual_base_balance,
                'equity': self.equity
            })

            return True
        else:
            # Real execution via Binance Spot
            logger.info(f"[LIVE] Executing {signal} via Binance Spot...")

            if self.exchange is None:
                logger.error("[X] Binance client unavailable")
                return False

            if order_side == 'SELL' and adjusted_size <= 0:
                logger.info("[LIVE] SELL signal ignored: no base asset balance available")
                return False

            order_quantity = self.exchange.calculate_order_quantity(adjusted_size)
            if order_quantity <= 0:
                logger.error("[X] Invalid order quantity calculated")
                return False

            result = self.exchange.execute_market_order(
                side=order_side,
                quantity=order_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=f"XAKCN {signal}"
            )
            
            if result:
                logger.info(f"[OK] Order executed: Ticket {result['order_id']}")
                self.trades.append({
                    'time': datetime.now(),
                    'signal': signal,
                    'price': current_price,
                    'size': adjusted_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strength': size_mult,
                    'ticket': result['order_id']
                })
                return True
            else:
                logger.error("[X] Order execution failed")
                return False
    
    def print_dashboard(self, analysis: Dict, data: Optional[pd.DataFrame] = None):
        """Print terminal trader dashboard."""
        ensemble_block = analysis.get('ensemble', {})
        components = ensemble_block.get('components', {}) if isinstance(ensemble_block, dict) else {}
        ml_block = analysis.get('ml', {})
        ml_prob = float(ml_block.get('probability', 0.5)) if isinstance(ml_block, dict) else 0.5
        market_stats = self._compute_market_stats(data)

        indicators: Dict[str, float] = {}
        if data is not None and len(data) >= 14:
            close_14 = data['close'].tail(14)
            std_14 = close_14.std()
            if std_14 and std_14 > 0:
                rsi_val = 50 + (close_14.iloc[-1] - close_14.mean()) / std_14 * 10
                indicators['rsi'] = max(0.0, min(100.0, float(rsi_val)))
            else:
                indicators['rsi'] = 50.0
            if 'volume' in data:
                indicators['obv'] = float(data['volume'].sum())
            indicators['macd'] = float(components.get('macd', 0.0))
            indicators['adx'] = float(components.get('adx', 0.0)) * 50.0

        dashboard_data = {
            'symbol': self.cfg.symbol,
            'timestamp': analysis['timestamp'],
            'mode': 'TEST' if self.cfg.test_mode else 'LIVE',
            'cycle': self.cycle_count,
            'next_update': f"{self.cfg.cycle_interval}s",
            'price': float(analysis.get('price', 0.0) or 0.0),
            'change_24h': float(market_stats.get('change_pct', 0.0) or 0.0),
            'high_24h': float(market_stats.get('high', 0.0) or 0.0),
            'low_24h': float(market_stats.get('low', 0.0) or 0.0),
            'variation_label': str(market_stats.get('variation_label', self.cfg.timeframe.upper())),
            'signal': analysis.get('signal', 'HOLD'),
            'confidence': analysis.get('confidence', 'LOW'),
            'score': float(analysis.get('score', 0.0) or 0.0),
            'ensemble_score': float(ensemble_block.get('score', 0.0) or 0.0) if isinstance(ensemble_block, dict) else 0.0,
            'ml_prob': ml_prob,
            'regime': analysis.get('regime', 'UNKNOWN'),
            'vol_regime': analysis.get('volatility_regime', 'NORMAL'),
            'vol_ratio': float(analysis.get('volatility_ratio', 1.0) or 1.0),
            'equity': float(self.equity),
            'initial': float(self.cfg.initial_capital),
            'trades': len(self.trades),
            'trade_history': self.trades[-5:],
            'components': components,
            'indicators': indicators,
        }
        visual.print_terminal_trader_dashboard(dashboard_data)
    
    def run_cycle(self):
        """Run one trading cycle."""
        self.cycle_count += 1
        
        logger.info(f"\n--- Cycle {self.cycle_count} ---")
        
        # Fetch data
        data = self.fetch_data()
        if data is None:
            logger.error("Failed to fetch data")
            return
        
        # Analyze
        analysis = self.analyze_market(data)
        
        # Print dashboard
        self.print_dashboard(analysis, data)
        
        # Execute
        if analysis['signal'] != 'HOLD':
            self.execute_signal(analysis, data)
        
        # Log status
        logger.info(f"Equity: ${self.equity:,.2f} | Trades: {len(self.trades)}")
    
    def run_continuous(self):
        """Run continuous trading loop."""
        self.running = True
        
        logger.info("\nIniciando trading continuo...")
        logger.info(f"Intervalo: {self.cfg.cycle_interval}s")
        if self.cfg.max_cycles > 0:
            logger.info(f"Ciclos maximos: {self.cfg.max_cycles}")
        logger.info("Pressione Ctrl+C para parar\n")
        
        try:
            while self.running:
                if self.cfg.max_cycles > 0 and self.cycle_count >= self.cfg.max_cycles:
                    logger.info("Max cycles reached. Stopping trading loop.")
                    break
                self.run_cycle()

                if self.cfg.max_cycles > 0 and self.cycle_count >= self.cfg.max_cycles:
                    logger.info("Max cycles reached. Stopping trading loop.")
                    break
                
                # Sleep with interrupt handling
                for _ in range(self.cfg.cycle_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            logger.info("\n\nStopping bot...")
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        
        logger.info("\n" + "=" * 70)
        logger.info("RESUMO FINAL")
        logger.info("=" * 70)
        logger.info(f"Ciclos: {self.cycle_count}")
        logger.info(f"Trades: {len(self.trades)}")
        logger.info(f"Equity final: ${self.equity:,.2f}")
        logger.info("=" * 70)
        logger.info("Bot parado.")


def run_demo_mode(cfg: TradingConfig):
    """Run demo mode with visual dashboard."""
    logger.info("Starting DEMO mode...")
    
    bot = UnifiedTradingBot(cfg)
    bot.initialize()
    
    # Demo loop with visual dashboard
    bot.running = True
    cycle = 0
    
    try:
        while bot.running:
            cycle += 1
            bot.cycle_count = cycle
            data = bot.fetch_data()
            
            if data is not None:
                analysis = bot.analyze_market(data)
                if analysis['signal'] != 'HOLD':
                    bot.execute_signal(analysis, data)
                market_stats = bot._compute_market_stats(data)
                
                # Build dashboard data
                # Calculate simple RSI approximation
                if len(data) >= 14:
                    close_14 = data['close'].tail(14)
                    rsi_val = 50 + (close_14.iloc[-1] - close_14.mean()) / close_14.std() * 10
                    rsi_val = max(0, min(100, rsi_val))  # Clamp to 0-100
                else:
                    rsi_val = 50
                
                dashboard_data = {
                    'symbol': cfg.symbol,
                    'timestamp': analysis['timestamp'],
                    'mode': 'DEMO',
                    'cycle': cycle,
                    'price': analysis['price'],
                    'change_24h': float(market_stats.get('change_pct', 0.0) or 0.0),
                    'high_24h': float(market_stats.get('high', 0.0) or 0.0),
                    'low_24h': float(market_stats.get('low', 0.0) or 0.0),
                    'variation_label': str(market_stats.get('variation_label', cfg.timeframe.upper())),
                    'signal': analysis['signal'],
                    'confidence': analysis['confidence'],
                    'score': analysis['score'],
                    'ensemble_score': analysis.get('ensemble', {}).get('score', 0.0),
                    'ml_prob': analysis.get('ml', {}).get('probability', 0.5),
                    'initial': cfg.initial_capital,
                    'equity': bot.equity,
                    'trades': len(bot.trades),
                    'trade_history': bot.trades[-5:],
                    'next_update': f"{cfg.cycle_interval}s",
                    'regime': analysis['regime'],
                    'vol_regime': analysis.get('volatility_regime', 'NORMAL'),
                    'vol_ratio': analysis.get('volatility_ratio', 1.0),
                    'components': analysis.get('ensemble', {}).get('components', {}),
                    'indicators': {
                        'rsi': rsi_val,
                        'macd': analysis.get('ensemble', {}).get('components', {}).get('macd', 0),
                        'adx': 25,
                        'obv': data['volume'].sum() if 'volume' in data else 0,
                    }
                }
                
                # Print visual dashboard
                visual.print_demo_dashboard(dashboard_data)
            else:
                logger.warning("Ciclo %s sem dados. Mantendo autopilot e tentando novamente.", cycle)

            if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
                logger.info("Max demo cycles reached. Stopping demo loop.")
                break

            # Wait for next cycle
            time.sleep(cfg.cycle_interval)
            
    except KeyboardInterrupt:
        logger.info("\nDEMO mode stopped.")
    finally:
        bot.shutdown()


def run_train_mode(cfg: TradingConfig):
    """Train ML model with visual dashboard using Binance spot data."""
    visual.print_header("ML TRAINING MODE", f"{cfg.symbol} | {cfg.cycle_interval} days of data")
    logger.info("Training ML model...")

    if not HAS_BINANCE:
        logger.error("python-binance not available. Install with: pip install python-binance")
        return

    logger.info("Fetching historical data from Binance Spot...")
    exchange_client = BinanceSpotTrader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        api_key=cfg.binance_api_key,
        api_secret=cfg.binance_secret_key,
        testnet=cfg.binance_testnet,
        recv_window=cfg.binance_recv_window
    )

    if not exchange_client.initialized:
        logger.error("Failed to connect to Binance Spot")
        return

    candles_per_day = 24 if cfg.timeframe == '1h' else 1
    total_candles = min(cfg.cycle_interval * candles_per_day * 2, 5000)
    data = exchange_client.get_market_data(limit=total_candles)

    if data is None or len(data) < 100:
        logger.error("Insufficient data from Binance Spot")
        return

    logger.info(f"[OK] Loaded {len(data)} candles from Binance Spot")

    engineer = FeatureEngineer()
    X_train, y_train, X_test, y_test = engineer.prepare_data(data, train_ratio=0.8)

    trainer = XGBoostTrainer()
    cv_results = trainer.cross_validate(X_train, y_train, n_splits=5)
    logger.info(f"CV Accuracy: {cv_results['accuracy_mean']:.3f}")

    trainer.train(X_train, y_train, X_test, y_test)
    metrics = trainer.evaluate(X_test, y_test)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.3f}")

    importance = trainer.get_feature_importance()
    logger.info("\nTop 10 Features:")
    logger.info(importance.head(10).to_string())

    trainer.save_model('xgboost_latest.pkl')
    logger.info("ML Model training complete!")


def run_backtest_mode(cfg: TradingConfig):
    """Run backtest with visual dashboard using Binance spot data."""
    from backtest.engine import BacktestEngine

    visual.print_backtest_header(cfg.symbol, cfg.cycle_interval)
    logger.info("Running backtest...")

    if not HAS_BINANCE:
        logger.error("python-binance not available. Install with: pip install python-binance")
        return

    from backtest.ensemble_scoring import create_signals_for_backtest

    logger.info("Fetching historical data from Binance Spot...")
    exchange_client = BinanceSpotTrader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        api_key=cfg.binance_api_key,
        api_secret=cfg.binance_secret_key,
        testnet=cfg.binance_testnet,
        recv_window=cfg.binance_recv_window
    )

    if not exchange_client.initialized:
        logger.error("Failed to connect to Binance Spot")
        return

    candles_per_day = 24 if cfg.timeframe == '1h' else 1
    total_candles = min(cfg.cycle_interval * candles_per_day, 5000)
    data = exchange_client.get_market_data(limit=total_candles)

    if data is None or len(data) < 100:
        logger.error("Insufficient data from Binance Spot")
        return

    logger.info(f"[OK] Loaded {len(data)} candles from Binance Spot")

    scorer = EnsembleScorer()
    if cfg.trend_filter:
        logger.info(
            "Trend filter enabled: EMA%s/EMA%s + ADX>%.1f",
            cfg.trend_ema_fast,
            cfg.trend_ema_slow,
            cfg.trend_adx_threshold
        )
    if cfg.volume_spike_filter:
        logger.info(
            "Volume spike filter enabled: volume > SMA(%s) * %.2f",
            cfg.volume_spike_window,
            cfg.volume_spike_multiplier
        )
    entries, exits = create_signals_for_backtest(
        data,
        scorer,
        use_trend_filter=cfg.trend_filter,
        trend_ema_fast=cfg.trend_ema_fast,
        trend_ema_slow=cfg.trend_ema_slow,
        trend_adx_threshold=cfg.trend_adx_threshold,
        ignore_exit_signals=cfg.ignore_exit_signals,
        use_volume_spike_filter=cfg.volume_spike_filter,
        volume_spike_window=cfg.volume_spike_window,
        volume_spike_multiplier=cfg.volume_spike_multiplier
    )

    engine = BacktestEngine(initial_cash=cfg.initial_capital, fees=0.001)
    strategy_name = 'Unified_Ensemble'
    if cfg.trend_filter:
        strategy_name += '_TrendFilter'

    if cfg.fixed_atr_rr:
        logger.info(
            "Fixed ATR R:R enabled | SL=%.2fx ATR | TP=%.2fx ATR | Ignore exits=%s",
            cfg.atr_sl_mult,
            cfg.atr_tp_mult,
            cfg.ignore_exit_signals
        )
        stops = DynamicStops(
            base_sl_mult=cfg.atr_sl_mult,
            base_tp_mult=cfg.atr_tp_mult,
            use_regime_multipliers=False
        )
        atr_series = EnhancedIndicators().atr(data, period=cfg.atr_period)
        sl_stop, tp_stop = stops.calculate_atr_stop_fractions(
            close=data['close'],
            atr=atr_series,
            sl_mult=cfg.atr_sl_mult,
            tp_mult=cfg.atr_tp_mult
        )

        strategy_name += '_FixedRR'
        result = engine.run_with_stops(
            data=data,
            entries=entries,
            exits=exits,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            strategy_name=strategy_name,
            symbol=cfg.symbol
        )
    else:
        result = engine.run_backtest(
            data, entries, exits,
            strategy_name=strategy_name,
            symbol=cfg.symbol
        )
    result.print_summary()


def run_optimize_mode(cfg: TradingConfig):
    """Optimize strategy parameters with visual progress using Binance spot data."""
    from backtest.optimization import StrategyOptimizer

    visual.print_optimization_header()
    logger.info("Optimizing parameters...")

    if not HAS_BINANCE:
        logger.error("python-binance not available. Install with: pip install python-binance")
        return

    from backtest.ensemble_scoring import create_signals_for_backtest

    logger.info("Fetching historical data from Binance Spot...")
    exchange_client = BinanceSpotTrader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        api_key=cfg.binance_api_key,
        api_secret=cfg.binance_secret_key,
        testnet=cfg.binance_testnet,
        recv_window=cfg.binance_recv_window
    )

    if not exchange_client.initialized:
        logger.error("Failed to connect to Binance Spot")
        return

    candles_per_day = 24 if cfg.timeframe == '1h' else 1
    total_candles = min(cfg.cycle_interval * candles_per_day, 3000)
    data = exchange_client.get_market_data(limit=total_candles)

    if data is None or len(data) < 100:
        logger.error("Insufficient data from Binance Spot")
        return

    logger.info(f"[OK] Loaded {len(data)} candles from Binance Spot")
    
    # Define strategy function
    def ensemble_strategy(data, **params):
        raw_weights = {
            'rsi': params.get('rsi_weight', 0.10),
            'macd': params.get('macd_weight', 0.10),
            'ema': params.get('ema_weight', 0.10),
            'adx': params.get('adx_weight', 0.15),
            'obv': params.get('obv_weight', 0.10),
            'ichimoku': params.get('ichimoku_weight', 0.10),
            'supertrend': params.get('supertrend_weight', 0.10),
            'bb': params.get('bb_weight', 0.10),
            'stoch': params.get('stoch_weight', 0.08),
            'williams': params.get('williams_weight', 0.07),
        }

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            return pd.Series(False, index=data.index), pd.Series(False, index=data.index)

        normalized = {key: value / total_weight for key, value in raw_weights.items()}

        weights = EnsembleWeights(
            rsi=normalized['rsi'],
            macd=normalized['macd'],
            ema=normalized['ema'],
            adx=normalized['adx'],
            obv=normalized['obv'],
            ichimoku=normalized['ichimoku'],
            supertrend=normalized['supertrend'],
            bb=normalized['bb'],
            stoch=normalized['stoch'],
            williams=normalized['williams'],
        )
        
        scorer = EnsembleScorer(weights)
        entries, exits = create_signals_for_backtest(
            data,
            scorer,
            use_trend_filter=cfg.trend_filter,
            trend_ema_fast=cfg.trend_ema_fast,
            trend_ema_slow=cfg.trend_ema_slow,
            trend_adx_threshold=cfg.trend_adx_threshold,
            use_volume_spike_filter=cfg.volume_spike_filter,
            volume_spike_window=cfg.volume_spike_window,
            volume_spike_multiplier=cfg.volume_spike_multiplier
        )
        return entries, exits
    
    # Optimize
    optimizer = StrategyOptimizer(
        data=data,
        strategy_func=ensemble_strategy,
        n_trials=cfg.optimize_trials
    )
    
    param_space = {
        'rsi_weight': (0.05, 0.20),
        'macd_weight': (0.05, 0.20),
        'ema_weight': (0.05, 0.20),
        'adx_weight': (0.10, 0.25),
        'obv_weight': (0.05, 0.15),
        'ichimoku_weight': (0.05, 0.15),
        'supertrend_weight': (0.05, 0.15),
        'bb_weight': (0.05, 0.15),
        'stoch_weight': (0.05, 0.12),
        'williams_weight': (0.05, 0.12),
    }
    
    best_params = optimizer.optimize(param_space, objective='sharpe')
    
    logger.info(f"\nOptimization complete!")
    logger.info(f"Best parameters: {best_params}")
    


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='XAKCN Unified Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Demo mode with visual dashboard
    python main.py --mode demo
    
    # Live trading (test mode)
    python main.py --mode trade --symbol BTCUSDT
    
    # Train ML model
    python main.py --mode train --days 90
    
    # Run backtest
    python main.py --mode backtest --days 180
    python main.py --mode backtest --days 90 --trend_filter
    python main.py --mode backtest --days 90 --trend_filter --atr-sl-mult 1 --atr-tp-mult 2
    python main.py --mode backtest --days 90 --trend_filter --volume_spike_filter
    
    # Optimize parameters
    python main.py --mode optimize --days 180 --trials 100
        """
    )
    
    parser.add_argument('--mode', '-m',
                       choices=['demo', 'trade', 'train', 'backtest', 'optimize'],
                       default='demo',
                       help='Operation mode (default: demo)')
    
    parser.add_argument('--symbol', '-s',
                       default=config.OPERATION_CODE,
                       help=f'Trading symbol (default: {config.OPERATION_CODE})')
    
    parser.add_argument('--timeframe', '-t',
                       default=config.CANDLE_PERIOD,
                       help=f'Timeframe (default: {config.CANDLE_PERIOD})')
    
    parser.add_argument('--days', '-d',
                       type=int,
                       default=90,
                       help='Days of data for train/backtest/optimize (default: 90)')

    parser.add_argument('--trials',
                       type=int,
                       default=50,
                       help='Optimization trials for --mode optimize (default: 50)')
    
    parser.add_argument('--interval', '-i',
                       type=int,
                       default=config.CYCLE_INTERVAL,
                       help=f'Cycle interval in seconds (default: {config.CYCLE_INTERVAL})')

    parser.add_argument('--cycles',
                       type=int,
                       default=0,
                       help='Max cycles for demo/trade (0 = infinite loop)')

    parser.add_argument('--trend_filter',
                       action='store_true',
                       help='Enable trend filter EMA20/EMA50 with ADX threshold')

    parser.add_argument('--trend-ema-fast',
                       type=int,
                       default=20,
                       help='Fast EMA period for trend filter (default: 20)')

    parser.add_argument('--trend-ema-slow',
                       type=int,
                       default=50,
                       help='Slow EMA period for trend filter (default: 50)')

    parser.add_argument('--trend-adx',
                       type=float,
                       default=22.0,
                       help='Minimum ADX for trend filter (default: 22)')

    parser.add_argument('--volume_spike_filter',
                       action='store_true',
                       help='Enable volume spike confirmation filter')

    parser.add_argument('--volume-spike-window',
                       type=int,
                       default=20,
                       help='Rolling window for volume spike filter (default: 20)')

    parser.add_argument('--volume-spike-multiplier',
                       type=float,
                       default=1.2,
                       help='Multiplier for volume spike threshold (default: 1.2)')

    parser.add_argument('--no-fixed-rr',
                       action='store_true',
                       help='Disable fixed ATR risk/reward backtest mode')

    parser.add_argument('--use-exit-signals',
                       action='store_true',
                       help='Use strategy exit signals (disable "ignore exits")')

    parser.add_argument('--atr-sl-mult',
                       type=float,
                       default=1.0,
                       help='ATR multiplier for stop loss (default: 1.0)')

    parser.add_argument('--atr-tp-mult',
                       type=float,
                       default=2.0,
                       help='ATR multiplier for take profit (default: 2.0)')

    parser.add_argument('--atr-period',
                       type=int,
                       default=14,
                       help='ATR period for fixed R:R stops (default: 14)')
    
    parser.add_argument('--capital',
                       type=float,
                       default=10000.0,
                       help='Initial capital (default: 10000)')
    
    parser.add_argument('--risk',
                       type=float,
                       default=risk_config.MAX_RISK_PER_TRADE,
                       help=f'Max risk per trade as fraction (default: {risk_config.MAX_RISK_PER_TRADE})')
    
    parser.add_argument('--live',
                       action='store_true',
                       help='Enable live trading (DANGEROUS - real money)')
    
    parser.add_argument('--no-ml',
                       action='store_true',
                       help='Disable ML predictions')
    
    parser.add_argument('--no-ensemble',
                       action='store_true',
                       help='Disable ensemble scoring')

    parser.add_argument('--binance-api-key',
                       default=config.BINANCE_API_KEY,
                       help='Binance API key (optional for public data modes)')

    parser.add_argument('--binance-secret-key',
                       default=config.BINANCE_SECRET_KEY,
                       help='Binance secret key (required for live trading)')

    parser.add_argument('--binance-testnet',
                       action='store_true',
                       default=config.BINANCE_TESTNET,
                       help='Use Binance Spot testnet')

    parser.add_argument('--binance-live-endpoint',
                       action='store_true',
                       help='Use Binance Spot live endpoint (disables testnet)')

    parser.add_argument('--recv-window',
                       type=int,
                       default=config.BINANCE_RECV_WINDOW,
                       help=f'Binance recvWindow in ms (default: {config.BINANCE_RECV_WINDOW})')
    
    args = parser.parse_args()

    if args.binance_live_endpoint:
        args.binance_testnet = False
    
    # Create config
    cli_risk = args.risk / 100 if args.risk > 1 else args.risk
    if cli_risk > 0.02:
        logger.warning("Risk per trade above 2% is not allowed. Capping to 0.02.")
        cli_risk = 0.02

    cfg = TradingConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        max_risk_per_trade=cli_risk,
        test_mode=not args.live,
        use_ml=not args.no_ml,
        use_ensemble=not args.no_ensemble,
        cycle_interval=args.interval if args.mode in ['trade', 'demo'] else args.days,
        max_cycles=max(0, args.cycles),
        optimize_trials=max(1, args.trials),
        trend_filter=args.trend_filter,
        trend_ema_fast=max(2, args.trend_ema_fast),
        trend_ema_slow=max(3, args.trend_ema_slow),
        trend_adx_threshold=max(0.0, args.trend_adx),
        volume_spike_filter=args.volume_spike_filter,
        volume_spike_window=max(2, args.volume_spike_window),
        volume_spike_multiplier=max(1.0, args.volume_spike_multiplier),
        fixed_atr_rr=not args.no_fixed_rr,
        ignore_exit_signals=not args.use_exit_signals,
        atr_sl_mult=max(0.1, args.atr_sl_mult),
        atr_tp_mult=max(0.1, args.atr_tp_mult),
        atr_period=max(2, args.atr_period),
        binance_api_key=args.binance_api_key,
        binance_secret_key=args.binance_secret_key,
        binance_testnet=args.binance_testnet,
        binance_recv_window=args.recv_window
    )

    if cfg.trend_ema_fast >= cfg.trend_ema_slow:
        logger.warning(
            "trend_ema_fast (%s) must be lower than trend_ema_slow (%s). Using defaults 20/50.",
            cfg.trend_ema_fast,
            cfg.trend_ema_slow
        )
        cfg.trend_ema_fast = 20
        cfg.trend_ema_slow = 50

    if cfg.atr_tp_mult <= cfg.atr_sl_mult:
        logger.warning(
            "atr_tp_mult (%s) should be greater than atr_sl_mult (%s). Using defaults 1:2.",
            cfg.atr_tp_mult,
            cfg.atr_sl_mult
        )
        cfg.atr_sl_mult = 1.0
        cfg.atr_tp_mult = 2.0

    if args.live and (not cfg.binance_api_key or not cfg.binance_secret_key):
        logger.error("Live trading requires BINANCE_API_KEY and BINANCE_SECRET_KEY.")
        return
    
    # Route to appropriate mode
    if args.mode == 'demo':
        run_demo_mode(cfg)
    
    elif args.mode == 'trade':
        bot = UnifiedTradingBot(cfg)
        bot.initialize()
        bot.run_continuous()
    
    elif args.mode == 'train':
        run_train_mode(cfg)
    
    elif args.mode == 'backtest':
        run_backtest_mode(cfg)
    
    elif args.mode == 'optimize':
        run_optimize_mode(cfg)


if __name__ == '__main__':
    main()
