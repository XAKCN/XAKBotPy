"""
Weighted Ensemble Scoring System
Combines multiple indicators into a unified score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from utils.enhanced_indicators import EnhancedIndicators, HAS_TALIB

# Try to import TA-Lib
try:
    import talib
except ImportError:
    talib = None


@dataclass
class EnsembleWeights:
    """Weights for ensemble scoring."""
    rsi: float = 0.10
    macd: float = 0.10
    ema: float = 0.10
    adx: float = 0.15
    obv: float = 0.10
    ichimoku: float = 0.10
    supertrend: float = 0.10
    bb: float = 0.10
    stoch: float = 0.08
    williams: float = 0.07
    
    def validate(self):
        """Ensure weights sum to 1.0."""
        total = sum([
            self.rsi, self.macd, self.ema, self.adx, self.obv,
            self.ichimoku, self.supertrend, self.bb, self.stoch, self.williams
        ])
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"


class EnsembleScorer:
    """
    Weighted ensemble scoring for trading signals.
    Combines multiple indicators into normalized scores.
    """
    
    def __init__(self, weights: Optional[EnsembleWeights] = None):
        """
        Initialize ensemble scorer.
        
        Args:
            weights: EnsembleWeights instance, uses defaults if None
        """
        self.weights = weights or EnsembleWeights()
        self.weights.validate()
        self.ei = EnhancedIndicators()
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate all indicators for scoring.
        
        Returns:
            Dictionary of indicator Series
        """
        indicators = {}
        
        if HAS_TALIB and talib is not None:
            # Use TA-Lib for better performance
            # RSI
            indicators['rsi'] = talib.RSI(data['close'], timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                data['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # EMAs
            indicators['ema_10'] = talib.EMA(data['close'], timeperiod=10)
            indicators['ema_20'] = talib.EMA(data['close'], timeperiod=20)
            indicators['ema_50'] = talib.EMA(data['close'], timeperiod=50)
            
            # Stochastic
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(
                data['high'], data['low'], data['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(
                data['high'], data['low'], data['close'], timeperiod=14
            )
        else:
            # Pandas implementations
            close = data['close']
            high = data['high']
            low = data['low']
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            indicators['macd'] = ema_12 - ema_26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # EMAs
            indicators['ema_10'] = close.ewm(span=10, adjust=False).mean()
            indicators['ema_20'] = close.ewm(span=20, adjust=False).mean()
            indicators['ema_50'] = close.ewm(span=50, adjust=False).mean()
            
            # Stochastic
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            indicators['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
            indicators['stoch_d'] = indicators['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            indicators['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        # ADX (from EnhancedIndicators - handles both cases)
        indicators['adx'], indicators['plus_di'], indicators['minus_di'] = \
            self.ei.adx_directional(data)
        
        # OBV
        indicators['obv'], indicators['obv_ma'], indicators['obv_delta'] = \
            self.ei.obv_normalized(data)
        
        # Ichimoku
        ichi = self.ei.ichimoku(data)
        indicators['tenkan_sen'] = ichi['tenkan_sen']
        indicators['kijun_sen'] = ichi['kijun_sen']
        indicators['cloud_top'] = ichi['cloud_top']
        indicators['cloud_bottom'] = ichi['cloud_bottom']
        
        # Supertrend
        indicators['supertrend'], indicators['supertrend_dir'] = \
            self.ei.supertrend(data)
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = \
            self.ei.bollinger_bands(data)
        
        # ATR for reference
        indicators['atr'] = self.ei.atr(data)
        
        return indicators
    
    def normalize_scores(self, indicators: Dict, data: pd.DataFrame) -> Dict[str, float]:
        """
        Normalize each indicator to [-1, 1] range.
        
        Returns:
            Dictionary of normalized scores (last value)
        """
        scores = {}
        close = data['close'].iloc[-1]
        
        # RSI: 0-100 -> -1 to 1 (oversold=1, overbought=-1)
        rsi = indicators['rsi'].iloc[-1]
        if not np.isnan(rsi):
            scores['rsi'] = (50 - rsi) / 50  # Inverted: low RSI = bullish
        else:
            scores['rsi'] = 0
        
        # MACD: Compare to signal
        macd = indicators['macd'].iloc[-1]
        macd_signal = indicators['macd_signal'].iloc[-1]
        macd_hist = indicators['macd_hist'].iloc[-1]
        if not np.isnan(macd) and not np.isnan(macd_signal):
            # Normalize based on histogram
            macd_range = abs(indicators['macd']).rolling(50).mean().iloc[-1] + 1e-10
            scores['macd'] = np.clip(macd_hist / macd_range, -1, 1)
        else:
            scores['macd'] = 0
        
        # EMA: Price relative to EMAs
        ema_10 = indicators['ema_10'].iloc[-1]
        ema_20 = indicators['ema_20'].iloc[-1]
        ema_50 = indicators['ema_50'].iloc[-1]
        
        if not np.isnan(ema_20):
            # Score based on price position relative to EMAs
            ema_score = 0
            if close > ema_10:
                ema_score += 0.3
            if close > ema_20:
                ema_score += 0.3
            if ema_10 > ema_20:
                ema_score += 0.2
            if ema_20 > ema_50:
                ema_score += 0.2
            scores['ema'] = ema_score
        else:
            scores['ema'] = 0
        
        # ADX: Trend strength filter
        adx = indicators['adx'].iloc[-1]
        plus_di = indicators['plus_di'].iloc[-1]
        minus_di = indicators['minus_di'].iloc[-1]
        
        if not np.isnan(adx) and adx > 25:  # Strong trend
            if plus_di > minus_di:
                scores['adx'] = min(adx / 50, 1.0)  # Bullish trend
            else:
                scores['adx'] = -min(adx / 50, 1.0)  # Bearish trend
        else:
            scores['adx'] = 0  # No clear trend
        
        # OBV: Volume confirmation
        obv_delta = indicators['obv_delta'].iloc[-1]
        if not np.isnan(obv_delta):
            # Normalize OBV delta
            scores['obv'] = np.clip((obv_delta - 1) * 2, -1, 1)
        else:
            scores['obv'] = 0
        
        # Ichimoku
        tenkan = indicators['tenkan_sen'].iloc[-1]
        kijun = indicators['kijun_sen'].iloc[-1]
        cloud_top = indicators['cloud_top'].iloc[-1]
        cloud_bottom = indicators['cloud_bottom'].iloc[-1]
        
        if not np.isnan(cloud_top):
            ichi_score = 0
            if close > cloud_top:
                ichi_score += 0.4
            elif close > cloud_bottom:
                ichi_score += 0.1
            if tenkan > kijun:
                ichi_score += 0.3
            if close > tenkan:
                ichi_score += 0.3
            scores['ichimoku'] = ichi_score
        else:
            scores['ichimoku'] = 0
        
        # Supertrend
        supertrend_dir = indicators['supertrend_dir'].iloc[-1]
        if not np.isnan(supertrend_dir):
            scores['supertrend'] = supertrend_dir * 1.0  # Already -1 or 1
        else:
            scores['supertrend'] = 0
        
        # Bollinger Bands
        bb_upper = indicators['bb_upper'].iloc[-1]
        bb_lower = indicators['bb_lower'].iloc[-1]
        bb_middle = indicators['bb_middle'].iloc[-1]
        
        if not np.isnan(bb_upper) and not np.isnan(bb_lower):
            # Position within bands
            band_range = bb_upper - bb_lower
            if band_range > 0:
                position = (close - bb_lower) / band_range
                # Mean reversion: high position = bearish, low = bullish
                scores['bb'] = (0.5 - position) * 2
            else:
                scores['bb'] = 0
        else:
            scores['bb'] = 0
        
        # Stochastic
        stoch_k = indicators['stoch_k'].iloc[-1]
        if not np.isnan(stoch_k):
            # Invert: low stochastic = bullish (oversold)
            scores['stoch'] = (50 - stoch_k) / 50
        else:
            scores['stoch'] = 0
        
        # Williams %R
        williams = indicators['williams_r'].iloc[-1]
        if not np.isnan(williams):
            # Normalize from [-100, 0] to [-1, 1]
            scores['williams'] = (williams + 50) / 50
        else:
            scores['williams'] = 0
        
        return scores
    
    def calculate_ensemble_score(self, data: pd.DataFrame) -> Tuple[str, str, float, Dict]:
        """
        Calculate final ensemble score.
        
        Returns:
            (decision, confidence, final_score, component_scores)
            decision: 'BUY', 'SELL', or 'HOLD'
            confidence: 'HIGH', 'MEDIUM', or 'LOW'
            final_score: Normalized score (-1 to 1)
            component_scores: Dict of individual indicator scores
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Normalize scores
        scores = self.normalize_scores(indicators, data)
        
        # Apply weights
        weighted_score = (
            scores['rsi'] * self.weights.rsi +
            scores['macd'] * self.weights.macd +
            scores['ema'] * self.weights.ema +
            scores['adx'] * self.weights.adx +
            scores['obv'] * self.weights.obv +
            scores['ichimoku'] * self.weights.ichimoku +
            scores['supertrend'] * self.weights.supertrend +
            scores['bb'] * self.weights.bb +
            scores['stoch'] * self.weights.stoch +
            scores['williams'] * self.weights.williams
        )
        
        # Determine decision
        if weighted_score > 0.3:
            decision = 'BUY'
            if weighted_score > 0.6:
                confidence = 'HIGH'
            else:
                confidence = 'MEDIUM'
        elif weighted_score < -0.3:
            decision = 'SELL'
            if weighted_score < -0.6:
                confidence = 'HIGH'
            else:
                confidence = 'MEDIUM'
        else:
            decision = 'HOLD'
            confidence = 'LOW'
        
        return decision, confidence, weighted_score, scores
    
    def check_confluence(self, data: pd.DataFrame, 
                         min_confirmations: int = 3) -> Tuple[str, int]:
        """
        Check for confluent signals from multiple independent sources.
        
        Args:
            data: OHLCV DataFrame
            min_confirmations: Minimum number of confirmations required
            
        Returns:
            (signal_strength, confirmation_count)
            signal_strength: 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
        """
        indicators = self.calculate_indicators(data)
        scores = self.normalize_scores(indicators, data)
        
        # Count bullish confirmations
        bullish_count = sum([
            scores['rsi'] > 0.3,
            scores['macd'] > 0.3,
            scores['ema'] > 0.3,
            scores['adx'] > 0.2,
            scores['obv'] > 0.2,
            scores['ichimoku'] > 0.3,
            scores['supertrend'] > 0,
            scores['bb'] > 0.3,
            scores['stoch'] > 0.3,
            scores['williams'] > 0.3
        ])
        
        # Count bearish confirmations
        bearish_count = sum([
            scores['rsi'] < -0.3,
            scores['macd'] < -0.3,
            scores['ema'] < -0.3,
            scores['adx'] < -0.2,
            scores['obv'] < -0.2,
            scores['ichimoku'] < -0.3,
            scores['supertrend'] < 0,
            scores['bb'] < -0.3,
            scores['stoch'] < -0.3,
            scores['williams'] < -0.3
        ])
        
        # Determine signal strength
        if bullish_count >= min_confirmations + 2 and bullish_count > bearish_count + 2:
            return 'STRONG_BUY', bullish_count
        elif bullish_count >= min_confirmations and bullish_count > bearish_count:
            return 'BUY', bullish_count
        elif bearish_count >= min_confirmations + 2 and bearish_count > bullish_count + 2:
            return 'STRONG_SELL', bearish_count
        elif bearish_count >= min_confirmations and bearish_count > bullish_count:
            return 'SELL', bearish_count
        else:
            return 'NEUTRAL', max(bullish_count, bearish_count)
    
    def get_signal_explanation(self, scores: Dict) -> list:
        """
        Generate human-readable explanation of the signal.
        
        Returns:
            List of active signals
        """
        explanations = []
        
        if scores['rsi'] > 0.3:
            explanations.append("RSI oversold (bullish)")
        elif scores['rsi'] < -0.3:
            explanations.append("RSI overbought (bearish)")
        
        if scores['macd'] > 0.3:
            explanations.append("MACD bullish crossover")
        elif scores['macd'] < -0.3:
            explanations.append("MACD bearish crossover")
        
        if scores['adx'] > 0.3:
            explanations.append("Strong uptrend (ADX)")
        elif scores['adx'] < -0.3:
            explanations.append("Strong downtrend (ADX)")
        
        if scores['supertrend'] > 0:
            explanations.append("Supertrend bullish")
        elif scores['supertrend'] < 0:
            explanations.append("Supertrend bearish")
        
        if scores['ichimoku'] > 0.3:
            explanations.append("Above Ichimoku cloud")
        elif scores['ichimoku'] < -0.3:
            explanations.append("Below Ichimoku cloud")
        
        if scores['obv'] > 0.3:
            explanations.append("Volume confirms uptrend (OBV)")
        elif scores['obv'] < -0.3:
            explanations.append("Volume confirms downtrend (OBV)")
        
        return explanations


def trend_filter_ema_adx(
    data: pd.DataFrame,
    adx: pd.Series,
    ema_fast_period: int = 20,
    ema_slow_period: int = 50,
    adx_threshold: float = 22.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Build trend filter masks based on EMA alignment and ADX strength.

    Returns:
        (is_uptrend, is_downtrend) boolean Series
    """
    close = data['close']
    ema_fast = close.ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_period, adjust=False).mean()

    adx_ok = adx > adx_threshold
    is_uptrend = (ema_fast > ema_slow) & adx_ok
    is_downtrend = (ema_fast < ema_slow) & adx_ok
    return is_uptrend.fillna(False), is_downtrend.fillna(False)


def volume_spike_filter(
    data: pd.DataFrame,
    window: int = 20,
    multiplier: float = 1.2
) -> pd.Series:
    """
    Build volume spike mask:
    volume[-1] > volume.rolling(window).mean() * multiplier

    Returns:
        Boolean Series where True indicates volume spike confirmation
    """
    avg_volume = data['volume'].rolling(window=window).mean()
    spike = data['volume'] > (avg_volume * multiplier)
    return spike.fillna(False)


def create_signals_for_backtest(
    data: pd.DataFrame,
    scorer: EnsembleScorer,
    use_trend_filter: bool = False,
    trend_ema_fast: int = 20,
    trend_ema_slow: int = 50,
    trend_adx_threshold: float = 22.0,
    ignore_exit_signals: bool = False,
    use_volume_spike_filter: bool = False,
    volume_spike_window: int = 20,
    volume_spike_multiplier: float = 1.2
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate entry/exit signals for VectorBT backtesting.
    Vectorized version for performance.
    
    Returns:
        (entries, exits) as boolean Series
    """
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    
    # Warmup period
    warmup = 50
    
    # Calculate indicators once for all data
    indicators = scorer.calculate_indicators(data)
    
    if use_trend_filter:
        trend_up, _trend_down = trend_filter_ema_adx(
            data=data,
            adx=indicators['adx'],
            ema_fast_period=trend_ema_fast,
            ema_slow_period=trend_ema_slow,
            adx_threshold=trend_adx_threshold
        )
    else:
        trend_up = pd.Series(True, index=data.index)

    if use_volume_spike_filter:
        volume_spike = volume_spike_filter(
            data=data,
            window=volume_spike_window,
            multiplier=volume_spike_multiplier
        )
    else:
        volume_spike = pd.Series(True, index=data.index)

    # Calculate scores for each bar
    for i in range(warmup, len(data)):
        # Use indicators calculated up to this point
        current_indicators = {k: v.iloc[:i+1] for k, v in indicators.items()}
        
        # Get last values
        last_data = data.iloc[:i+1]
        scores = scorer.normalize_scores(current_indicators, last_data)
        
        # Calculate weighted score
        weighted_score = (
            scores['rsi'] * scorer.weights.rsi +
            scores['macd'] * scorer.weights.macd +
            scores['ema'] * scorer.weights.ema +
            scores['adx'] * scorer.weights.adx +
            scores['obv'] * scorer.weights.obv +
            scores['ichimoku'] * scorer.weights.ichimoku +
            scores['supertrend'] * scorer.weights.supertrend +
            scores['bb'] * scorer.weights.bb +
            scores['stoch'] * scorer.weights.stoch +
            scores['williams'] * scorer.weights.williams
        )
        
        # Generate signals
        if use_trend_filter:
            # Long-only behavior for spot:
            # 1) Enter only when signal is bullish and trend filter confirms.
            # 2) Exit when bearish signal appears OR trend confirmation is lost.
            if weighted_score > 0.3 and bool(trend_up.iloc[i]) and bool(volume_spike.iloc[i]):
                entries.iloc[i] = True
            elif (not ignore_exit_signals) and (weighted_score < -0.3 or (not bool(trend_up.iloc[i]))):
                exits.iloc[i] = True
        else:
            if weighted_score > 0.3 and bool(volume_spike.iloc[i]):
                entries.iloc[i] = True
            elif (not ignore_exit_signals) and weighted_score < -0.3:
                exits.iloc[i] = True
    
    return entries, exits


if __name__ == '__main__':
    # Test ensemble scoring
    print("Ensemble Scoring Test")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    trend = np.cumsum(np.random.randn(n) * 0.3)
    data = pd.DataFrame({
        'open': 100 + trend,
        'high': 100 + trend + 1,
        'low': 100 + trend - 1,
        'close': 100 + trend,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Create scorer
    scorer = EnsembleScorer()
    
    # Calculate score
    decision, confidence, score, components = scorer.calculate_ensemble_score(data)
    
    print(f"Decision: {decision}")
    print(f"Confidence: {confidence}")
    print(f"Final Score: {score:.3f}")
    print("\nComponent Scores:")
    for name, value in components.items():
        print(f"  {name:12}: {value:+.3f}")
    
    # Check confluence
    signal, count = scorer.check_confluence(data)
    print(f"\nConfluence Signal: {signal} ({count} confirmations)")
    
    # Get explanation
    explanations = scorer.get_signal_explanation(components)
    print(f"\nActive Signals:")
    for exp in explanations:
        print(f"  - {exp}")
