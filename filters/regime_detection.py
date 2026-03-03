"""
Market Regime Detection
Identifies market conditions using HMM and clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detect market regime (trending, ranging, volatile)."""
    
    def __init__(self, method: str = 'hmm'):
        """
        Initialize regime detector.
        
        Args:
            method: 'hmm' (Hidden Markov Model) or 'kmeans'
        """
        self.method = method
        self.model = None
        self.regime_labels = {
            0: 'TRENDING_UP',
            1: 'TRENDING_DOWN', 
            2: 'RANGING',
            3: 'HIGH_VOLATILITY'
        }
        
    def fit(self, data: pd.DataFrame, n_regimes: int = 3):
        """
        Fit regime detection model.
        
        Args:
            data: OHLCV DataFrame
            n_regimes: Number of regimes to detect
        """
        # Prepare features
        features = self._prepare_regime_features(data)
        
        if self.method == 'hmm':
            self._fit_hmm(features, n_regimes)
        else:
            self._fit_kmeans(features, n_regimes)
            
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict current regime.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Series with regime labels
        """
        if self.model is None:
            raise ValueError("Fit model first")
        
        features = self._prepare_regime_features(data)
        
        if self.method == 'hmm':
            regimes = self.model.predict(features)
        else:
            regimes = self.model.predict(features)
        
        return pd.Series(regimes, index=data.index)
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for regime detection.
        
        Features:
        - Returns (current volatility)
        - Trend strength (ADX proxy)
        - Price position within range
        """
        features = pd.DataFrame(index=data.index)
        
        # Returns
        features['return'] = data['close'].pct_change()
        
        # Volatility (absolute returns)
        features['volatility'] = features['return'].abs()
        
        # Trend strength (simple proxy)
        features['trend'] = data['close'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # Range position
        rolling_high = data['high'].rolling(20).max()
        rolling_low = data['low'].rolling(20).min()
        features['range_position'] = (data['close'] - rolling_low) / (rolling_high - rolling_low)
        
        # Drop NaN
        features = features.dropna()
        
        # Normalize
        features = (features - features.mean()) / features.std()
        
        return features.values
    
    def _fit_hmm(self, features: np.ndarray, n_components: int = 3):
        """Fit Hidden Markov Model."""
        try:
            from hmmlearn import hmm
            
            self.model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type="full",
                random_state=42
            )
            self.model.fit(features)
            
            logger.info(f"HMM fitted with {n_components} regimes")
            
        except ImportError:
            logger.warning("hmmlearn not installed. Using KMeans instead.")
            self._fit_kmeans(features, n_components)
    
    def _fit_kmeans(self, features: np.ndarray, n_clusters: int = 3):
        """Fit K-Means clustering."""
        from sklearn.cluster import KMeans
        
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )
        self.model.fit(features)
        
        logger.info(f"KMeans fitted with {n_clusters} clusters")


class SimpleRegimeDetector:
    """Rule-based regime detection (no ML required)."""
    
    def __init__(self):
        self.regimes = []
        
    def detect_regime(self, data: pd.DataFrame, lookback: int = 50) -> str:
        """
        Detect regime using simple rules.
        
        Args:
            data: OHLCV DataFrame
            lookback: Period for analysis
            
        Returns:
            Regime label
        """
        if len(data) < lookback:
            return 'UNKNOWN'
        
        recent = data.iloc[-lookback:]
        
        # Calculate metrics
        returns = recent['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(lookback)
        
        # Trend strength
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Range analysis
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        
        # ADX if available
        try:
            from utils.enhanced_indicators import EnhancedIndicators
            adx = EnhancedIndicators.adx(recent, period=14).iloc[-1]
        except:
            adx = 20  # Default
        
        # Regime detection logic
        if volatility > 0.05:  # High volatility
            if adx > 25:
                return 'TRENDING_VOLATILE'
            else:
                return 'CHOPPY_VOLATILE'
        
        if adx > 30:  # Strong trend
            if price_change > 0.05:
                return 'STRONG_UPTREND'
            elif price_change < -0.05:
                return 'STRONG_DOWNTREND'
        
        if adx > 20:  # Moderate trend
            if price_change > 0.02:
                return 'WEAK_UPTREND'
            elif price_change < -0.02:
                return 'WEAK_DOWNTREND'
        
        # Range bound
        if price_range < 0.05:
            return 'TIGHT_RANGE'
        else:
            return 'WIDE_RANGE'
    
    def get_regime_adaptation(self, regime: str) -> Dict:
        """
        Get strategy adaptations for regime.
        
        Returns:
            Dictionary with adaptation parameters
        """
        adaptations = {
            'STRONG_UPTREND': {
                'position_size_mult': 1.2,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 4.0,
                'trailing_stop': True,
                'preferred_strategy': 'trend_following',
                'trade_direction': 'long_only'
            },
            'STRONG_DOWNTREND': {
                'position_size_mult': 1.2,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 4.0,
                'trailing_stop': True,
                'preferred_strategy': 'trend_following',
                'trade_direction': 'short_only'
            },
            'WEAK_UPTREND': {
                'position_size_mult': 1.0,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 3.0,
                'trailing_stop': False,
                'preferred_strategy': 'balanced',
                'trade_direction': 'long_bias'
            },
            'WEAK_DOWNTREND': {
                'position_size_mult': 1.0,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 3.0,
                'trailing_stop': False,
                'preferred_strategy': 'balanced',
                'trade_direction': 'short_bias'
            },
            'TIGHT_RANGE': {
                'position_size_mult': 0.8,
                'stop_loss_atr_mult': 1.0,
                'take_profit_atr_mult': 2.0,
                'trailing_stop': False,
                'preferred_strategy': 'mean_reversion',
                'trade_direction': 'both'
            },
            'WIDE_RANGE': {
                'position_size_mult': 0.9,
                'stop_loss_atr_mult': 1.5,
                'take_profit_atr_mult': 3.0,
                'trailing_stop': False,
                'preferred_strategy': 'breakout',
                'trade_direction': 'both'
            },
            'CHOPPY_VOLATILE': {
                'position_size_mult': 0.5,
                'stop_loss_atr_mult': 3.0,
                'take_profit_atr_mult': 2.0,
                'trailing_stop': False,
                'preferred_strategy': 'reduce_exposure',
                'trade_direction': 'none'
            },
            'TRENDING_VOLATILE': {
                'position_size_mult': 0.8,
                'stop_loss_atr_mult': 3.0,
                'take_profit_atr_mult': 5.0,
                'trailing_stop': True,
                'preferred_strategy': 'trend_with_wide_stops',
                'trade_direction': 'both'
            },
            'UNKNOWN': {
                'position_size_mult': 1.0,
                'stop_loss_atr_mult': 2.0,
                'take_profit_atr_mult': 3.0,
                'trailing_stop': False,
                'preferred_strategy': 'default',
                'trade_direction': 'both'
            }
        }
        
        return adaptations.get(regime, adaptations['UNKNOWN'])


class VolatilityRegime:
    """Detect volatility regime for position sizing."""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        
    def get_volatility_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate current volatility relative to historical.
        
        Returns:
            Volatility ratio (>1 means higher than normal)
        """
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < self.long_window:
            return 1.0
        
        short_vol = returns.iloc[-self.short_window:].std()
        long_vol = returns.iloc[-self.long_window:].std()
        
        if long_vol == 0:
            return 1.0
        
        return short_vol / long_vol
    
    def get_regime(self, data: pd.DataFrame) -> str:
        """
        Classify volatility regime.
        
        Returns:
            'LOW', 'NORMAL', 'HIGH', or 'EXTREME'
        """
        ratio = self.get_volatility_ratio(data)
        
        if ratio < 0.7:
            return 'LOW'
        elif ratio < 1.3:
            return 'NORMAL'
        elif ratio < 2.0:
            return 'HIGH'
        else:
            return 'EXTREME'
    
    def get_position_size_multiplier(self, data: pd.DataFrame) -> float:
        """
        Get position size multiplier based on volatility.
        
        Returns:
            Multiplier (0.5 = half size, 1.0 = normal)
        """
        regime = self.get_regime(data)
        
        multipliers = {
            'LOW': 1.2,
            'NORMAL': 1.0,
            'HIGH': 0.7,
            'EXTREME': 0.5
        }
        
        return multipliers[regime]


if __name__ == '__main__':
    print("Regime Detection Test")
    print("=" * 60)
    
    # Generate sample data with different regimes
    np.random.seed(42)
    n = 500
    
    # Trending period
    trend = np.cumsum(np.ones(n//3) * 0.1 + np.random.randn(n//3) * 0.5)
    
    # Ranging period
    range_period = 100 + np.sin(np.linspace(0, 4*np.pi, n//3)) * 10 + np.random.randn(n//3) * 2
    
    # Volatile period
    volatile = np.cumsum(np.random.randn(n - 2*(n//3)) * 2)
    
    prices = np.concatenate([trend, range_period, volatile]) + 100
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n)),
        'low': prices - np.abs(np.random.randn(n)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Test simple detector
    detector = SimpleRegimeDetector()
    
    # Test different windows
    for i in range(100, n, 100):
        window = data.iloc[i-50:i]
        regime = detector.detect_regime(window)
        adaptation = detector.get_regime_adaptation(regime)
        
        print(f"Period {i-50}-{i}: {regime}")
        print(f"  Position size: {adaptation['position_size_mult']}x")
        print(f"  Strategy: {adaptation['preferred_strategy']}")
    
    # Test volatility regime
    vol_regime = VolatilityRegime()
    ratio = vol_regime.get_volatility_ratio(data)
    vol_regime_label = vol_regime.get_regime(data)
    
    print(f"\nCurrent Volatility Ratio: {ratio:.2f}")
    print(f"Volatility Regime: {vol_regime_label}")
    print(f"Position Size Multiplier: {vol_regime.get_position_size_multiplier(data):.2f}x")
