"""
Feature Engineering for ML Models
Creates features from technical indicators for XGBoost.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler

from utils.enhanced_indicators import EnhancedIndicators, HAS_TALIB

# Try to import TA-Lib
try:
    import talib
except ImportError:
    talib = None


class FeatureEngineer:
    """Engineer features from market data for ML."""
    
    def __init__(self):
        self.ei = EnhancedIndicators()
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_features(self, data: pd.DataFrame, include_lags: bool = True) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            data: OHLCV DataFrame
            include_lags: Include lagged features
            
        Returns:
            DataFrame with all features
        """
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features = self._add_price_features(features, df)
        
        # Technical indicators
        features = self._add_indicator_features(features, df)
        
        # Volume features
        features = self._add_volume_features(features, df)
        
        # Lagged features
        if include_lags:
            features = self._add_lagged_features(features)
        
        # Time features
        features = self._add_time_features(features, df)
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df['close']
        
        # Returns
        features['return_1h'] = close.pct_change(1)
        features['return_4h'] = close.pct_change(4)
        features['return_24h'] = close.pct_change(24)
        
        # Log returns
        features['log_return'] = np.log(close / close.shift(1))
        
        # Volatility
        features['volatility_20h'] = features['return_1h'].rolling(20).std()
        features['volatility_50h'] = features['return_1h'].rolling(50).std()
        
        # Price position
        features['close_to_high'] = close / df['high'].rolling(20).max()
        features['close_to_low'] = close / df['low'].rolling(20).min()
        
        # Price vs moving averages
        if HAS_TALIB and talib is not None:
            ema_10 = talib.EMA(close, timeperiod=10)
            ema_20 = talib.EMA(close, timeperiod=20)
            ema_50 = talib.EMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
        else:
            ema_10 = close.ewm(span=10, adjust=False).mean()
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()
            sma_200 = close.rolling(window=200).mean()
        
        features['close_over_ema10'] = close / ema_10
        features['close_over_ema20'] = close / ema_20
        features['close_over_ema50'] = close / ema_50
        features['close_over_sma200'] = close / sma_200
        
        features['ema10_over_ema20'] = ema_10 / ema_20
        features['ema20_over_ema50'] = ema_20 / ema_50
        
        # Trend strength
        features['price_momentum_10'] = close.pct_change(10)
        features['price_momentum_20'] = close.pct_change(20)
        
        return features
    
    def _add_indicator_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        close = df['close']
        high = df['high']
        low = df['low']
        
        if HAS_TALIB and talib is not None:
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            
            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)
            
            # CCI
            features['cci'] = talib.CCI(high, low, close)
            
            # MFI
            features['mfi'] = talib.MFI(high, low, close, df['volume'])
        else:
            # Pandas implementations
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            
            # Stochastic
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(window=3).mean()
            
            # Williams %R
            features['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            # CCI (simplified)
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mean_dev = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            features['cci'] = (tp - sma_tp) / (0.015 * mean_dev)
            
            # MFI (simplified - use RSI of typical price * volume)
            tp_vol = tp * df['volume']
            features['mfi'] = rsi  # Approximation
        
        features['rsi'] = rsi
        features['rsi_normalized'] = (rsi - 50) / 50  # Normalize to [-1, 1]
        features['rsi_slope'] = rsi.diff(5)  # RSI momentum
        
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        features['macd_hist_slope'] = macd_hist.diff(3)
        
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_diff'] = stoch_k - stoch_d
        
        # ADX (trend strength)
        adx, plus_di, minus_di = self.ei.adx_directional(df)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.ei.bollinger_bands(df)
        features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # ATR (volatility)
        atr = self.ei.atr(df)
        features['atr'] = atr
        features['atr_ratio'] = atr / close
        
        # Supertrend
        _, supertrend_dir = self.ei.supertrend(df)
        features['supertrend'] = supertrend_dir
        
        # Ichimoku
        ichi = self.ei.ichimoku(df)
        features['tenkan_kijun_diff'] = (ichi['tenkan_sen'] - ichi['kijun_sen']) / close
        features['price_over_cloud'] = (close - ichi['cloud_bottom']) / (ichi['cloud_top'] - ichi['cloud_bottom'])
        
        return features
    
    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df['volume']
        close = df['close']
        
        # Volume moving averages
        features['volume'] = volume
        features['volume_ma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_ma_20']
        
        # OBV
        obv, obv_ma, obv_delta = self.ei.obv_normalized(df)
        features['obv'] = obv
        features['obv_delta'] = obv_delta
        
        # Volume trend
        features['volume_trend'] = np.where(volume > volume.shift(1), 1, -1)
        
        # Price-Volume relationship
        features['price_volume_corr'] = close.rolling(20).corr(volume)
        
        # VWAP deviation
        vwap = self.ei.vwap(df)
        features['vwap_deviation'] = (close - vwap) / vwap
        
        return features
    
    def _add_lagged_features(self, features: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged versions of key features."""
        key_features = ['rsi', 'macd_hist', 'return_1h', 'adx', 'volume_ratio']
        
        for feature in key_features:
            if feature in features.columns:
                for lag in lags:
                    features[f'{feature}_lag_{lag}'] = features[feature].shift(lag)
        
        return features
    
    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Hour of day (0-23)
        features['hour'] = df.index.hour
        
        # Day of week (0-6)
        features['day_of_week'] = df.index.dayofweek
        
        # Is weekend
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # Cyclical encoding for day
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def create_target(self, data: pd.DataFrame, lookahead: int = 1) -> pd.Series:
        """
        Create target variable for classification.
        
        Args:
            data: OHLCV DataFrame
            lookahead: Number of periods to look ahead
            
        Returns:
            Series with target (1 = up, 0 = down)
        """
        future_return = data['close'].shift(-lookahead) / data['close'] - 1
        target = (future_return > 0).astype(int)
        return target
    
    def create_regression_target(self, data: pd.DataFrame, lookahead: int = 1) -> pd.Series:
        """
        Create target variable for regression (future returns).
        
        Args:
            data: OHLCV DataFrame
            lookahead: Number of periods to look ahead
            
        Returns:
            Series with future returns
        """
        return data['close'].shift(-lookahead) / data['close'] - 1
    
    def prepare_data(self, data: pd.DataFrame, 
                     train_ratio: float = 0.8,
                     scale: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare full dataset for ML training.
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # Create features
        features = self.create_features(data)
        
        # Create target
        target = self.create_target(data)
        
        # Remove NaN and inf rows
        features = features.replace([np.inf, -np.inf], np.nan)
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        # Clip extreme values
        features = features.clip(-1e6, 1e6)
        
        # Split train/test
        split_idx = int(len(features) * train_ratio)
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        # Scale features
        if scale:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
            return X_train_scaled, y_train, X_test_scaled, y_test
        
        return X_train, y_train, X_test, y_test
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names


if __name__ == '__main__':
    # Test feature engineering
    print("Feature Engineering Test")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2023-01-01', periods=n, freq='1h')
    
    trend = np.cumsum(np.random.randn(n) * 0.5)
    data = pd.DataFrame({
        'open': 100 + trend,
        'high': 100 + trend + np.abs(np.random.randn(n)),
        'low': 100 + trend - np.abs(np.random.randn(n)),
        'close': 100 + trend,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data)
    target = engineer.create_target(data)
    
    print(f"Created {features.shape[1]} features")
    print(f"Feature names: {features.columns.tolist()[:10]}...")
    print(f"\nSample features:")
    print(features.head())
    print(f"\nTarget distribution:")
    print(target.value_counts())
