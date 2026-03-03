"""
Enhanced Technical Indicators
Additional high-value indicators for crypto trading.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

# Try to import TA-Lib, provide fallback if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("WARNING: TA-Lib not installed. Using pandas implementations.")


class EnhancedIndicators:
    """Extended indicator library with crypto-optimized settings."""
    
    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX).
        Measures trend strength. Values > 25 indicate strong trend.
        """
        if HAS_TALIB:
            adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)
            return pd.Series(adx, index=data.index)
        else:
            # Pandas implementation
            return EnhancedIndicators._adx_pandas(data, period)
    
    @staticmethod
    def _adx_pandas(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Pandas-based ADX implementation."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        plus_dm[plus_dm <= minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0
        
        # Smooth TR and DM
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def adx_directional(data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ADX with +DI and -DI.
        Returns: (adx, plus_di, minus_di)
        """
        if HAS_TALIB:
            adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=period)
            plus_di = talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
            minus_di = talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=period)
        else:
            # Pandas implementation
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
        
        return pd.Series(adx, index=data.index), \
               pd.Series(plus_di, index=data.index), \
               pd.Series(minus_di, index=data.index)
    
    @staticmethod
    def obv(data: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV).
        Cumulative volume-based indicator. Confirms price trends.
        """
        if HAS_TALIB:
            obv = talib.OBV(data['close'], data['volume'])
        else:
            # Pandas implementation
            close_diff = data['close'].diff()
            volume = data['volume']
            
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(data)):
                if close_diff.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close_diff.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
        
        return pd.Series(obv, index=data.index)
    
    @staticmethod
    def obv_normalized(data: pd.DataFrame, ma_period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        OBV with moving average and delta.
        Returns: (obv, obv_ma, obv_delta)
        """
        obv = EnhancedIndicators.obv(data)
        obv_ma = obv.rolling(window=ma_period).mean()
        obv_delta = obv / obv_ma
        
        return pd.Series(obv, index=data.index), \
               pd.Series(obv_ma, index=data.index), \
               pd.Series(obv_delta, index=data.index)
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR).
        Volatility measure for dynamic stops and position sizing.
        """
        if HAS_TALIB:
            atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=period)
        else:
            # Pandas implementation
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
        
        return pd.Series(atr, index=data.index)
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, 
                       period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        Returns: (upper, middle, lower)
        """
        if HAS_TALIB:
            upper, middle, lower = talib.BBANDS(
                data['close'], 
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
        else:
            # Pandas implementation
            middle = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
        
        return pd.Series(upper, index=data.index), \
               pd.Series(middle, index=data.index), \
               pd.Series(lower, index=data.index)
    
    @staticmethod
    def supertrend(data: pd.DataFrame, 
                   period: int = 10, 
                   multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """
        Supertrend indicator.
        Returns: (supertrend_value, trend_direction)
        trend_direction: 1 = uptrend, -1 = downtrend
        """
        atr = EnhancedIndicators.atr(data, period)
        
        hl2 = (data['high'] + data['low']) / 2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = 1
            else:
                close = data['close'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                prev_supertrend = supertrend.iloc[i-1]
                
                if prev_close > prev_supertrend:
                    supertrend.iloc[i] = max(lower_band.iloc[i], prev_supertrend)
                    trend.iloc[i] = 1
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], prev_supertrend)
                    trend.iloc[i] = -1
        
        return supertrend, trend
    
    @staticmethod
    def ichimoku(data: pd.DataFrame,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52,
                 displacement: int = 26) -> dict:
        """
        Ichimoku Cloud indicator.
        Comprehensive trend, support/resistance, and momentum indicator.
        
        Returns dict with:
        - tenkan_sen: Conversion line
        - kijun_sen: Base line
        - senkou_span_a: Leading span A
        - senkou_span_b: Leading span B
        - chikou_span: Lagging span
        - cloud_top: Upper cloud boundary
        - cloud_bottom: Lower cloud boundary
        """
        # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for 9 periods
        tenkan_high = data['high'].rolling(window=tenkan_period).max()
        tenkan_low = data['low'].rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for 26 periods
        kijun_high = data['high'].rolling(window=kijun_period).max()
        kijun_low = data['low'].rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26 periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for 52 periods, shifted forward
        senkou_b_high = data['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = data['low'].rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Close shifted backward 26 periods
        chikou_span = data['close'].shift(-displacement)
        
        # Cloud boundaries
        cloud_top = pd.concat([senkou_span_a, senkou_span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_span_a, senkou_span_b], axis=1).min(axis=1)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom
        }
    
    @staticmethod
    def ichimoku_signal(ichimoku_data: dict, current_price: float) -> str:
        """
        Generate signal from Ichimoku data.
        
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        tenkan = ichimoku_data['tenkan_sen'].iloc[-1]
        kijun = ichimoku_data['kijun_sen'].iloc[-1]
        cloud_top = ichimoku_data['cloud_top'].iloc[-1]
        cloud_bottom = ichimoku_data['cloud_bottom'].iloc[-1]
        
        # Bullish conditions
        if (current_price > cloud_top and 
            tenkan > kijun and 
            current_price > tenkan):
            return 'BULLISH'
        
        # Bearish conditions
        elif (current_price < cloud_bottom and 
              tenkan < kijun and 
              current_price < tenkan):
            return 'BEARISH'
        
        return 'NEUTRAL'
    
    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP).
        Important intraday support/resistance level.
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return pd.Series(vwap, index=data.index)
    
    @staticmethod
    def vwap_daily(data: pd.DataFrame) -> pd.Series:
        """
        Resetting VWAP (resets daily).
        More relevant for intraday trading.
        """
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Group by date and calculate VWAP for each day
        data_copy = data.copy()
        data_copy['typical_price'] = typical_price
        data_copy['tp_vol'] = typical_price * data['volume']
        
        vwap = data_copy.groupby(data_copy.index.date).apply(
            lambda x: x['tp_vol'].cumsum() / x['volume'].cumsum()
        ).reset_index(level=0, drop=True)
        
        return vwap
    
    @staticmethod
    def rsi_divergence(data: pd.DataFrame, rsi_period: int = 14, 
                       lookback: int = 20) -> Tuple[bool, bool]:
        """
        Detect RSI divergence.
        
        Returns: (bullish_divergence, bearish_divergence)
        Bullish: Price makes lower low, RSI makes higher low
        Bearish: Price makes higher high, RSI makes lower high
        """
        if HAS_TALIB:
            rsi = talib.RSI(data['close'], timeperiod=rsi_period)
        else:
            # Pandas RSI implementation
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if len(data) < lookback:
            return False, False
        
        # Get recent window
        price_window = data['close'].iloc[-lookback:]
        rsi_window = rsi.iloc[-lookback:]
        
        # Simplified divergence detection
        bullish_div = False
        bearish_div = False
        
        if (price_window.iloc[-1] < price_window.iloc[-lookback//2] and 
            rsi_window.iloc[-1] > rsi_window.iloc[-lookback//2]):
            bullish_div = True
            
        if (price_window.iloc[-1] > price_window.iloc[-lookback//2] and 
            rsi_window.iloc[-1] < rsi_window.iloc[-lookback//2]):
            bearish_div = True
        
        return bullish_div, bearish_div
    
    @staticmethod
    def chandelier_exit(data: pd.DataFrame, period: int = 22, multiplier: float = 3.0) -> pd.Series:
        """
        Chandelier Exit.
        Volatility-based exit for trends.
        """
        atr = EnhancedIndicators.atr(data, period)
        highest_high = data['high'].rolling(window=period).max()
        
        chandelier = highest_high - (multiplier * atr)
        return pd.Series(chandelier, index=data.index)
    
    @staticmethod
    def keltner_channels(data: pd.DataFrame, 
                        ema_period: int = 20, 
                        atr_period: int = 10,
                        multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        Similar to Bollinger Bands but uses ATR.
        """
        if HAS_TALIB:
            middle = talib.EMA(data['close'], timeperiod=ema_period)
        else:
            middle = data['close'].ewm(span=ema_period, adjust=False).mean()
        
        atr = EnhancedIndicators.atr(data, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return pd.Series(upper, index=data.index), \
               pd.Series(middle, index=data.index), \
               pd.Series(lower, index=data.index)
    
    @staticmethod
    def market_structure(data: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Analyze market structure (higher highs, lower lows).
        
        Returns dict with trend structure information.
        """
        highs = data['high'].rolling(window=lookback).max()
        lows = data['low'].rolling(window=lookback).min()
        
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        prev_high = highs.iloc[-2] if len(highs) > 1 else current_high
        prev_low = lows.iloc[-2] if len(lows) > 1 else current_low
        
        # Determine structure
        higher_high = current_high > prev_high
        higher_low = current_low > prev_low
        lower_high = current_high < prev_high
        lower_low = current_low < prev_low
        
        # Trend determination
        if higher_high and higher_low:
            structure = 'BULLISH'
        elif lower_high and lower_low:
            structure = 'BEARISH'
        else:
            structure = 'CONSOLIDATING'
        
        return {
            'structure': structure,
            'higher_high': higher_high,
            'higher_low': higher_low,
            'lower_high': lower_high,
            'lower_low': lower_low,
            'recent_high': prev_high,
            'recent_low': prev_low
        }
    
    @staticmethod
    def volume_profile(data: pd.DataFrame, bins: int = 20) -> dict:
        """
        Calculate volume profile (POC, Value Area).
        
        Returns dict with volume profile levels.
        """
        # Price range
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            low = data['low'].min() + (i * bin_size)
            high = low + bin_size
            
            mask = (data['close'] >= low) & (data['close'] < high)
            volume = data.loc[mask, 'volume'].sum()
            
            volume_profile[(low + high) / 2] = volume
        
        # Point of Control (highest volume)
        poc = max(volume_profile, key=volume_profile.get)
        
        # Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.70
        
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        cumulative = 0
        value_area = []
        
        for level, vol in sorted_levels:
            cumulative += vol
            value_area.append(level)
            if cumulative >= target_volume:
                break
        
        return {
            'poc': poc,
            'value_area_high': max(value_area),
            'value_area_low': min(value_area),
            'volume_profile': volume_profile
        }


# Convenience function for getting all enhanced indicators
def get_all_indicators(data: pd.DataFrame) -> dict:
    """
    Calculate all enhanced indicators at once.
    
    Returns dictionary with all indicators.
    """
    ei = EnhancedIndicators()
    
    adx, plus_di, minus_di = ei.adx_directional(data)
    obv, obv_ma, obv_delta = ei.obv_normalized(data)
    atr = ei.atr(data)
    bb_upper, bb_middle, bb_lower = ei.bollinger_bands(data)
    supertrend_val, supertrend_dir = ei.supertrend(data)
    ichimoku_data = ei.ichimoku(data)
    vwap = ei.vwap(data)
    kelt_upper, kelt_middle, kelt_lower = ei.keltner_channels(data)
    market_struct = ei.market_structure(data)
    
    return {
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'obv': obv,
        'obv_ma': obv_ma,
        'obv_delta': obv_delta,
        'atr': atr,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'supertrend': supertrend_val,
        'supertrend_dir': supertrend_dir,
        **ichimoku_data,
        'vwap': vwap,
        'keltner_upper': kelt_upper,
        'keltner_middle': kelt_middle,
        'keltner_lower': kelt_lower,
        **market_struct
    }


if __name__ == '__main__':
    # Test enhanced indicators
    print("Enhanced Indicators Test")
    print(f"TA-Lib Available: {HAS_TALIB}")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(n) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(n) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000, 10000, n)
    })
    
    ei = EnhancedIndicators()
    
    # Test ADX
    adx = ei.adx(data)
    print(f"ADX (last): {adx.iloc[-1]:.2f}")
    
    # Test OBV
    obv, obv_ma, obv_delta = ei.obv_normalized(data)
    print(f"OBV Delta (last): {obv_delta.iloc[-1]:.2f}")
    
    # Test ATR
    atr = ei.atr(data)
    print(f"ATR (last): {atr.iloc[-1]:.2f}")
    
    # Test Bollinger Bands
    bb_upper, bb_middle, bb_lower = ei.bollinger_bands(data)
    print(f"BB Width: {(bb_upper.iloc[-1] - bb_lower.iloc[-1]):.2f}")
    
    # Test Supertrend
    st, st_dir = ei.supertrend(data)
    print(f"Supertrend Direction: {st_dir.iloc[-1]}")
    
    # Test Ichimoku
    ichi = ei.ichimoku(data)
    print(f"Ichimoku Signal: {ei.ichimoku_signal(ichi, data['close'].iloc[-1])}")
    
    print("\nAll indicators calculated successfully!")
