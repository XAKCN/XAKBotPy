"""
Data pipeline for backtesting.
Manages historical OHLCV data from Binance Spot.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from exchange.binance_spot_client import BinanceSpotTrader, HAS_BINANCE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Manages historical data for backtesting from Binance Spot."""

    def __init__(
        self,
        data_dir: str = "data",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        recv_window: int = 10000
    ):
        self.data_dir = data_dir
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.recv_window = recv_window
        os.makedirs(data_dir, exist_ok=True)

    def _interval_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes."""
        mapping = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "3d": 4320,
            "1w": 10080,
        }
        return mapping.get(timeframe.lower(), 60)

    def _estimate_limit(
        self,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str],
        hard_cap: int = 1500
    ) -> int:
        """
        Estimate number of candles from date range.
        Binance endpoint limit is max 1500 per request.
        """
        if start_date is None and end_date is None:
            return hard_cap

        if start_date is None:
            start_dt = datetime.now() - timedelta(days=365)
        else:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        minutes = max(1, self._interval_to_minutes(timeframe))
        delta_minutes = max(1, int((end_dt - start_dt).total_seconds() / 60))
        estimated = max(50, min(hard_cap, delta_minutes // minutes))
        return estimated

    def _download_batch(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Download one batch from Binance Spot."""
        if not HAS_BINANCE:
            raise ImportError("python-binance package required. Install with: pip install python-binance")

        trader = BinanceSpotTrader(
            symbol=symbol,
            timeframe=timeframe,
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet,
            recv_window=self.recv_window
        )
        if not trader.initialized:
            raise ConnectionError("Failed to connect to Binance Spot")

        data = trader.get_market_data(limit=limit)
        if data is None or data.empty:
            return pd.DataFrame()
        return data

    def download_from_binance_spot(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_format: str = "csv"
    ) -> pd.DataFrame:
        """
        Download historical data from Binance Spot.

        Args:
            symbol: Trading symbol (example: BTCUSDT).
            timeframe: Candle interval (1m, 5m, 15m, 1h, 4h, 1d).
            start_date: Start date (YYYY-MM-DD) used to estimate sample size.
            end_date: End date (YYYY-MM-DD) used to estimate sample size.
            save_format: csv or parquet.
        """
        limit = self._estimate_limit(timeframe, start_date, end_date)
        logger.info(
            "Downloading %s %s data from Binance Spot (%s to %s, limit=%s)",
            symbol,
            timeframe,
            start_date or "auto",
            end_date or "now",
            limit
        )

        df = self._download_batch(symbol=symbol, timeframe=timeframe, limit=limit)
        if df.empty:
            logger.error("No data received from Binance Spot for %s", symbol)
            return df

        df = self._calculate_features(df)
        self._save_data(df, symbol, timeframe, save_format)
        logger.info("Downloaded %s candles from Binance Spot", len(df))
        return df

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for ML and analysis."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        df["volatility"] = df["returns"].rolling(window=20).std()
        df["price_change"] = df["close"] - df["open"]
        df["price_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100

        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"] * 100

        df["tr1"] = df["high"] - df["low"]
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["true_range"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        return df

    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, fmt: str):
        """Save data to disk."""
        filename = f"{symbol}_{timeframe}"
        if fmt == "parquet":
            try:
                filepath = os.path.join(self.data_dir, f"{filename}.parquet")
                df.to_parquet(filepath, compression="zstd")
                logger.info("Data saved to %s", filepath)
                return
            except ImportError:
                logger.warning("pyarrow not installed, saving CSV instead")

        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        df.to_csv(filepath)
        logger.info("Data saved to %s", filepath)

    def load_data(self, symbol: str, timeframe: str, fmt: str = "csv") -> Optional[pd.DataFrame]:
        """Load data from disk."""
        filename = f"{symbol}_{timeframe}"

        if fmt == "parquet":
            try:
                filepath = os.path.join(self.data_dir, f"{filename}.parquet")
                if os.path.exists(filepath):
                    return pd.read_parquet(filepath)
            except ImportError:
                logger.warning("pyarrow not installed, trying CSV")
                fmt = "csv"

        if fmt == "csv":
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            if os.path.exists(filepath):
                return pd.read_csv(filepath, index_col=0, parse_dates=True)

        logger.warning("Data file not found for %s %s", symbol, timeframe)
        return None

    def get_multi_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str] = ["1h", "4h", "1d"]
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes."""
        result: Dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            df = self.load_data(symbol, timeframe)
            if df is None:
                logger.info("Downloading %s data...", timeframe)
                df = self.download_from_binance_spot(symbol, timeframe)
            result[timeframe] = df
        return result


if __name__ == "__main__":
    print("Binance Spot Data Pipeline Example")
    print("=" * 60)

    pipeline = DataPipeline(testnet=True)
    try:
        df = pipeline.download_from_binance_spot(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2025-01-01",
            end_date="2026-01-01"
        )
        print(f"Downloaded {len(df)} candles")
        print(df.head())
    except Exception as error:
        print(f"Error: {error}")
