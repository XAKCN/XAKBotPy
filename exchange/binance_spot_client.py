"""
Binance Spot client for market data and order execution.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional
import logging
import uuid

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False
    Client = None
    BinanceAPIException = Exception
    logger.warning("python-binance not installed. Run: pip install python-binance")


class BinanceSpotTrader:
    """Interface for trading via Binance Spot."""

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        recv_window: int = 10000
    ):
        if not HAS_BINANCE:
            raise ImportError("python-binance required. Install with: pip install python-binance")

        self.requested_symbol = symbol
        self.symbol = symbol.upper()
        self.timeframe_str = timeframe
        self.interval = self._convert_interval(timeframe)
        self.api_key = api_key or ""
        self.api_secret = api_secret or ""
        self.testnet = testnet
        self.recv_window = recv_window

        self.client: Optional[Client] = None
        self.initialized = False
        self._exchange_info: Optional[Dict] = None
        self.symbol_info: Optional[Dict] = None
        self.symbol_filters: Dict[str, Dict] = {}

        self._initialize()

    def _initialize(self) -> bool:
        """Initialize Binance client and resolve symbol metadata."""
        try:
            self.client = Client(self.api_key, self.api_secret, requests_params={"timeout": 20})
            if self.testnet:
                self.client.API_URL = "https://testnet.binance.vision/api"

            self._exchange_info = self.client.get_exchange_info()
            resolved_symbol = self._resolve_symbol(self.requested_symbol)
            if not resolved_symbol:
                logger.error("Unable to resolve spot symbol: %s", self.requested_symbol)
                return False

            self.symbol = resolved_symbol
            self.symbol_info = self._get_symbol_info(self.symbol)
            if self.symbol_info is None:
                logger.error("Unable to load symbol info: %s", self.symbol)
                return False

            self.symbol_filters = {
                item["filterType"]: item for item in self.symbol_info.get("filters", [])
            }

            logger.info(
                "Binance Spot initialized | requested=%s | resolved=%s | timeframe=%s | testnet=%s",
                self.requested_symbol,
                self.symbol,
                self.timeframe_str,
                self.testnet
            )
            logger.info(
                "Symbol spec | qty_step=%s | min_qty=%s | tick_size=%s",
                self._get_quantity_step(),
                self._get_min_qty(),
                self._get_tick_size()
            )
            self.initialized = True
            return True
        except Exception as error:
            logger.error(f"Binance initialization error: {error}")
            return False

    def _convert_interval(self, timeframe: str) -> str:
        """Convert timeframe to Binance interval."""
        tf_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE if HAS_BINANCE else "1m",
            "3m": Client.KLINE_INTERVAL_3MINUTE if HAS_BINANCE else "3m",
            "5m": Client.KLINE_INTERVAL_5MINUTE if HAS_BINANCE else "5m",
            "15m": Client.KLINE_INTERVAL_15MINUTE if HAS_BINANCE else "15m",
            "30m": Client.KLINE_INTERVAL_30MINUTE if HAS_BINANCE else "30m",
            "1h": Client.KLINE_INTERVAL_1HOUR if HAS_BINANCE else "1h",
            "2h": Client.KLINE_INTERVAL_2HOUR if HAS_BINANCE else "2h",
            "4h": Client.KLINE_INTERVAL_4HOUR if HAS_BINANCE else "4h",
            "6h": Client.KLINE_INTERVAL_6HOUR if HAS_BINANCE else "6h",
            "8h": Client.KLINE_INTERVAL_8HOUR if HAS_BINANCE else "8h",
            "12h": Client.KLINE_INTERVAL_12HOUR if HAS_BINANCE else "12h",
            "1d": Client.KLINE_INTERVAL_1DAY if HAS_BINANCE else "1d",
            "1w": Client.KLINE_INTERVAL_1WEEK if HAS_BINANCE else "1w",
        }
        return tf_map.get(timeframe.lower(), Client.KLINE_INTERVAL_1HOUR if HAS_BINANCE else "1h")

    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Return symbol info from cached exchange info."""
        if not self._exchange_info:
            return None
        for item in self._exchange_info.get("symbols", []):
            if item.get("symbol") == symbol:
                return item
        return None

    def _resolve_symbol(self, requested_symbol: str) -> Optional[str]:
        """Resolve symbol among available spot symbols."""
        if not self._exchange_info:
            return None

        normalized = requested_symbol.replace("/", "").replace("-", "").upper()
        candidates = [normalized]
        if not normalized.endswith("USDT"):
            candidates.append(f"{normalized}USDT")
        if normalized.endswith("USD"):
            candidates.append(normalized.replace("USD", "USDT"))

        available = {
            item.get("symbol"): item
            for item in self._exchange_info.get("symbols", [])
            if item.get("status") == "TRADING"
        }

        for candidate in candidates:
            if candidate in available:
                return candidate

        for symbol_name in available.keys():
            if symbol_name.startswith(normalized) or normalized in symbol_name:
                return symbol_name
        return None

    def _to_decimal(self, value: float | str) -> Decimal:
        return Decimal(str(value))

    def _get_lot_filter(self) -> Dict:
        return self.symbol_filters.get("LOT_SIZE", {})

    def _get_price_filter(self) -> Dict:
        return self.symbol_filters.get("PRICE_FILTER", {})

    def _get_quantity_step(self) -> Decimal:
        lot_filter = self._get_lot_filter()
        return self._to_decimal(lot_filter.get("stepSize", "0.000001"))

    def _get_min_qty(self) -> Decimal:
        lot_filter = self._get_lot_filter()
        return self._to_decimal(lot_filter.get("minQty", "0.000001"))

    def _get_max_qty(self) -> Decimal:
        lot_filter = self._get_lot_filter()
        return self._to_decimal(lot_filter.get("maxQty", "100000"))

    def _get_tick_size(self) -> Decimal:
        price_filter = self._get_price_filter()
        return self._to_decimal(price_filter.get("tickSize", "0.01"))

    def _quantize_down(self, value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        steps = (value / step).quantize(Decimal("1"), rounding=ROUND_DOWN)
        return steps * step

    def _normalize_quantity(self, quantity: float) -> float:
        """Normalize quantity by LOT_SIZE rules."""
        qty = self._to_decimal(quantity)
        if qty <= 0:
            return 0.0

        step = self._get_quantity_step()
        min_qty = self._get_min_qty()
        max_qty = self._get_max_qty()

        qty = self._quantize_down(qty, step)
        if qty < min_qty:
            return 0.0
        if qty > max_qty:
            qty = self._quantize_down(max_qty, step)
        return float(qty)

    def _normalize_price(self, price: float) -> float:
        """Normalize price by PRICE_FILTER tick size."""
        px = self._to_decimal(price)
        if px <= 0:
            return 0.0
        tick = self._get_tick_size()
        px = self._quantize_down(px, tick)
        return float(px)

    def _extract_assets(self) -> Dict[str, str]:
        """Return base and quote assets for current symbol."""
        if not self.symbol_info:
            return {"base": "BTC", "quote": "USDT"}
        return {
            "base": self.symbol_info.get("baseAsset", "BTC"),
            "quote": self.symbol_info.get("quoteAsset", "USDT"),
        }

    def calculate_order_quantity(self, position_size_units: float) -> float:
        """Convert strategy size to Binance spot order quantity."""
        return self._normalize_quantity(position_size_units)

    def _fetch_klines(self, limit: int) -> List[List]:
        """Fetch spot klines with pagination for limits above 1000."""
        if not self.client:
            return []

        remaining = max(1, limit)
        end_time: Optional[int] = None
        collected: List[List] = []

        while remaining > 0:
            batch_limit = min(1000, remaining)
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": batch_limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            batch = self.client.get_klines(**params)
            if not batch:
                break

            collected = batch + collected
            remaining -= len(batch)

            oldest_open_time = int(batch[0][0])
            end_time = oldest_open_time - 1

            if len(batch) < batch_limit:
                break

        # Deduplicate and keep only latest `limit` candles.
        unique = {}
        for row in collected:
            unique[int(row[0])] = row
        ordered = [unique[key] for key in sorted(unique.keys())]
        return ordered[-limit:]

    def get_market_data(self, limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data from Binance Spot."""
        if not self.initialized or self.client is None:
            logger.error("Binance client not initialized")
            return None

        try:
            klines = self._fetch_klines(limit=min(max(limit, 50), 5000))
            if not klines:
                logger.error("No kline data received for %s", self.symbol)
                return None

            df = pd.DataFrame(klines, columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ])
            df["time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("time", inplace=True)

            for column in ["open", "high", "low", "close", "volume"]:
                df[column] = df[column].astype(float)

            logger.info("Binance spot data received: %s candles for %s", len(df), self.symbol)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as error:
            logger.error(f"Error fetching Binance spot data: {error}")
            return None

    def get_current_price(self) -> Optional[Dict]:
        """Get latest spot price."""
        if not self.initialized or self.client is None:
            return None
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            last_price = float(ticker["price"])
            return {
                "bid": last_price,
                "ask": last_price,
                "last": last_price,
                "time": datetime.utcnow(),
                "volume": 0.0,
            }
        except Exception as error:
            logger.error(f"Error getting current price: {error}")
            return None

    def _get_balance_entry(self, asset: str) -> Optional[Dict]:
        """Get spot asset balance entry."""
        if not self.initialized or self.client is None:
            return None
        if not self.api_key or not self.api_secret:
            return None

        try:
            balance = self.client.get_asset_balance(asset=asset, recvWindow=self.recv_window)
            return balance
        except BinanceAPIException as error:
            logger.error("Binance API error (balance %s): %s", asset, error)
            return None
        except Exception as error:
            logger.error("Error getting balance %s: %s", asset, error)
            return None

    def get_asset_balance(self, asset: str) -> Dict[str, float]:
        """Return free/locked/total balance for one asset."""
        entry = self._get_balance_entry(asset)
        if entry is None:
            return {"asset": asset, "free": 0.0, "locked": 0.0, "total": 0.0}

        free = float(entry.get("free", 0.0))
        locked = float(entry.get("locked", 0.0))
        return {
            "asset": asset,
            "free": free,
            "locked": locked,
            "total": free + locked,
        }

    def get_quote_asset_free(self) -> float:
        assets = self._extract_assets()
        return self.get_asset_balance(assets["quote"])["free"]

    def get_base_asset_free(self) -> float:
        assets = self._extract_assets()
        return self.get_asset_balance(assets["base"])["free"]

    def get_account_info(self) -> Optional[Dict]:
        """Get spot account summary in quote currency."""
        if not self.initialized or self.client is None:
            return None
        if not self.api_key or not self.api_secret:
            return None

        assets = self._extract_assets()
        quote_asset = assets["quote"]

        quote_balance = self.get_asset_balance(quote_asset)
        if quote_balance["total"] == 0 and quote_balance["free"] == 0 and quote_balance["locked"] == 0:
            # Either true zero or API inaccessible; verify connectivity with get_account.
            try:
                self.client.get_account(recvWindow=self.recv_window)
            except BinanceAPIException as error:
                logger.error("Binance API error (account info): %s", error)
                return None
            except Exception as error:
                logger.error("Error getting account info: %s", error)
                return None

        balance_total = quote_balance["total"]
        balance_free = quote_balance["free"]

        return {
            "balance": balance_total,
            "equity": balance_total,
            "profit": 0.0,
            "margin": 0.0,
            "margin_free": balance_free,
            "margin_level": 0.0,
            "currency": quote_asset,
            "leverage": 1.0,
            "quote_free": balance_free,
        }

    def _normalize_side(self, side: str) -> Optional[str]:
        value = side.strip().upper()
        if value in {"BUY", "LONG"}:
            return "BUY"
        if value in {"SELL", "SHORT"}:
            return "SELL"
        return None

    def execute_market_order(
        self,
        side: str,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "XAKCN Bot",
    ) -> Optional[Dict]:
        """Execute spot market order (SL/TP are informational only in this client)."""
        if not self.initialized or self.client is None:
            logger.error("Binance client not initialized")
            return None

        if not self.api_key or not self.api_secret:
            logger.error("Missing Binance API credentials for live order execution")
            return None

        normalized_side = self._normalize_side(side)
        if normalized_side is None:
            logger.error("Invalid order side: %s", side)
            return None

        normalized_qty = self._normalize_quantity(quantity)
        if normalized_qty <= 0:
            logger.error("Order quantity below minimum after normalization: %s", quantity)
            return None

        try:
            current_price_info = self.get_current_price()
            current_price = float(current_price_info["last"]) if current_price_info else 0.0

            # Cap quantity by available wallet in spot mode.
            if normalized_side == "BUY" and current_price > 0:
                quote_free = self.get_quote_asset_free()
                affordable_qty = quote_free / current_price
                normalized_qty = self._normalize_quantity(min(normalized_qty, affordable_qty * 0.999))
            elif normalized_side == "SELL":
                base_free = self.get_base_asset_free()
                normalized_qty = self._normalize_quantity(min(normalized_qty, base_free))

            if normalized_qty <= 0:
                logger.error("Insufficient balance for %s order on %s", normalized_side, self.symbol)
                return None

            order_id_suffix = uuid.uuid4().hex[:12]
            order = self.client.create_order(
                symbol=self.symbol,
                side=normalized_side,
                type="MARKET",
                quantity=normalized_qty,
                newClientOrderId=f"XAKCN-{order_id_suffix}",
                recvWindow=self.recv_window,
            )

            warning_messages: List[str] = []
            if stop_loss is not None or take_profit is not None:
                warning_messages.append("SL/TP auto-order not enabled for spot client")

            logger.info(
                "[OK] Spot order executed: %s %.6f %s",
                normalized_side,
                normalized_qty,
                self.symbol
            )

            executed_qty = float(order.get("executedQty", 0.0) or 0.0)
            quote_qty = float(order.get("cummulativeQuoteQty", 0.0) or 0.0)
            avg_price = (quote_qty / executed_qty) if executed_qty > 0 else 0.0

            return {
                "order_id": order.get("orderId"),
                "deal_id": order.get("clientOrderId"),
                "volume": executed_qty if executed_qty > 0 else normalized_qty,
                "price": avg_price,
                "retcode": "OK",
                "comment": comment if not warning_messages else f"{comment} | {'; '.join(warning_messages)}",
            }
        except BinanceAPIException as error:
            logger.error("Binance API error (order): %s", error)
            return None
        except Exception as error:
            logger.error(f"Error executing order: {error}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get spot holdings for base asset as a pseudo-position."""
        if not self.initialized or self.client is None:
            return []
        if not self.api_key or not self.api_secret:
            return []

        assets = self._extract_assets()
        base_balance = self.get_asset_balance(assets["base"])
        total_base = base_balance["total"]
        if total_base <= 0:
            return []

        current_price_info = self.get_current_price()
        current_price = float(current_price_info["last"]) if current_price_info else 0.0
        return [{
            "ticket": "SPOT",
            "symbol": self.symbol,
            "type": "BUY",
            "volume": total_base,
            "open_price": 0.0,
            "current_price": current_price,
            "profit": 0.0,
            "sl": None,
            "tp": None,
            "time": datetime.utcnow(),
        }]

    def shutdown(self):
        """Shutdown hook for API parity."""
        self.initialized = False


_binance_trader: Optional[BinanceSpotTrader] = None


def get_binance_trader(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = True,
    recv_window: int = 10000
) -> BinanceSpotTrader:
    """Get or create Binance spot trader singleton instance."""
    global _binance_trader
    if _binance_trader is None:
        _binance_trader = BinanceSpotTrader(
            symbol=symbol,
            timeframe=timeframe,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            recv_window=recv_window
        )
    return _binance_trader
