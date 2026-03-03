# XAKBotPy - Binance Spot Trading Bot

Unified quantitative trading bot with:
- Ensemble scoring
- Optional ML model (XGBoost)
- Adaptive risk controls
- Binance Spot data and execution

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Configure

Copy `.env.example` to `.env` and set:
- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`
- `BINANCE_TESTNET=true` (recommended first)
- `OPERATION_CODE=BTCUSDT`
- `CANDLE_PERIOD=1h`

## 3. Main Commands (`main.py`)

```bash
# Show CLI help
python main.py --help

# Demo mode (visual dashboard, infinite by default)
python main.py --mode demo --symbol BTCUSDT

# Demo mode with fixed cycles (recommended for tests)
python main.py --mode demo --symbol BTCUSDT --cycles 1 --interval 2
# Demo wallet starts with fictitious 10000 USDT (change with --capital)
python main.py --mode demo --symbol BTCUSDT --cycles 1 --capital 10000
# Demo with live market-data endpoint for faster cycle updates
python main.py --mode demo --symbol BTCUSDT --cycles 5 --interval 2 --binance-live-endpoint

# Live spot trading
python main.py --mode trade --symbol BTCUSDT --live

# Train ML model
python main.py --mode train --symbol BTCUSDT --days 180

# Backtest
python main.py --mode backtest --symbol BTCUSDT --days 365

# Backtest with trend filter (EMA20/EMA50 + ADX>22)
python main.py --mode backtest --symbol BTCUSDT --days 90 --trend_filter
# Custom trend filter params
python main.py --mode backtest --symbol BTCUSDT --days 90 --trend_filter --trend-ema-fast 20 --trend-ema-slow 50 --trend-adx 22
# Fixed R:R 1:2 with ATR (SL=1x ATR, TP=2x ATR, ignore exits)
python main.py --mode backtest --symbol BTCUSDT --days 90 --trend_filter --atr-sl-mult 1 --atr-tp-mult 2
# Volume Spike filter (volume > rolling(20) * 1.2)
python main.py --mode backtest --symbol BTCUSDT --days 90 --trend_filter --volume_spike_filter

# Optimize strategy
python main.py --mode optimize --symbol BTCUSDT --days 365 --trials 100
```

## 4. Spot Trading Behavior

- `BUY` opens/increases base asset.
- `SELL` only closes/reduces existing base asset balance.
- No synthetic short positions in spot mode.
- SL/TP parameters are calculated and logged; automatic SL/TP order placement is not enabled in the spot client.
- In `demo` and test execution, bot uses virtual wallet balances (default 10000 USDT).

## 5. Safe Rollout

1. Start in `TEST_MODE=true` and `BINANCE_TESTNET=true`.
2. Run `--mode demo --cycles 1` and verify logs/dashboard.
3. Validate symbol/timeframe and signal behavior.
4. Switch to `--live` only after validation.

Notes:
- In `TEST_MODE`, if testnet candles repeat for many cycles, the bot auto-switches to Binance live public endpoint for data refresh.
- To force live data from startup, use `--binance-live-endpoint`.
