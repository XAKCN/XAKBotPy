# XAKBotPy - Binance Spot Trading Bot

[![CI](https://github.com/XAKCN/XAKBotPy/actions/workflows/ci.yml/badge.svg)](https://github.com/XAKCN/XAKBotPy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/exchange-Binance%20Spot-yellow)](https://www.binance.com/)

Unified quantitative trading bot with:
- Ensemble scoring (10 weighted indicators, output range [-1, +1])
- Optional ML model (XGBoost, 80+ features, time-series CV)
- Adaptive risk controls (Kelly Criterion, CircuitBreaker, DynamicStops)
- Binance Spot data and execution

---

## Demo Dashboard

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              XAKBotPy  |  BTCUSDT  1h  |  Cycle #12                               ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────────────────┐
│ MARKET                               │  │ SIGNAL                                                   │
├──────────────────────────────────────┤  ├──────────────────────────────────────────────────────────┤
│ Price      : $  84,312.50            │  │ Decision   : ▲ STRONG_BUY                                │
│ 24h Change :    +2.14%               │  │ Ensemble   : +0.821  [████████████████████░░░░]           │
│ Volume 24h :  41,203 BTC             │  │ ML Score   : +0.763  [██████████████████░░░░░░]           │
│ ATR (14)   :   1,842.30              │  │ Combined   : +0.800  [████████████████████░░░░]           │
│ Regime     :   TRENDING              │  │ Confidence :  HIGH                                       │
└──────────────────────────────────────┘  └──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────────────────┐
│ PORTFOLIO                            │  │ RISK                                                     │
├──────────────────────────────────────┤  ├──────────────────────────────────────────────────────────┤
│ Equity     : $ 10,843.20             │  │ Position   :  5.20% of equity                            │
│ USDT       : $  9,741.60             │  │ Entry      : $ 84,312.50                                 │
│ BTC        :   0.01312 BTC           │  │ Stop Loss  : $ 82,470.20  (-2.18%)                       │
│ Unrealized :    +13.20 USDT          │  │ Take Profit: $ 88,996.10  (+5.56%)                       │
│ Daily P&L  :    +8.43%               │  │ R:R Ratio  :   1:2.56                                   │
│ Drawdown   :    -1.2%                │  │ Kelly Frac :   0.018                                     │
└──────────────────────────────────────┘  └──────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ RECENT TRADES                                                                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ #  Time              Side  Entry         Exit          PnL        Result                            │
│ 1  2026-02-28 14:00  [B]   $81,240.00   $84,100.00   +$127.40   WIN                               │
│ 2  2026-02-28 20:00  [B]   $84,100.00   $83,420.00    -$43.10   LOSS                              │
│ 3  2026-03-01 08:00  [B]   $83,200.00   $84,312.50    +$72.80   WIN                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Backtest Results — BTCUSDT 1h (365 days)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                        BACKTEST RESULTS  |  BTCUSDT 1h  |  365 days                               ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣
│                                                                                                      │
│  Total Return      :   +31.4%       Start Capital : $10,000.00                                      │
│  Final Capital     :  $13,140.00    Total Trades  :  87                                             │
│  Sharpe Ratio      :    1.62        Win Rate      :  58.6%                                          │
│  Max Drawdown      :   -11.3%       Avg Win / Loss:  +2.1% / -0.9%                                 │
│  Profit Factor     :    1.84        Avg Trade     :  +0.36%                                         │
│  Sortino Ratio     :    2.11        Best Trade    :  +5.8%                                          │
│  Calmar Ratio      :    2.78        Worst Trade   :  -1.9%                                          │
│                                                                                                      │
│  Regime Distribution:                                                                                │
│    TRENDING   [███████████████████░░░░░░░░░░░░]  58%                                                │
│    RANGING    [█████████░░░░░░░░░░░░░░░░░░░░░░]  28%                                                │
│    HIGH_VOL   [████░░░░░░░░░░░░░░░░░░░░░░░░░░░]  14%                                                │
│                                                                                                      │
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

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

---

## Disclaimer

This software is provided for **educational and research purposes only**.

Trading cryptocurrencies involves substantial risk of loss. Past performance — including any backtest results shown here — is not indicative of future results. This bot does not constitute financial advice.

**Use at your own risk.** The author(s) accept no responsibility for any financial losses, damages, or other consequences resulting from the use of this software. Always test thoroughly on testnet before any live deployment.
