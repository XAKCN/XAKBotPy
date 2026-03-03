# XAKBotPy - Agent Guidelines

## Project Context

Production-ready quantitative trading bot for Binance Spot. Unified architecture with ML, ensemble scoring, and adaptive risk management.

## Architecture

### Core Components

| Component | Purpose |
|-----------|---------|
| `main.py` | Single entry point with 5 modes |
| `backtest/` | VectorBT engine + Optuna optimization |
| `ml/` | XGBoost pipeline with 80 features |
| `filters/` | Kelly Criterion + regime detection |
| `utils/` | Enhanced indicators + visualization |
| `exchange/` | Binance Spot client |

### Operation Modes

```bash
python main.py --mode {demo|trade|train|backtest|optimize}
```

## Binance Spot Requirements

- Binance account with Spot enabled
- API key and secret configured for live mode
- Use testnet first (`BINANCE_TESTNET=true`)
- Symbol must exist in spot market (example: `BTCUSDT`)

## Coding Standards

### Language
- **English** for all code, comments, and documentation
- No special characters in source files

### Formatting
- ASCII box drawing: `+`, `-`, `|`, `=`
- Fixed-width columns
- No emojis (use `[B]`, `[S]`, `[H]`)
- Currency: `$` (USDT for spot account display)

### Type Hints
```python
def calculate_position_size(
    equity: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
```

## Key Algorithms

### Tiered Decision
```python
def tiered_decision(ensemble_score, ml_score):
    combined = 0.6 * ensemble + 0.4 * ml  # If ML enabled

    if combined > 0.75:   return 'STRONG_BUY',  1.0
    elif combined > 0.55: return 'BUY',         0.75
    elif combined > 0.30: return 'WEAK_BUY',    0.50
    elif combined < -0.75: return 'STRONG_SELL', 1.0
    elif combined < -0.55: return 'SELL',        0.75
    elif combined < -0.30: return 'WEAK_SELL',   0.50
    else: return 'HOLD', 0.0
```

### Kelly Criterion
```python
def kelly_fraction(win_rate, avg_win, avg_loss):
    R = abs(avg_win / avg_loss)
    kelly = win_rate - ((1 - win_rate) / R)
    return max(0, min(kelly * 0.5, 0.02))  # Cap at 2%
```

## Risk Limits (Hard Constraints)

Never modify without explicit approval:
- Max risk per trade: 2%
- Max position size: 20%
- Max daily loss: 5%
- Max drawdown: 15%
- Max consecutive losses: 3

## Testing Requirements

Before production:
1. Backtest on 1+ year of data
2. Verify Sharpe > 1.0
3. Check max drawdown < 20%
4. Test in TEST_MODE for 100+ cycles
5. Validate visual output alignment

## Common Issues

### Binance API / Spot
- Ensure API key has Spot permission
- For live mode, confirm IP whitelist (if enabled)
- Start with testnet to validate behavior
- Check symbol name (`BTCUSDT`, `ETHUSDT`)

### Data
- Minimum 500 candles for warm-up
- Validate timeframe matches strategy
- Testnet may have shorter history than live endpoint

### Performance
- Use vectorized operations (pandas/numpy)
- Avoid loops in indicator calculations
- Cache feature calculations when possible

## Dependencies

- `python-binance` - Primary spot data/trading interface
- `xgboost` - ML predictions
- `optuna` - Optimization
- `vectorbt` - Backtesting
- `ta-lib` - Optional (fallback to pandas)

## Symbol Names

Common Binance Spot symbols:
- `BTCUSDT` - Bitcoin
- `ETHUSDT` - Ethereum
- `BNBUSDT` - BNB
- `SOLUSDT` - Solana
- `XRPUSDT` - XRP

Symbol availability varies by Binance listing and account region.
