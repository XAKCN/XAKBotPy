# Binance Spot Setup Guide

## 1. Create API Key

1. Open Binance account settings.
2. Create a new API key.
3. Enable Spot and Reading permissions.
4. Restrict IP addresses if possible.

## 2. Testnet First

Use testnet before any live funds:

```env
BINANCE_TESTNET=true
TEST_MODE=true
```

## 3. Configure Bot

```env
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
OPERATION_CODE=BTCUSDT
CANDLE_PERIOD=1h
```

## 4. Run

```bash
python main.py --mode demo --symbol BTCUSDT --cycles 1
python main.py --mode trade --symbol BTCUSDT --live
```

## 5. Validate

1. Check BTCUSDT 1h candles load correctly.
2. Confirm dashboard shows ensemble score.
3. Confirm no Binance API errors in logs.
4. In live mode, verify:
   - `get_positions()` reflects base asset holdings
   - quote asset free balance is available (e.g., USDT)

## 6. Go Live Safely

1. Keep `BINANCE_TESTNET=true` while validating behavior.
2. Switch to `BINANCE_TESTNET=false` only after validation.
3. Use small size and monitor logs.
4. Keep risk constraints unchanged.

