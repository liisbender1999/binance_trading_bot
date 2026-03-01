# Trading Bot (Binance)

A trading bot for **Binance USD-M Futures** (crypto). It runs a multi-indicator strategy (RSI, MACD, BB, etc.) and places market orders with configurable leverage. Use demo/testnet for paper trading.

## Setup

1. **Create a virtual environment (recommended)**

   ```powershell
   cd C:\Users\Kasutaja\alpaca-trading-bot
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment**

   - Copy `.env.example` to `.env`.
   - Set `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` (use [Futures Testnet](https://testnet.binancefuture.com) for paper trading).
   - Keep `BINANCE_TESTNET=true` for paper; set to `false` with live keys for real trading.

## Run the bot

```powershell
python main.py
```

- The bot checks the strategy every `CHECK_INTERVAL_SECONDS` (default 60). Crypto trades 24/7.
- Stop with **Ctrl+C**.

## Backtest strategies (no real orders)

Test the same strategy on historical data before going live.

**Binance data** (no API key needed; uses public klines):

```powershell
python backtest_binance.py --symbol BTCUSD --years 2
python backtest_binance.py --symbol BTCUSD --timeframe 1Hour --days 60
python backtest_binance.py --symbol BTCUSD --timeframe 5Min --days 14 --show-trades
```

- Reports total return %, max drawdown, number of trades, win rate.
- **Strategy**: Edit `strategy.py`; the live bot and backtest use `compute_signal()`.

## Project layout

| File | Purpose |
|------|--------|
| `main.py` | Entry point; runs the bot loop. |
| `bot.py` | Main loop: fetch bars → signal → place/cancel orders. |
| `backtest.py` | Backtest engine: run strategy on history, print report. |
| `strategy.py` | SMA crossover: buy when fast SMA crosses above slow, sell when it crosses below. |
| `binance_client.py` | Wrapper for Binance USD-M Futures (account, positions, bars, orders). |
| `config.py` | Loads settings from `.env`. |

## Customization

- **Symbol**: Set `TRADE_SYMBOL` in `.env` (e.g. `AAPL`, `SPY`, `QQQ`).
- **Position size**: Set `POSITION_SIZE` (number of shares per trade).
- **Strategy**: Edit `strategy.py` to change the logic (e.g. different SMA periods or another indicator).
- **Check interval**: Set `CHECK_INTERVAL_SECONDS` (e.g. 300 for 5 minutes).

## Safety

- Start with **paper trading** and verify behavior before switching to live.
- Never commit `.env` or share your API keys.
- This bot is for education; use at your own risk.
