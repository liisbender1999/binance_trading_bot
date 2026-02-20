# Alpaca Trading Bot

A real trading bot that uses the Alpaca API (paper or live). It runs a **simple SMA crossover strategy** (10/20 day) on a configurable symbol and places market orders when the market is open.

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
   - Add your Alpaca API key and secret from [Alpaca Dashboard](https://app.alpaca.markets/).
   - Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets` for **paper trading** (no real money).
   - For **live trading**, change to `ALPACA_BASE_URL=https://api.alpaca.markets` only when you’re ready.

   Example `.env`:

   ```
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   TRADE_SYMBOL=SPY
   POSITION_SIZE=1
   CHECK_INTERVAL_SECONDS=60
   ```

## Run the bot

```powershell
python main.py
```

- The bot checks the strategy every `CHECK_INTERVAL_SECONDS` (default 60).
- It only trades when the market is open.
- Stop with **Ctrl+C**.

## Backtest strategies (no real orders)

Test the same strategy on historical data before going live.

**Alpaca data** (uses your Alpaca credentials):

```powershell
python backtest.py
python backtest.py --symbol BTCUSD --years 2 --cash 50000
```

**Binance data** (no API key needed; uses public klines — good before running the bot on Binance):

```powershell
python backtest_binance.py --symbol BTCUSD --years 2
python backtest_binance.py --symbol BTCUSD --timeframe 1Hour --days 60
python backtest_binance.py --symbol BTCUSD --timeframe 5Min --days 14 --show-trades
```

- Reports total return %, max drawdown, number of trades, win rate.
- **Strategy**: Edit `strategy.py`; the live bot and both backtests use `compute_signal()`, so any change is tested the same way.

## Project layout

| File | Purpose |
|------|--------|
| `main.py` | Entry point; runs the bot loop. |
| `bot.py` | Main loop: fetch bars → signal → place/cancel orders. |
| `backtest.py` | Backtest engine: run strategy on history, print report. |
| `strategy.py` | SMA crossover: buy when fast SMA crosses above slow, sell when it crosses below. |
| `alpaca_client.py` | Wrapper for Alpaca API (account, positions, bars, orders). |
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
