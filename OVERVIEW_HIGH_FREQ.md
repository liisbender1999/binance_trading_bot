# High-Frequency Bot: Overview and Results

## Design goals

- **Trade frequency:** About 1 trade every 1–2 days (~125–250 trades/year).
- **Win rate:** Target ≥ 65%.
- **Return:** Target ≥ 40% per year.
- **Backtest:** Run on historical data and report results.

---

## What was implemented

### 1. Data and timeframe

- **1-hour bars** instead of daily, so the strategy is evaluated many times per day and can enter/exit more often.
- **Data source:** `optimize_high_freq.py` and `backtest_high_freq.py` use **yfinance** to load 1h OHLCV (e.g. BTC-USD) for a chosen number of days (e.g. 90–365).

### 2. Entry logic (unchanged)

- Same multi-indicator system: **RSI, MACD, Bollinger Bands, Volume, Support/Resistance**.
- **ENTRY_MIN_INDICATORS = 1** in the high-freq preset so **any one** indicator can trigger a long (more trades).
- All five indicators remain available; only the “min indicators” setting is relaxed for frequency.

### 3. Exit logic (tuned for high frequency)

- **Tight take-profit (TP):** 2–3% (e.g. 3% in the saved preset) to lock gains quickly.
- **Tight trailing stop:** 1.2–1.3× ATR so winners are cut less aggressively than with a wide trail, but risk is still limited.
- **Breakeven:** Once price is 0.5–1% above entry, stop is moved to breakeven so open risk is reduced.
- **Stop loss:** 1.2× ATR (tighter than the swing preset) to keep losses small per trade.

These choices aim to:
- Increase the number of round-trips (more trades per year).
- Favor “many small wins / small losses” to support a higher win rate and stable equity curve, rather than rare large moves.

### 4. Optimization

- **`optimize_high_freq.py`** sweeps:
  - Take-profit: 2%, 3%, 4%, 5%, 6%
  - Trailing ATR multiplier: 1.2, 1.5
  - Breakeven trigger: 0.5%, 1%
  - Entry: 1 or 2 indicators
- It scores configs by trade frequency (e.g. ≥125 trades/year), then by return and win rate.
- Best config is written to **`high_freq_preset.json`** and used by **`backtest_high_freq.py`**.

### 5. Backtest

- **`backtest_high_freq.py`**:
  - Loads 1h bars (same way as the optimizer).
  - Reads `high_freq_preset.json` (or uses built-in defaults).
  - Runs the same strategy as the main backtest (same entry/exit formulas, same 30% capital per trade).
  - Reports total return, max drawdown, number of trades, **trades per year**, and win rate.

### 6. Running the high-freq bot live

To run the **live** bot in a high-frequency style (1h bars, tight TP/trail/breakeven):

- Set in `.env` (or equivalent):
  - `BOT_MODE=scalping`
  - `BAR_TIMEFRAME=1Hour`
  - `SCALP_TP_PCT=0.03`        (3% TP, match preset)
  - `SCALP_TRAIL_ATR_MULT=1.3`
  - `SCALP_BREAKEVEN_TRIGGER_PCT=0.01`
  - `ENTRY_MIN_INDICATORS=1`   (or 2 for fewer, stricter entries)
- Then run the usual bot (e.g. `python main.py`). It will use 1h bars and the scalping exit parameters above.

The “bot” that trades every 1–2 days in practice is this same codebase, with the above config and 1h data.

---

## Backtest results (what we actually get)

Backtests were run on **BTCUSD 1h** over **~90–120 days** of recent data.

| Metric           | Target   | Typical result (recent BTC 1h) |
|-----------------|----------|---------------------------------|
| Trades per year | 125–250 | **600–750** (frequency goal met or exceeded) |
| Win rate        | ≥ 65%   | **~32–35%** (below target)      |
| Total return    | ≥ 40%/y | **Negative** (about -2% to -10% over the period) |
| Max drawdown   | —        | About **-10% to -13%**          |

So:

- **Trade frequency:** The setup achieves **more** than 1 trade every 1–2 days (often 2+ trades per day in the backtest).
- **Win rate and return:** On this data, **65% win rate and 40% annual return were not achieved**. Win rate stays in the low 30s and return is negative over the tested window.

### Why 65% / 40% is hard to get in backtest

- **65% win rate** with **40% return** and **high frequency** would require a very strong, consistent edge on that symbol and period. Many strategies that achieve high win rates do so with fewer trades (e.g. swing on daily) or in specific regimes.
- On **recent BTC 1h** data, the same entry logic and tight TP/trail/breakeven produce many trades but **more losing than winning** trades, so win rate stays ~33% and return is negative.
- Hitting 65% and 40% in a backtest would likely require:
  - A very favorable slice of history (high overfitting risk), or
  - A different market/period, or
  - A different strategy (e.g. different indicators, filters, or risk rules).

The high-freq bot and backtest are built to **pursue** your goals (trade every 1–2 days, 65% WR, 40% return) and to **report honestly** what the same logic does on real data.

---

## Files added/used

| File | Purpose |
|------|--------|
| `optimize_high_freq.py` | Load 1h bars, sweep TP/trail/breakeven/entry, pick best preset, save `high_freq_preset.json`. |
| `backtest_high_freq.py` | Load 1h bars, run backtest with preset (or defaults), report return, WR, trades/year, drawdown. |
| `high_freq_preset.json` | Saved best parameters (TP, trail, breakeven, entry_min_indicators, stop). |
| `OVERVIEW_HIGH_FREQ.md` | This overview: goals, what was done, and backtest results. |

---

## Commands

```bash
# Optimize (sweep parameters, save preset)
python optimize_high_freq.py --symbol BTCUSD --days 120

# Backtest with saved preset
python backtest_high_freq.py --symbol BTCUSD --days 120

# Optional: relax frequency so fewer but “better” trades are considered
python optimize_high_freq.py --symbol BTCUSD --days 365 --min-trades-per-year 80
```

---

## Summary

- **Bot that trades every 1–2 days:** Implemented by using **1-hour bars**, **single-indicator entry** (or 2), and **tight TP/trail/breakeven**; backtest shows **600–750+ trades/year** (goal met).
- **Backtest:** Implemented in `backtest_high_freq.py`; run with the preset from `optimize_high_freq.py`.
- **65% win rate and 40% return:** Set as **targets** and optimized toward, but **not reached** on recent BTC 1h data; real backtest results are ~33% WR and negative return. The design and tuning steps above are what was done to get as close as the current strategy and data allow, and to make the results transparent.
