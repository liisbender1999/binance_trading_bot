"""
Parameter sweep: run backtest with different TP/SL/trail and find configs with 20-50% total return.

Usage: python optimize.py [--symbol BTCUSD] [--years 2]
"""
import argparse
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from backtest import run_backtest, MIN_BARS
from alpaca_client import AlpacaClient

YF_SYMBOL_ALIASES = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"}


def load_bars(symbol: str, years: float = 2) -> pd.DataFrame:
    limit = max(500, int(years * 252))
    try:
        client = AlpacaClient()
        bars = client.get_bars_range(symbol=symbol, limit=limit, timeframe="1Day")
    except Exception:
        bars = pd.DataFrame()
    if bars.empty or len(bars) < MIN_BARS:
        try:
            import yfinance as yf
            yf_symbol = YF_SYMBOL_ALIASES.get(symbol.upper(), symbol)
            period = f"{max(2, int(years))}y"
            ticker = yf.Ticker(yf_symbol)
            bars = ticker.history(period=period)
            if bars.empty and yf_symbol != symbol:
                bars = yf.Ticker(symbol).history(period=period)
            if bars is not None and not bars.empty:
                if "Close" in bars.columns:
                    bars = bars.rename(columns={"Close": "close"})
                if "close" not in bars.columns:
                    bars = bars.rename(columns={bars.columns[3]: "close"})
                if "High" in bars.columns and "Low" in bars.columns:
                    bars = bars.rename(columns={"High": "high", "Low": "low"})
                if "high" not in bars.columns:
                    bars["high"] = bars["close"]
                if "low" not in bars.columns:
                    bars["low"] = bars["close"]
                if "Volume" in bars.columns and "volume" not in bars.columns:
                    bars = bars.rename(columns={"Volume": "volume"})
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
        except Exception as e:
            print(f"yfinance failed: {e}")
    return bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--years", type=float, default=2)
    parser.add_argument("--cash", type=float, default=100_000)
    args = parser.parse_args()

    print(f"Loading {args.years}y of data for {args.symbol}...")
    bars = load_bars(args.symbol, args.years)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (need {MIN_BARS}+ bars).")
        return

    print(f"Running parameter sweep on {len(bars)} bars...\n")

    # Grid: aim for 20-50% return (wider TP and looser stop for crypto)
    tp_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    sl_values = [2.5, 3.0, 3.5, 4.0]
    trail_values = [2.0, 2.5, 3.0]

    results = []
    total_combos = len(tp_values) * len(sl_values) * len(trail_values)
    n = 0
    for tp in tp_values:
        for sl in sl_values:
            for trail in trail_values:
                n += 1
                if n % 20 == 0 or n == total_combos:
                    print(f"  {n}/{total_combos}...", end="\r")
                r = run_backtest(
                    bars,
                    initial_cash=args.cash,
                    verbose=False,
                    take_profit_pct=tp,
                    stop_atr_mult=sl,
                    trail_atr_mult=trail,
                )
                results.append({
                    "tp_pct": tp * 100,
                    "sl_atr": sl,
                    "trail_atr": trail,
                    "return_pct": r.total_return_pct,
                    "trades": r.num_trades,
                    "max_dd_pct": r.max_drawdown_pct,
                    "win_rate_pct": r.win_rate_pct,
                })

    print()
    # Sort by return descending
    results.sort(key=lambda x: -x["return_pct"])

    # Target 20-50%
    in_range = [x for x in results if 20 <= x["return_pct"] <= 50]
    if in_range:
        print("=== Configs with 20-50% total return ===\n")
        for i, row in enumerate(in_range[:15], 1):
            print(f"  {i}. TP={row['tp_pct']:.0f}%  SL={row['sl_atr']}xATR  trail={row['trail_atr']}xATR  "
                  f"return={row['return_pct']:+.1f}%  trades={row['trades']}  max_dd={row['max_dd_pct']:.1f}%  win={row['win_rate_pct']:.0f}%")
        best = in_range[0]
    else:
        print("=== No config reached 20-50% on this data. Top 10 by return ===\n")
        for i, row in enumerate(results[:10], 1):
            print(f"  {i}. TP={row['tp_pct']:.0f}%  SL={row['sl_atr']}xATR  trail={row['trail_atr']}xATR  "
                  f"return={row['return_pct']:+.1f}%  trades={row['trades']}  max_dd={row['max_dd_pct']:.1f}%  win={row['win_rate_pct']:.0f}%")
        best = results[0]

    print("\n--- Recommended (best in or near 20-50%) ---")
    print(f"  TAKE_PROFIT_PCT = {best['tp_pct']/100:.2f}   # {best['tp_pct']:.0f}%")
    print(f"  STOP_ATR_MULT   = {best['sl_atr']}")
    print(f"  TRAIL_ATR_MULT  = {best['trail_atr']}   # in strategy.py")
    print(f"  Backtest return: {best['return_pct']:+.1f}%  ({best['trades']} trades, win rate {best['win_rate_pct']:.0f}%)")
    print("\nApply these in bot.py (TP, SL) and strategy.py (TRAIL_ATR_MULT).")


if __name__ == "__main__":
    main()
