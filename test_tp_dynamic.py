"""
Test take-profit percentages (simple exit) and dynamic TP/SL over 2 years.
Reports best config for each mode and overall.

Usage: python test_tp_dynamic.py [--symbol BTCUSD]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import run_backtest, MIN_BARS
from optimize import load_bars


def main():
    parser = argparse.ArgumentParser(description="Test TP % and dynamic TP/SL, 2-year backtest")
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to backtest")
    parser.add_argument("--cash", type=float, default=100_000)
    args = parser.parse_args()

    years = 2.0
    print(f"Loading {years} years of data for {args.symbol}...")
    bars = load_bars(args.symbol, years)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (need {MIN_BARS}+ bars).")
        return

    print(f"Running tests on {len(bars)} bars (2-year period).\n")

    # --- 1) Simple exit: sweep take-profit percentages (fixed SL 2.5, trail 3.0) ---
    tp_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    simple_results = []
    for tp in tp_values:
        r = run_backtest(
            bars,
            initial_cash=args.cash,
            verbose=False,
            take_profit_pct=tp,
            stop_atr_mult=2.5,
            trail_atr_mult=3.0,
            use_dynamic_tp_sl=False,
        )
        simple_results.append({
            "mode": "simple",
            "tp_pct": tp * 100,
            "sl_atr": 2.5,
            "trail_atr": 3.0,
            "return_pct": r.total_return_pct,
            "trades": r.num_trades,
            "max_dd_pct": r.max_drawdown_pct,
            "win_rate_pct": r.win_rate_pct,
        })

    # --- 2) Dynamic TP/SL: tiered 5%->10%->15% with RSI; vary SL and trail ---
    dynamic_configs = [
        (2.5, 3.0),
        (2.5, 2.5),
        (3.0, 3.0),
        (3.0, 2.5),
        (2.0, 2.5),
    ]
    dynamic_results = []
    for sl, trail in dynamic_configs:
        r = run_backtest(
            bars,
            initial_cash=args.cash,
            verbose=False,
            stop_atr_mult=sl,
            trail_atr_mult=trail,
            use_dynamic_tp_sl=True,
        )
        dynamic_results.append({
            "mode": "dynamic",
            "tp_pct": None,
            "sl_atr": sl,
            "trail_atr": trail,
            "return_pct": r.total_return_pct,
            "trades": r.num_trades,
            "max_dd_pct": r.max_drawdown_pct,
            "win_rate_pct": r.win_rate_pct,
        })

    # --- Report: simple TP sweep ---
    print("=" * 60)
    print("  SIMPLE EXIT - Take profit % sweep (2 years, SL=2.5xATR, trail=3xATR)")
    print("=" * 60)
    simple_results.sort(key=lambda x: -x["return_pct"])
    for i, row in enumerate(simple_results, 1):
        print(f"  TP={row['tp_pct']:.0f}%   return={row['return_pct']:+.1f}%   trades={row['trades']}   "
              f"max_dd={row['max_dd_pct']:.1f}%   win={row['win_rate_pct']:.0f}%")
    best_simple = simple_results[0]
    print(f"\n  Best simple: TP={best_simple['tp_pct']:.0f}%  ->  return={best_simple['return_pct']:+.1f}%  "
          f"({best_simple['trades']} trades, max_dd={best_simple['max_dd_pct']:.1f}%)\n")

    # --- Report: dynamic TP/SL ---
    print("=" * 60)
    print("  DYNAMIC EXIT - Tiered TP (5%->10%->15%...) + RSI; SL/trail sweep (2 years)")
    print("=" * 60)
    dynamic_results.sort(key=lambda x: -x["return_pct"])
    for i, row in enumerate(dynamic_results, 1):
        print(f"  SL={row['sl_atr']}x trail={row['trail_atr']}x   return={row['return_pct']:+.1f}%   "
              f"trades={row['trades']}   max_dd={row['max_dd_pct']:.1f}%   win={row['win_rate_pct']:.0f}%")
    best_dynamic = dynamic_results[0]
    print(f"\n  Best dynamic: SL={best_dynamic['sl_atr']}x trail={best_dynamic['trail_atr']}x  ->  "
          f"return={best_dynamic['return_pct']:+.1f}%  ({best_dynamic['trades']} trades, "
          f"max_dd={best_dynamic['max_dd_pct']:.1f}%)\n")

    # --- Overall best (2-year result) ---
    all_results = simple_results + dynamic_results
    all_results.sort(key=lambda x: -x["return_pct"])
    best = all_results[0]
    print("=" * 60)
    print("  BEST OVERALL (2-year backtest)")
    print("=" * 60)
    if best["mode"] == "simple":
        print(f"  Mode:    Simple TP + trailing stop")
        print(f"  TP:      {best['tp_pct']:.0f}%")
        print(f"  SL:      {best['sl_atr']}x ATR   trail: {best['trail_atr']}x ATR")
    else:
        print(f"  Mode:    Dynamic TP/SL (tiered 5%->10%->15% with RSI)")
        print(f"  SL:      {best['sl_atr']}x ATR   trail: {best['trail_atr']}x ATR")
    print(f"  Return:  {best['return_pct']:+.1f}%")
    print(f"  Trades:  {best['trades']}   Win rate: {best['win_rate_pct']:.0f}%   Max drawdown: {best['max_dd_pct']:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
