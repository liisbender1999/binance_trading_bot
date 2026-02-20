"""
Backtest the high-frequency preset (1h bars, tight TP/trail/breakeven).
Target: ~1 trade every 1–2 days, win rate >= 65%, return >= 40%/year.

Usage: python backtest_high_freq.py [--symbol BTCUSD] [--days 365]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import run_backtest, MIN_BARS, print_report
from optimize_high_freq import load_1h_bars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cash", type=float, default=100_000)
    args = parser.parse_args()

    preset_path = Path(__file__).resolve().parent / "high_freq_preset.json"
    if preset_path.exists():
        with open(preset_path) as f:
            preset = json.load(f)
        tp = preset["take_profit_pct"]
        trail = preset["trail_atr_mult"]
        breakeven = preset["breakeven_trigger_pct"]
        stop = preset["stop_atr_mult"]
        overrides = {"entry_min_indicators": preset["entry_min_indicators"]}
        print(f"Using preset: TP={tp*100:.1f}%  trail={trail}x  breakeven={breakeven*100:.1f}%  min_ind={preset['entry_min_indicators']}")
    else:
        tp, trail, breakeven, stop = 0.03, 1.5, 0.01, 1.2
        overrides = {"entry_min_indicators": 1}
        print(f"No preset found; using defaults: TP={tp*100:.0f}%  trail={trail}x  breakeven={breakeven*100:.1f}%")

    print(f"Loading {args.days} days of 1h data for {args.symbol}...")
    bars = load_1h_bars(args.symbol, args.days)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (need {MIN_BARS}+ bars).")
        return

    years = (bars.index.max() - bars.index.min()).days / 365.0 if len(bars) > 1 else 1.0
    print(f"Running high-freq backtest on {len(bars)} bars (~{years:.2f} years)...\n")
    result = run_backtest(
        bars,
        initial_cash=args.cash,
        take_profit_pct=tp,
        stop_atr_mult=stop,
        trail_atr_mult=trail,
        breakeven_trigger_pct=breakeven,
        indicator_overrides=overrides,
    )
    print_report(result, args.symbol)
    trades_per_year = result.num_trades / years if years else 0
    print(f"  Trades per year:  {trades_per_year:.1f}")
    print()


if __name__ == "__main__":
    main()
