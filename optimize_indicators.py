"""
Backtest with different indicator parameters (RSI level, BB period, BB std, min_indicators) and report best results.
Uses ENTRY_MODE=or (RSI OR MACD OR BB). Run with daily bars and longer history.

Usage:
  python optimize_indicators.py [--symbol BTCUSD] [--years 2]
  python optimize_indicators.py --sort win_rate --top 10   # rank by win rate
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import run_backtest, MIN_BARS
from optimize import load_bars


def main():
    parser = argparse.ArgumentParser(description="Optimize indicator parameters (RSI, BB, min_indicators) via backtest")
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to backtest")
    parser.add_argument("--years", type=float, default=2, help="Years of daily data")
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--top", type=int, default=15, help="Number of top configs to print")
    parser.add_argument("--sort", choices=["return", "win_rate"], default="return",
                        help="Sort by total_return (default) or win_rate")
    args = parser.parse_args()

    print(f"Loading {args.years}y of daily data for {args.symbol}...")
    bars = load_bars(args.symbol, args.years)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (need {MIN_BARS}+ bars).")
        return

    print(f"Running indicator grid on {len(bars)} bars (ENTRY_MODE=or: RSI / MACD / BB / Volume / S/R)...\n")

    rsi_levels = [25, 30, 35, 40, 45, 50]
    bb_periods = [15, 20, 25]
    bb_stds = [1.5, 2.0, 2.5]
    min_indicators_list = [1, 2]
    volume_ma_periods = [15, 20, 25]
    sr_lookbacks = [15, 20, 25]
    sr_tolerance_pcts = [0.01, 0.02]

    results = []
    total = (
        len(rsi_levels) * len(bb_periods) * len(bb_stds) * len(min_indicators_list)
        * len(volume_ma_periods) * len(sr_lookbacks) * len(sr_tolerance_pcts)
    )
    n = 0
    for rsi_level in rsi_levels:
        for bb_period in bb_periods:
            for bb_std in bb_stds:
                for min_indicators in min_indicators_list:
                    for volume_ma_period in volume_ma_periods:
                        for sr_lookback in sr_lookbacks:
                            for sr_tolerance_pct in sr_tolerance_pcts:
                                n += 1
                                if n % 100 == 0 or n == total:
                                    print(f"  {n}/{total}...", end="\r")
                                overrides = {
                                    "rsi_entry_level": rsi_level,
                                    "bb_period": bb_period,
                                    "bb_std": bb_std,
                                    "entry_min_indicators": min_indicators,
                                    "volume_ma_period": volume_ma_period,
                                    "sr_lookback": sr_lookback,
                                    "sr_tolerance_pct": sr_tolerance_pct,
                                }
                                r = run_backtest(
                                    bars,
                                    initial_cash=args.cash,
                                    verbose=False,
                                    indicator_overrides=overrides,
                                )
                                results.append({
                                    "rsi_entry_level": rsi_level,
                                    "bb_period": bb_period,
                                    "bb_std": bb_std,
                                    "entry_min_indicators": min_indicators,
                                    "volume_ma_period": volume_ma_period,
                                    "sr_lookback": sr_lookback,
                                    "sr_tolerance_pct": sr_tolerance_pct,
                                    "return_pct": r.total_return_pct,
                                    "trades": r.num_trades,
                                    "max_dd_pct": r.max_drawdown_pct,
                                    "win_rate_pct": r.win_rate_pct,
                                })

    print()
    if args.sort == "win_rate":
        # Prefer configs with enough trades; then by win rate
        results.sort(key=lambda x: (-x["win_rate_pct"], -x["trades"]))
        sort_label = "WIN RATE"
    else:
        results.sort(key=lambda x: -x["return_pct"])
        sort_label = "RETURN"

    print("=" * 70)
    print(f"  TOP {args.top} INDICATOR CONFIGS BY {sort_label} ({args.symbol}, {args.years}y)")
    print("=" * 70)
    for i, row in enumerate(results[: args.top], 1):
        print(
            f"  {i:2}. RSI>{row['rsi_entry_level']}  BB={row['bb_period']}/{row['bb_std']}  vol={row['volume_ma_period']}  SR={row['sr_lookback']}/{row['sr_tolerance_pct']}  min={row['entry_min_indicators']}  "
            f"return={row['return_pct']:+.1f}%  trades={row['trades']}  max_dd={row['max_dd_pct']:.1f}%  win={row['win_rate_pct']:.0f}%"
        )

    best = results[0]
    print("\n" + "=" * 70)
    print("  BEST INDICATOR SETTINGS (copy to strategy.py or .env)")
    print("=" * 70)
    print(f"  RSI_ENTRY_LEVEL      = {best['rsi_entry_level']}   # buy when RSI > this")
    print(f"  BB_PERIOD            = {best['bb_period']}   # Bollinger Band period")
    print(f"  BB_STD               = {best['bb_std']}   # Bollinger Band std devs")
    print(f"  VOLUME_MA_PERIOD     = {best['volume_ma_period']}   # volume above this MA = volume signal")
    print(f"  SR_LOOKBACK          = {best['sr_lookback']}   # bars for support/resistance level")
    print(f"  SR_TOLERANCE_PCT     = {best['sr_tolerance_pct']}   # within this fraction of S/R level")
    print(f"  ENTRY_MIN_INDICATORS = {best['entry_min_indicators']}   # require this many of RSI/MACD/BB/Volume/SR (2 = higher win rate)")
    print(f"\n  Backtest result: return={best['return_pct']:+.1f}%  trades={best['trades']}  "
          f"max_drawdown={best['max_dd_pct']:.1f}%  win_rate={best['win_rate_pct']:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
