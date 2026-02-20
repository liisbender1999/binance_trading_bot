"""
High-frequency preset: target ~1 trade every 1–2 days, win rate >= 65%, return >= 40%/year.
Uses 1-hour bars for more signals, tight TP/trailing/breakeven to lock wins and cut losses.

Usage: python optimize_high_freq.py [--symbol BTCUSD] [--days 365]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from backtest import run_backtest, MIN_BARS

YF_SYMBOL_ALIASES = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"}


def load_1h_bars(symbol: str, days: int = 365) -> pd.DataFrame:
    """Load 1-hour OHLCV bars (yfinance; max ~730d for 1h)."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance required: pip install yfinance")
        return pd.DataFrame()
    yf_symbol = YF_SYMBOL_ALIASES.get(symbol.upper(), symbol)
    period = f"{min(730, max(60, days))}d"
    bars = yf.Ticker(yf_symbol).history(period=period, interval="1h")
    if bars.empty and yf_symbol != symbol:
        bars = yf.Ticker(symbol).history(period=period, interval="1h")
    if bars is None or bars.empty:
        return pd.DataFrame()
    if "Close" in bars.columns:
        bars = bars.rename(columns={"Close": "close"})
    if "close" not in bars.columns and len(bars.columns) >= 4:
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
    return bars


def main():
    parser = argparse.ArgumentParser(description="Optimize for high frequency, 65%+ win rate, 40%+ return")
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--days", type=int, default=365, help="Days of 1h data (max 730)")
    parser.add_argument("--cash", type=float, default=100_000)
    parser.add_argument("--min-trades-per-year", type=float, default=125, help="Target ~1 trade every 3 days")
    parser.add_argument("--target-wr", type=float, default=65.0)
    parser.add_argument("--target-return", type=float, default=40.0)
    args = parser.parse_args()

    print(f"Loading {args.days} days of 1h data for {args.symbol}...")
    bars = load_1h_bars(args.symbol, args.days)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (need {MIN_BARS}+ bars).")
        return

    days_actual = (bars.index.max() - bars.index.min()).days if len(bars) > 1 else 0
    years_actual = days_actual / 365.0 if days_actual else 1.0
    print(f"Got {len(bars)} bars (~{days_actual} days). Sweeping TP / trail / breakeven / entry...\n")

    tp_values = [0.02, 0.03, 0.04, 0.05]
    trail_values = [1.2, 1.5]
    breakeven_values = [0.005, 0.01]
    min_ind_values = [1, 2]

    results = []
    total = len(tp_values) * len(trail_values) * len(breakeven_values) * len(min_ind_values)
    n = 0
    for tp in tp_values:
        for trail in trail_values:
            for be in breakeven_values:
                for min_ind in min_ind_values:
                    n += 1
                    if n % 20 == 0 or n == total:
                        print(f"  {n}/{total}...", end="\r")
                    overrides = {"entry_min_indicators": min_ind}
                    r = run_backtest(
                        bars,
                        initial_cash=args.cash,
                        verbose=False,
                        take_profit_pct=tp,
                        stop_atr_mult=1.2,
                        trail_atr_mult=trail,
                        breakeven_trigger_pct=be,
                        indicator_overrides=overrides,
                    )
                    trades_per_year = (r.num_trades / years_actual) if years_actual else 0
                    results.append({
                        "tp_pct": tp * 100,
                        "trail_atr": trail,
                        "breakeven_pct": be * 100,
                        "entry_min_indicators": min_ind,
                        "return_pct": r.total_return_pct,
                        "num_trades": r.num_trades,
                        "trades_per_year": trades_per_year,
                        "win_rate_pct": r.win_rate_pct,
                        "max_dd_pct": r.max_drawdown_pct,
                    })

    print()
    # Require minimum trade frequency
    min_trades_in_period = args.min_trades_per_year * years_actual
    candidates = [x for x in results if x["num_trades"] >= min_trades_in_period]
    # Sort by: meet WR and return targets first, then by return
    def score(x):
        wr_ok = 1 if x["win_rate_pct"] >= args.target_wr else 0
        ret_ok = 1 if x["return_pct"] >= args.target_return else 0
        return (wr_ok, ret_ok, x["return_pct"], x["win_rate_pct"])

    candidates.sort(key=score, reverse=True)
    if not candidates:
        candidates = sorted(results, key=lambda x: (x["win_rate_pct"], x["return_pct"]), reverse=True)

    print("=" * 72)
    print("  HIGH-FREQUENCY PRESET (target: ~1 trade/1–2 days, WR>=65%%, return>=40%/year)")
    print("=" * 72)
    print(f"  Symbol: {args.symbol}  |  Bars: {len(bars)}  |  Period: ~{years_actual:.2f} years")
    print()
    for i, row in enumerate(candidates[:20], 1):
        meets = []
        if row["trades_per_year"] >= args.min_trades_per_year:
            meets.append("freq")
        if row["win_rate_pct"] >= args.target_wr:
            meets.append("WR")
        if row["return_pct"] >= args.target_return:
            meets.append("ret")
        tag = "  [{}]".format(",".join(meets)) if meets else ""
        print(
            f"  {i:2}. TP={row['tp_pct']:.0f}%  trail={row['trail_atr']}x  BE={row['breakeven_pct']:.1f}%  min_ind={row['entry_min_indicators']}  "
            f"return={row['return_pct']:+.1f}%  trades={row['num_trades']} ({row['trades_per_year']:.0f}/yr)  "
            f"WR={row['win_rate_pct']:.0f}%  max_dd={row['max_dd_pct']:.1f}%{tag}"
        )

    best = candidates[0]
    print()
    print("=" * 72)
    print("  RECOMMENDED HIGH-FREQ SETTINGS (best trade-off)")
    print("=" * 72)
    print(f"  TAKE_PROFIT_PCT     = {best['tp_pct']/100:.2f}   # {best['tp_pct']:.0f}%")
    print(f"  TRAIL_ATR_MULT      = {best['trail_atr']}")
    print(f"  BREAKEVEN_TRIGGER   = {best['breakeven_pct']/100:.2f}   # {best['breakeven_pct']:.1f}%")
    print(f"  ENTRY_MIN_INDICATORS = {best['entry_min_indicators']}")
    print(f"  Backtest: return={best['return_pct']:+.1f}%  trades={best['num_trades']} ({best['trades_per_year']:.0f}/yr)  "
          f"win_rate={best['win_rate_pct']:.1f}%  max_dd={best['max_dd_pct']:.1f}%")
    print("=" * 72)

    # Save preset for backtest
    preset = {
        "take_profit_pct": best["tp_pct"] / 100,
        "trail_atr_mult": best["trail_atr"],
        "breakeven_trigger_pct": best["breakeven_pct"] / 100,
        "entry_min_indicators": best["entry_min_indicators"],
        "stop_atr_mult": 1.2,
    }
    preset_path = Path(__file__).resolve().parent / "high_freq_preset.json"
    import json
    with open(preset_path, "w") as f:
        json.dump(preset, f, indent=2)
    print(f"\n  Preset saved to {preset_path.name} (use with backtest_high_freq.py)")


if __name__ == "__main__":
    main()
