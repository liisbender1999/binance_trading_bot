"""
Grid search over dynamic TP/SL parameters to maximize return while keeping win rate high
and limiting big losses. Uses same data and entry logic as backtest.

  python optimize_exit_params.py --symbol BTCUSD --years 2
  python optimize_exit_params.py --symbol BTCUSD --years 2 --min-win-rate 55
"""
import argparse
import sys
from itertools import product

from backtest import run_backtest, BacktestResult, MIN_BARS
from alpaca_client import AlpacaClient


def get_bars(symbol: str, years: float = 2, timeframe: str = "1Day"):
    """Fetch bars (daily by default). Fallback to yfinance if Alpaca has no data."""
    client = AlpacaClient()
    tf = (timeframe or "1Day").strip()
    is_intraday = tf.lower() in ("5min", "1min", "5m", "1m", "15min", "15m", "1hour", "1h")
    if is_intraday:
        tfl = tf.lower()
        bars_per_day = 24 if ("1h" in tfl or "hour" in tfl) else (288 if "5" in tfl or "5m" in tfl else 1440)
        limit = min(5000, max(MIN_BARS, int(14 * bars_per_day)))
    else:
        limit = max(500, int(years * 252))
    bars = client.get_bars_range(symbol=symbol, limit=limit, timeframe=tf)
    YF_SYMBOL_ALIASES = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"}
    if bars.empty or len(bars) < MIN_BARS:
        try:
            import yfinance as yf
            yf_symbol = YF_SYMBOL_ALIASES.get(symbol.upper(), symbol)
            if is_intraday:
                period = "14d"
                interval = "1h" if "1h" in tf.lower() or "hour" in tf.lower() else "5m"
                bars = yf.Ticker(yf_symbol).history(period=period, interval=interval)
            else:
                period = f"{max(2, int(years))}y"
                bars = yf.Ticker(yf_symbol).history(period=period)
            if bars is not None and not bars.empty:
                if "Close" in bars.columns:
                    bars = bars.rename(columns={"Close": "close"})
                if "High" in bars.columns and "Low" in bars.columns:
                    bars = bars.rename(columns={"High": "high", "Low": "low"})
                if "high" not in bars.columns:
                    bars["high"] = bars["close"]
                if "low" not in bars.columns:
                    bars["low"] = bars["close"]
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
        except Exception as e:
            print(f"yfinance failed: {e}", file=sys.stderr)
    return bars if not bars.empty and len(bars) >= MIN_BARS else None


def trade_stats(result: BacktestResult):
    """From closed trades compute avg loss per loser and worst single-trade loss %."""
    sells = [t for t in result.trades if t[1] == "sell" and t[3] is not None]
    if not sells:
        return None, None, None
    losses = [t[3] for t in sells if t[3] < 0]
    if not losses:
        return 0.0, 0.0, None  # no losing trades
    avg_loss = sum(losses) / len(losses)
    worst_loss = min(losses)
    # Approximate worst loss as % of initial capital (each trade ~30% of equity at time)
    initial = 100_000.0
    worst_loss_pct = (worst_loss / initial) * 100 if initial else 0
    return avg_loss, worst_loss, worst_loss_pct


def main():
    from config import MAX_POSITIONS, USE_DYNAMIC_TP_EMA_ADX

    parser = argparse.ArgumentParser(description="Optimize dynamic TP/step and ATR stop")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Symbol (default: BTCUSD)")
    parser.add_argument("--years", type=float, default=2, help="Years of daily data (default: 2)")
    parser.add_argument("--timeframe", type=str, default="1Day", help="1Day for swing")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash")
    parser.add_argument("--min-win-rate", type=float, default=52.0, help="Min win rate %% to consider (default: 52)")
    parser.add_argument("--max-drawdown", type=float, default=None, help="Max acceptable drawdown %% (e.g. -25)")
    args = parser.parse_args()

    if not USE_DYNAMIC_TP_EMA_ADX:
        print("Set USE_DYNAMIC_TP_EMA_ADX=true in .env for this optimizer to be relevant.")
        sys.exit(1)

    print(f"Loading {args.years}y of {args.timeframe} bars for {args.symbol}...")
    bars = get_bars(args.symbol, args.years, args.timeframe)
    if bars is None:
        print("Not enough data.")
        sys.exit(1)
    print(f"Running grid over {len(bars)} bars (max_positions={MAX_POSITIONS})...\n")

    # Grid: first target %, step %, stop ATR multiplier
    first_pct_options = [0.03, 0.05, 0.07]
    step_pct_options = [0.03, 0.05, 0.07]
    stop_atr_options = [1.5, 2.0, 2.5, 3.0]

    results = []
    total_combos = len(first_pct_options) * len(step_pct_options) * len(stop_atr_options)
    n = 0
    for first_pct, step_pct, stop_atr in product(first_pct_options, step_pct_options, stop_atr_options):
        n += 1
        if n % 10 == 0 or n == total_combos:
            print(f"  {n}/{total_combos} ...", flush=True)
        res = run_backtest(
            bars,
            initial_cash=args.cash,
            verbose=False,
            use_dynamic_ema_adx=True,
            dynamic_tp_first_pct=first_pct,
            dynamic_tp_step_pct=step_pct,
            stop_atr_mult_dynamic=stop_atr,
            max_positions=MAX_POSITIONS,
        )
        avg_loss, worst_loss, worst_loss_pct = trade_stats(res)
        results.append({
            "first_pct": first_pct,
            "step_pct": step_pct,
            "stop_atr": stop_atr,
            "return_pct": res.total_return_pct,
            "win_rate_pct": res.win_rate_pct,
            "num_trades": res.num_trades,
            "max_dd_pct": res.max_drawdown_pct,
            "avg_loss": avg_loss,
            "worst_loss_pct": worst_loss_pct,
        })

    # Sort by return (best first)
    results.sort(key=lambda x: -x["return_pct"])

    # Risk-adjusted: return per unit of drawdown (higher = better), avoid huge single losses
    for r in results:
        dd = abs(r["max_dd_pct"]) or 1
        r["return_per_dd"] = r["return_pct"] / dd
        r["worst_trade_abs"] = abs(r["worst_loss_pct"]) if r["worst_loss_pct"] else 0

    # Filter by min win rate and optional max drawdown
    min_wr = args.min_win_rate
    max_dd = args.max_drawdown
    filtered = [r for r in results if r["win_rate_pct"] >= min_wr]
    if max_dd is not None:
        filtered = [r for r in filtered if r["max_dd_pct"] >= max_dd]  # drawdown is negative

    print("\n" + "=" * 100)
    print("  TOP 20 BY TOTAL RETURN (all combos)")
    print("  first%  step%  stopATR  return%   win%   trades  maxDD%   worstTrade%  return/|DD|")
    print("=" * 100)
    for r in results[:20]:
        wr = r["win_rate_pct"]
        wt = r["worst_loss_pct"] or 0
        rpd = r["return_per_dd"]
        print(f"  {r['first_pct']*100:5.0f}   {r['step_pct']*100:5.0f}   {r['stop_atr']:5.1f}   "
              f"{r['return_pct']:+6.2f}   {wr:5.1f}   {r['num_trades']:5}   {r['max_dd_pct']:6.2f}   {wt:6.2f}   {rpd:5.2f}")

    if filtered:
        best = filtered[0]
        print("\n" + "=" * 90)
        print(f"  BEST WITH WIN RATE >= {min_wr}% (and max_drawdown >= {max_dd}% if set)")
        print("=" * 90)
        print(f"  first_pct={best['first_pct']*100:.0f}%  step_pct={best['step_pct']*100:.0f}%  "
              f"stop_atr_mult={best['stop_atr']}")
        print(f"  -> Total return: {best['return_pct']:+.2f}%  Win rate: {best['win_rate_pct']:.1f}%  "
              f"Trades: {best['num_trades']}  Max DD: {best['max_dd_pct']:.2f}%  Worst trade: {best['worst_loss_pct'] or 0:.2f}%")
        print("\n  Suggested .env / config (strategy constants):")
        print(f"    DYNAMIC_TP_FIRST_PCT={best['first_pct']}")
        print(f"    DYNAMIC_TP_STEP_PCT={best['step_pct']}")
        print(f"    STOP_ATR_MULT_DYNAMIC={best['stop_atr']}")
    else:
        print(f"\n  No combo with win_rate >= {min_wr}%. Showing best by return among all:")
        best = results[0]
        print(f"  first_pct={best['first_pct']*100:.0f}%  step_pct={best['step_pct']*100:.0f}%  "
              f"stop_atr_mult={best['stop_atr']}  return={best['return_pct']:+.2f}%  win_rate={best['win_rate_pct']:.1f}%")

    print()


if __name__ == "__main__":
    main()
