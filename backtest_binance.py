"""
Backtest the same strategy as the live Binance bot, using Binance historical klines.

No API keys required (public data). Same entry/exit logic as backtest.run_backtest().

Usage:
  Swing (daily):   python backtest_binance.py --symbol BTCUSD --years 2
  Intraday (1h):   python backtest_binance.py --symbol BTCUSD --timeframe 1Hour --days 60
  Scalping (5Min): python backtest_binance.py --symbol BTCUSD --timeframe 5Min --days 14
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from backtest import (
    MIN_BARS,
    run_backtest,
    print_report,
    print_entry_indicator_stats,
)


def _symbol_to_ccxt(symbol: str) -> str:
    """BTCUSD -> BTC/USDT, ETHUSD -> ETH/USDT."""
    s = (symbol or "BTCUSD").strip().upper()
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT"
    if s.endswith("USD") and len(s) >= 6:
        base = s[:-3]
        return f"{base}/USDT"
    return f"{s}/USDT"


def _timeframe_to_binance(tf: str) -> str:
    """1Day -> 1d, 5Min -> 5m, etc."""
    s = (tf or "1Day").strip().lower()
    if s in ("1d", "1day", "day"):
        return "1d"
    if s in ("1m", "1min", "minute"):
        return "1m"
    if s in ("5m", "5min"):
        return "5m"
    if s in ("15m", "15min"):
        return "15m"
    if s in ("1h", "1hour", "hour"):
        return "1h"
    return "1d"


def _tf_interval_ms(tf: str) -> int:
    """Interval in milliseconds for pagination."""
    s = _timeframe_to_binance(tf)
    if s == "1d":
        return 86400 * 1000
    if s == "1h":
        return 3600 * 1000
    if s == "5m":
        return 5 * 60 * 1000
    if s == "15m":
        return 15 * 60 * 1000
    if s == "1m":
        return 60 * 1000
    return 86400 * 1000


def fetch_binance_ohlcv(symbol_ccxt: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV from Binance (public, no API key). Paginates if limit > 1000."""
    try:
        import ccxt
    except ImportError:
        raise ImportError("Install ccxt: pip install ccxt")

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    tf = _timeframe_to_binance(timeframe)
    interval_ms = _tf_interval_ms(timeframe)
    chunk = 1000  # Binance max per request
    all_ohlcv = []

    while len(all_ohlcv) < limit:
        fetch_limit = min(chunk, limit - len(all_ohlcv))
        since = None
        if all_ohlcv:
            # Next chunk: older than the oldest we have (ccxt returns [timestamp, o, h, l, c, v] ascending)
            since = all_ohlcv[0][0] - interval_ms * fetch_limit
        try:
            ohlcv = exchange.fetch_ohlcv(symbol_ccxt, tf, since=since, limit=fetch_limit)
        except Exception as e:
            raise RuntimeError(f"Binance fetch_ohlcv failed: {e}") from e
        if not ohlcv:
            break
        # Prepend older data so final list is chronological [oldest ... newest]
        all_ohlcv = ohlcv + all_ohlcv
        if len(ohlcv) < fetch_limit:
            break
        if len(all_ohlcv) >= limit:
            all_ohlcv = all_ohlcv[-limit:]  # keep most recent `limit` bars
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close"]].astype(float)


def main() -> None:
    from config import (
        MAX_POSITIONS,
        USE_DYNAMIC_TP_EMA_ADX,
        SCALP_TP_PCT,
        SCALP_TRAIL_ATR_MULT,
        SCALP_STOP_ATR_MULT,
        SCALP_BREAKEVEN_TRIGGER_PCT,
    )

    parser = argparse.ArgumentParser(
        description="Backtest strategy on Binance historical klines (no API key)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Symbol (default: BTCUSD; maps to BTCUSDT on Binance)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2,
        help="Years of history for daily bars (default: 2)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of history for intraday backtest (default: 7)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1Day",
        help="Bar timeframe: 1Day (swing), 1Hour, 5Min or 1Min (scalping)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000,
        help="Initial cash (default: 100000)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show why entries were skipped (no trades)",
    )
    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="Print all trades after the summary report",
    )
    parser.add_argument(
        "--max-hold-days",
        type=int,
        default=3,
        help="Max holding period in full days before forcing exit (default: 3)",
    )
    parser.add_argument(
        "--rsi-level",
        type=int,
        default=None,
        metavar="N",
        help="Override RSI entry level: only buy when RSI > N",
    )
    parser.add_argument(
        "--rsi-require-macd",
        action="store_true",
        help="Require MACD above signal for RSI entries",
    )
    args = parser.parse_args()

    symbol = (args.symbol or "BTCUSD").strip().upper()
    symbol_ccxt = _symbol_to_ccxt(symbol)
    tf = (args.timeframe or "1Day").strip()
    is_intraday = tf.lower() in ("5min", "1min", "5m", "1m", "15min", "15m", "1hour", "1h")

    if is_intraday:
        tfl = tf.lower()
        if "1h" in tfl or "hour" in tfl:
            bars_per_day = 24
        elif "5" in tfl or "5m" in tfl:
            bars_per_day = 288
        elif "1m" in tfl and "min" in tfl:
            bars_per_day = 1440
        else:
            bars_per_day = 96
        limit = min(5000, max(MIN_BARS, args.days * bars_per_day))
        print(f"Fetching last {limit} {tf} bars from Binance for {symbol_ccxt} (~{args.days} days)...")
    else:
        limit = max(500, int(args.years * 365))
        limit = min(limit, 2000)  # cap to avoid huge fetches
        print(f"Fetching last {limit} daily bars from Binance for {symbol_ccxt}...")

    bars = fetch_binance_ohlcv(symbol_ccxt, tf, limit)
    if bars.empty or len(bars) < MIN_BARS:
        print(f"Not enough data (got {len(bars)} bars, need {MIN_BARS}+). Try --years 2 or --days 14.")
        return

    indicator_overrides = {}
    if args.rsi_level is not None:
        indicator_overrides["rsi_entry_level"] = args.rsi_level
    if args.rsi_require_macd:
        indicator_overrides["rsi_require_macd"] = True

    if is_intraday:
        tp = SCALP_TP_PCT
        trail = SCALP_TRAIL_ATR_MULT
        sl = SCALP_STOP_ATR_MULT
        breakeven = SCALP_BREAKEVEN_TRIGGER_PCT
        print(
            f"Running SCALPING backtest on {len(bars)} bars "
            f"(TP={tp*100:.1f}%, trail={trail}xATR, breakeven @ {breakeven*100:.1f}%, max {MAX_POSITIONS} lots)..."
        )
        result = run_backtest(
            bars,
            initial_cash=args.cash,
            verbose=args.verbose,
            take_profit_pct=tp,
            stop_atr_mult=sl,
            trail_atr_mult=trail,
            breakeven_trigger_pct=breakeven,
            max_positions=MAX_POSITIONS,
            use_dynamic_ema_adx=USE_DYNAMIC_TP_EMA_ADX,
            max_hold_days=args.max_hold_days,
            indicator_overrides=indicator_overrides or None,
        )
    else:
        dyn = "dynamic 5%+ EMA/ADX" if USE_DYNAMIC_TP_EMA_ADX else "TP 10%, trail 3xATR"
        rsi_info = ""
        if indicator_overrides:
            parts = []
            if "rsi_entry_level" in indicator_overrides:
                parts.append(f"RSI>{indicator_overrides['rsi_entry_level']}")
            if indicator_overrides.get("rsi_require_macd"):
                parts.append("+MACD")
            if parts:
                rsi_info = f", {', '.join(parts)}"
        print(
            f"Running backtest on {len(bars)} bars "
            f"(30% capital, {dyn}, max {MAX_POSITIONS} lots{rsi_info})..."
        )
        result = run_backtest(
            bars,
            initial_cash=args.cash,
            verbose=args.verbose,
            max_positions=MAX_POSITIONS,
            use_dynamic_ema_adx=USE_DYNAMIC_TP_EMA_ADX,
            max_hold_days=args.max_hold_days,
            indicator_overrides=indicator_overrides or None,
        )

    print_report(result, symbol)
    if args.verbose:
        print_entry_indicator_stats(result)

    if args.show_trades and result.trades:
        print("TRADES (chronological):")
        print("date              side   price        pnl        reason")
        for dt, side, price, pnl, reason in result.trades:
            pnl_str = " " if pnl is None else f"{pnl:+.2f}"
            print(f"{dt}  {side:4s}  {price:8.2f}  {pnl_str:10s}  {reason}")


if __name__ == "__main__":
    main()
