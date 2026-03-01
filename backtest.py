"""
Backtest engine: same strategy as live bot (RSI entry, TP + trailing + breakeven).

Swing (daily):   python backtest.py --symbol BTCUSD --years 2
Scalping (5Min):  python backtest.py --symbol BTCUSD --timeframe 5Min --days 7
Scalping (1Hour): python backtest.py --symbol BTCUSD --timeframe 1Hour --days 14
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from strategy import (
    compute_signal,
    should_exit_fixed_tp_sl,
    should_exit_tp_sl_trailing,
    should_exit_dynamic,
    should_exit_dynamic_ema_adx,
    TRAIL_ATR_MULT as DEFAULT_TRAIL_ATR,
    DYNAMIC_TP_FIRST_PCT,
    DYNAMIC_TP_STEP_PCT,
)

# Indicator combinations to skip as entry (normalized as sorted tags joined by '+')
BAD_ENTRY_COMBOS = {
    "adx+aroon+fib+macd+rsi",
    "adx+aroon+fib+macd+rsi+stoch",
    "adx+aroon+fib+stoch",
    "adx+aroon+bb",
    "adx+aroon+bb+rsi",
    "adx+aroon+bb+sr",
    "adx+aroon+bb+rsi+sr",
    "adx+aroon+macd+rsi",
    "adx+aroon+macd+rsi+sr",
    "adx+aroon+macd+rsi+stoch",
    "adx+aroon+rsi+sr",
    "adx+aroon+stoch",
    "adx+aroon",
    "adx+aroon+rsi",
    "adx+fib",
    "adx+fib+macd+rsi",
    "adx+fib+macd+rsi+stoch",
    "adx+rsi",
    "adx+rsi+sr",
    "adx+macd+rsi",
    "adx+macd+rsi+sr",
    "aroon+bb+rsi",
    "aroon",
    "aroon+fib+macd+rsi+stoch",
    "aroon+macd+rsi",
    "aroon+sr",
    "aroon+stoch",
    "bb",
    "bb+rsi",
    "fib",
    "fib+macd+rsi",
    "fib+stoch",
    "macd+rsi",
    "macd+rsi+stoch",
    "rsi",
    "sr",
    "stoch",
}

# Indicator combinations to treat as "good" (use 60% capital instead of 30%)
GOOD_ENTRY_COMBOS = {
    "adx+aroon+macd",
    "adx+sr",
    "aroon+bb+stoch",
    "aroon+fib+macd+rsi",
    "aroon+fib+stoch",
    "aroon+macd+rsi",
    "aroon+macd+rsi+sr",
    "aroon+macd+rsi+stoch",
    "aroon+rsi",
    "macd+rsi+sr",
}

# Match live bot (optimized for ~20-50% on 4y BTC backtest)
CAPITAL_PCT = 0.30
TAKE_PROFIT_PCT = 0.10   # 10%
STOP_ATR_MULT = 2.5
MIN_BARS = 40
MAX_POSITIONS_DEFAULT = 4


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    total_return_pct: float
    num_trades: int
    max_drawdown_pct: float
    win_rate_pct: float
    equity_curve: pd.Series
    trades: list


def run_backtest(
    bars: pd.DataFrame,
    initial_cash: float = 100_000.0,
    verbose: bool = False,
    take_profit_pct: float = None,
    stop_atr_mult: float = None,
    trail_atr_mult: float = None,
    breakeven_trigger_pct: float = None,
    use_fixed_tp_sl: bool = False,
    fixed_tp_pct: float = 0.015,
    fixed_sl_pct: float = 0.015,
    use_dynamic_tp_sl: bool = False,
    use_dynamic_ema_adx: bool = False,
    dynamic_tp_first_pct: float = None,
    dynamic_tp_step_pct: float = None,
    stop_atr_mult_dynamic: float = None,
    indicator_overrides: dict = None,
    max_positions: int = None,
    max_hold_days: int = 3,
) -> BacktestResult:
    """
    Run strategy with 30% capital per entry. Up to max_positions (default 4) open at once.
    If use_dynamic_ema_adx: dynamic TP (5%->10%->...) with EMA/ADX hold-or-take; stop = ATR then previous target.
    If use_dynamic_tp_sl: tiered TP with RSI; else if not dynamic: simple TP + trailing + breakeven.
    breakeven_trigger_pct: when None, strategy default (3%); use 0.005 for scalping.
    """
    tp = take_profit_pct if take_profit_pct is not None else TAKE_PROFIT_PCT
    sl = stop_atr_mult if stop_atr_mult is not None else STOP_ATR_MULT
    trail = trail_atr_mult if trail_atr_mult is not None else DEFAULT_TRAIL_ATR
    max_lots = max_positions if max_positions is not None else MAX_POSITIONS_DEFAULT
    use_dynamic = use_dynamic_ema_adx or use_dynamic_tp_sl
    # Dynamic EMA/ADX exit overrides (for optimization)
    dyn_first = dynamic_tp_first_pct if dynamic_tp_first_pct is not None else DYNAMIC_TP_FIRST_PCT
    dyn_step = dynamic_tp_step_pct if dynamic_tp_step_pct is not None else DYNAMIC_TP_STEP_PCT
    if bars is None or len(bars) < MIN_BARS or "close" not in bars.columns:
        return BacktestResult(
            total_return_pct=0.0,
            num_trades=0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            equity_curve=pd.Series(),
            trades=[],
        )

    # Ensure high/low for ATR
    if "high" not in bars.columns:
        bars = bars.copy()
        bars["high"] = bars["close"]
    if "low" not in bars.columns:
        bars["low"] = bars["close"]

    bars = bars.sort_index()
    close = bars["close"].astype(float)
    dates = bars.index.tolist()

    cash = initial_cash
    # Each lot: {"shares", "entry_price", "atr_at_entry", "high_water_mark", "indicator_tags", "entry_date"}
    lots: list = []
    trades: list = []
    equity_curve = []
    reason_counts = {}
    last_entry_date = None  # date of last opened lot (YYYY-MM-DD)
    kwargs_exit = dict(take_profit_pct=tp, stop_atr_mult=sl, trail_atr_mult=trail)
    if breakeven_trigger_pct is not None:
        kwargs_exit["breakeven_trigger_pct"] = breakeven_trigger_pct

    for i in range(MIN_BARS, len(bars)):
        bars_so_far = bars.iloc[: i + 1]
        price = close.iloc[i]

        # Update high water mark for each lot and check exit
        to_remove = []
        for idx, lot in enumerate(lots):
            lot["high_water_mark"] = max(lot["high_water_mark"], price)
            # Max holding period: force exit after `max_hold_days` full days from entry_date
            should_exit = False
            reason = ""
            entry_date_str = lot.get("entry_date")
            if entry_date_str:
                try:
                    entry_date = datetime.fromisoformat(entry_date_str).date()
                    age_days = (dates[i].date() - entry_date).days
                except Exception:
                    age_days = 0
                if age_days >= max_hold_days:
                    should_exit = True
                    reason = "max_hold"

            if not should_exit:
                if use_fixed_tp_sl:
                    should_exit, reason = should_exit_fixed_tp_sl(
                        price, lot["entry_price"],
                        tp_pct=fixed_tp_pct,
                        sl_pct=fixed_sl_pct,
                    )
                elif use_dynamic_ema_adx:
                    tp_first = lot.get("tp_first_pct", DYNAMIC_TP_FIRST_PCT)
                    tp_step = lot.get("tp_step_pct", DYNAMIC_TP_STEP_PCT)
                    target_pct = lot.get("current_target_pct", tp_first)
                    raised = lot.get("raised_stop")
                    should_exit, reason, new_target, new_raised = should_exit_dynamic_ema_adx(
                        price, lot["entry_price"], lot["atr_at_entry"], lot["high_water_mark"],
                        target_pct, raised, bars_so_far,
                        tp_first_pct=tp_first,
                        tp_step_pct=tp_step,
                    )
                    lot["current_target_pct"] = new_target
                    lot["raised_stop"] = new_raised
                elif use_dynamic_tp_sl:
                    target_pct = lot.get("current_target_pct", 0.05)
                    raised = lot.get("raised_stop")
                    should_exit, reason, new_target, new_raised = should_exit_dynamic(
                        price, lot["entry_price"], lot["atr_at_entry"], lot["high_water_mark"],
                        target_pct, raised, bars_so_far, stop_atr_mult=sl, trail_atr_mult=trail,
                    )
                    lot["current_target_pct"] = new_target
                    lot["raised_stop"] = new_raised
                else:
                    should_exit, reason = should_exit_tp_sl_trailing(
                        price, lot["entry_price"], lot["atr_at_entry"], lot["high_water_mark"],
                        **kwargs_exit,
                    )
            if should_exit:
                to_remove.append(idx)
                cash += price * lot["shares"]
                pnl = (price - lot["entry_price"]) * lot["shares"]
                trades.append((dates[i], "sell", price, pnl, reason))

        for idx in reversed(to_remove):
            lots.pop(idx)

        # Entry: when signal is buy and we have room for another lot
        if len(lots) < max_lots:
            overrides = indicator_overrides or {}
            result = compute_signal(bars_so_far, **overrides)
            if verbose and result.signal != "buy":
                r = result.reason or "hold"
                reason_counts[r] = reason_counts.get(r, 0) + 1
            if result.signal == "buy" and result.atr_at_entry is not None:
                # --- Entry guards: per-indicator and per-day ---
                entry_reason = result.reason or ""
                entry_tags = {t for t in entry_reason.split("+") if t}
                # 0) Skip known-bad indicator combinations
                if entry_tags:
                    combo_key = "+".join(sorted(entry_tags))
                    if combo_key in BAD_ENTRY_COMBOS:
                        if verbose:
                            reason_counts["skip_bad_combo"] = reason_counts.get("skip_bad_combo", 0) + 1
                        continue
                # 1) Prevent same-indicator stacking: if any current lot was opened by the same indicator tag(s), skip
                if entry_tags:
                    active_tags = set()
                    for lot in lots:
                        active_tags.update(lot.get("indicator_tags", []))
                    if entry_tags & active_tags:
                        if verbose:
                            reason_counts["skip_same_indicator"] = reason_counts.get("skip_same_indicator", 0) + 1
                        # Skip opening another lot from the same indicator
                        continue
                # 2) Only one new lot per day: if we already opened a lot today, wait until next day
                entry_date = dates[i].date()
                if last_entry_date is not None and entry_date.isoformat() == last_entry_date:
                    if verbose:
                        reason_counts["skip_same_day"] = reason_counts.get("skip_same_day", 0) + 1
                    continue

                # Position sizing: 60% capital for good combos, else default 30%
                combo_key = "+".join(sorted(entry_tags)) if entry_tags else ""
                capital_pct = 0.70 if combo_key in GOOD_ENTRY_COMBOS else CAPITAL_PCT
                value_to_use = cash * capital_pct
                if value_to_use > 0 and price > 0:
                    shares = value_to_use / price
                    if shares >= 1e-9:
                        cash -= shares * price
                        new_lot = {
                            "shares": shares,
                            "entry_price": price,
                            "atr_at_entry": result.atr_at_entry,
                            "high_water_mark": price,
                            "indicator_tags": list(entry_tags),
                            "entry_date": entry_date.isoformat(),
                        }
                        if use_dynamic_ema_adx or use_dynamic_tp_sl:
                            # Per-combo dynamic TP: good combos 5%/5%, others 4%/3%
                            if combo_key in GOOD_ENTRY_COMBOS:
                                tp_first = DYNAMIC_TP_FIRST_PCT
                                tp_step = DYNAMIC_TP_STEP_PCT
                            else:
                                tp_first = 0.04
                                tp_step = 0.03
                            new_lot["tp_first_pct"] = tp_first
                            new_lot["tp_step_pct"] = tp_step
                            new_lot["current_target_pct"] = tp_first if use_dynamic_ema_adx else DYNAMIC_TP_FIRST_PCT
                            new_lot["raised_stop"] = None
                        lots.append(new_lot)
                        # Store which indicators fired on entry (e.g. "rsi+bb") for later analysis
                        trades.append((dates[i], "buy", price, None, entry_reason or "entry"))
                        last_entry_date = entry_date.isoformat()

        equity = cash + sum(lot["shares"] * price for lot in lots)
        equity_curve.append((dates[i], equity))

    equity_series = pd.Series(
        {d: e for d, e in equity_curve},
        dtype=float,
    ).sort_index()

    if len(equity_series) < 2:
        total_return_pct = 0.0
        max_drawdown_pct = 0.0
    else:
        total_return_pct = (equity_series.iloc[-1] - initial_cash) / initial_cash * 100
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak.replace(0, 1) * 100
        max_drawdown_pct = drawdown.min()

    sell_trades = [t for t in trades if t[1] == "sell" and t[3] is not None]
    num_trades = len(sell_trades)
    wins = sum(1 for t in sell_trades if t[3] and t[3] > 0)
    win_rate_pct = (wins / num_trades * 100) if num_trades else 0.0

    if verbose and reason_counts:
        print("\n  Why no entry (counts):")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")

    return BacktestResult(
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct,
        equity_curve=equity_series,
        trades=trades,
    )


def print_report(result: BacktestResult, symbol: str) -> None:
    """Print a simple backtest report."""
    print("\n" + "=" * 50)
    print(f"  BACKTEST REPORT — {symbol}")
    print("=" * 50)
    print(f"  Total return:     {result.total_return_pct:+.2f}%")
    print(f"  Max drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"  Number of trades: {result.num_trades}")
    print(f"  Win rate:         {result.win_rate_pct:.1f}%")
    if result.num_trades == 0:
        print("  (Run with --verbose to see why no entries triggered)")
    print("=" * 50 + "\n")


def print_entry_indicator_stats(result: BacktestResult) -> None:
    """Aggregate performance by entry indicator combination (from buy-trade reasons)."""
    # We pair each sell with the most recent preceding buy's reason.
    combo_stats = {}
    last_buy_reason = None
    for dt, side, price, pnl, reason in result.trades:
        if side == "buy":
            last_buy_reason = reason or "entry"
        elif side == "sell" and pnl is not None and last_buy_reason is not None:
            key = "+".join(sorted(t for t in (last_buy_reason or "").split("+") if t)) or "entry"
            stats = combo_stats.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
            stats["trades"] += 1
            stats["pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1

    if not combo_stats:
        return

    print("Indicator combo performance (by entry reason):")
    print("  combo                 trades  win%    total_pnl")
    for combo, s in sorted(combo_stats.items(), key=lambda kv: -kv[1]["pnl"]):
        win_pct = (s["wins"] / s["trades"] * 100.0) if s["trades"] else 0.0
        print(f"  {combo:20s}  {s['trades']:5d}  {win_pct:5.1f}%  {s['pnl']:10.2f}")
    print()


def main() -> None:
    from config import (
        SCALP_TP_PCT,
        SCALP_TRAIL_ATR_MULT,
        SCALP_STOP_ATR_MULT,
        SCALP_BREAKEVEN_TRIGGER_PCT,
    )

    parser = argparse.ArgumentParser(description="Backtest strategy on historical data")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol (default: from .env TRADE_SYMBOL)")
    parser.add_argument("--years", type=float, default=2, help="Years of history for daily bars (default: 2)")
    parser.add_argument("--days", type=int, default=7, help="Days of history for intraday (5Min/1Min) backtest (default: 7)")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Bar timeframe: 1Day (swing), 1Hour, 5Min or 1Min (scalping)")
    parser.add_argument("--cash", type=float, default=100_000, help="Initial cash (default: 100000)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show why entries were skipped (no trades)")
    parser.add_argument(
        "--show-trades",
        action="store_true",
        help="Print all trades (date, side, price, PnL, reason) after the summary report",
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
        help="Override RSI entry level: only buy when RSI > N (e.g. 35 or 40 for tighter momentum)",
    )
    parser.add_argument(
        "--rsi-require-macd",
        action="store_true",
        help="Require MACD above signal for RSI entries (momentum confirmation)",
    )
    args = parser.parse_args()

    # Alpaca backtest entrypoint; import client lazily so Binance-only workflows
    # (e.g. backtest_binance.py) can import this module without requiring Alpaca.
    from alpaca_client import AlpacaClient
    symbol = args.symbol or client.symbol
    tf = (args.timeframe or "1Day").strip()
    is_intraday = tf.lower() in ("5min", "1min", "5m", "1m", "15min", "15m", "1hour", "1h")

    if is_intraday:
        # 1Hour: 24, 5Min: ~288, 1Min: ~1440, 15Min: 96 bars/day (24h)
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
        print(f"Fetching last {limit} {tf} bars for {symbol} (~{args.days} days)...")
    else:
        limit = max(500, int(args.years * 252))
        print(f"Fetching last {limit} daily bars for {symbol}...")

    bars = client.get_bars_range(symbol=symbol, limit=limit, timeframe=tf)
    YF_SYMBOL_ALIASES = {"BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"}

    if bars.empty or len(bars) < MIN_BARS:
        print("Alpaca returned no data. Trying yfinance...")
        try:
            import yfinance as yf
            yf_symbol = YF_SYMBOL_ALIASES.get(symbol.upper(), symbol)
            if is_intraday:
                tfl = tf.lower()
                if "1h" in tfl or "hour" in tfl:
                    interval = "1h"
                elif "5" in tfl or "5m" in tfl:
                    interval = "5m"
                else:
                    interval = "1m"
                period = f"{min(60, max(1, args.days))}d"
                bars = yf.Ticker(yf_symbol).history(period=period, interval=interval)
                if bars.empty and yf_symbol != symbol:
                    bars = yf.Ticker(symbol).history(period=period, interval=interval)
            else:
                period = f"{max(2, int(args.years))}y"
                bars = yf.Ticker(yf_symbol).history(period=period)
                if bars.empty and yf_symbol != symbol:
                    bars = yf.Ticker(symbol).history(period=period)
            if bars is not None and not bars.empty:
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
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
        except Exception as e:
            print(f"yfinance failed: {e}")
        if bars.empty or len(bars) < MIN_BARS:
            print(f"Not enough data (need {MIN_BARS}+ bars). For intraday try --days 7 or --timeframe 1Day for years.")
            return

    from config import MAX_POSITIONS, USE_DYNAMIC_TP_EMA_ADX
    indicator_overrides = {}
    if args.rsi_level is not None:
        indicator_overrides["rsi_entry_level"] = args.rsi_level
    if args.rsi_require_macd:
        indicator_overrides["rsi_require_macd"] = True
    if is_intraday:
        tp, trail, sl = SCALP_TP_PCT, SCALP_TRAIL_ATR_MULT, SCALP_STOP_ATR_MULT
        breakeven = SCALP_BREAKEVEN_TRIGGER_PCT
        print(f"Running SCALPING backtest on {len(bars)} bars (TP={tp*100:.1f}%%, trail={trail}xATR, breakeven @ {breakeven*100:.1f}%%, max {MAX_POSITIONS} lots)...")
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
        dyn = "dynamic 5%%+ EMA/ADX" if USE_DYNAMIC_TP_EMA_ADX else "TP 10%%, trail 3xATR"
        rsi_info = ""
        if indicator_overrides:
            parts = []
            if "rsi_entry_level" in indicator_overrides:
                parts.append(f"RSI>{indicator_overrides['rsi_entry_level']}")
            if indicator_overrides.get("rsi_require_macd"):
                parts.append("+MACD")
            if parts:
                rsi_info = f", {', '.join(parts)}"
        print(f"Running backtest on {len(bars)} bars (30%% capital, {dyn}, max {MAX_POSITIONS} lots{rsi_info})...")
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

    # Optionally dump all trades to the console for inspection
    if args.show_trades and result.trades:
        print("TRADES (chronological):")
        print("date              side   price        pnl        reason")
        for dt, side, price, pnl, reason in result.trades:
            pnl_str = " " if pnl is None else f"{pnl:+.2f}"
            print(f"{dt}  {side:4s}  {price:8.2f}  {pnl_str:10s}  {reason}")


if __name__ == "__main__":
    main()
