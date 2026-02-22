"""Main bot loop: multi-lot (max 4), 30% capital per entry, TP + trailing + breakeven."""
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

from datetime import datetime

from config import BROKER
if BROKER == "binance":
    from binance_client import BinanceClient as _Client
else:
    from alpaca_client import AlpacaClient as _Client

# No-ops when db module is missing (e.g. db.py not in deploy); always defined to avoid NameError
def _db_noop(*args, **kwargs):
    pass

def _get_total_profit_default():
    return 0.0

try:
    from db import init_db, insert_trade, get_total_profit
    _db_available = True
except ImportError:
    _db_available = False
    init_db = insert_trade = _db_noop
    get_total_profit = _get_total_profit_default

from config import (
    BOT_MODE,
    BAR_TIMEFRAME,
    BAR_LIMIT,
    MIN_BARS,
    CHECK_INTERVAL_SECONDS,
    SCALP_TP_PCT,
    SCALP_TRAIL_ATR_MULT,
    SCALP_STOP_ATR_MULT,
    SCALP_BREAKEVEN_TRIGGER_PCT,
    SCALP_CHECK_INTERVAL,
    MAX_POSITIONS,
    TRADE_CRYPTO,
    USE_DYNAMIC_TP_EMA_ADX,
)
from strategy import (
    compute_signal,
    should_exit_tp_sl_trailing,
    should_exit_dynamic_ema_adx,
    atr as atr_fn,
    TRAIL_ATR_MULT as DEFAULT_TRAIL_ATR,
    DYNAMIC_TP_FIRST_PCT,
    DYNAMIC_TP_STEP_PCT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Indicator combinations to skip as entry (normalized as sorted tags joined by '+')
BAD_ENTRY_COMBOS = {
    "adx+aroon+fib+stoch",
    "adx+aroon+bb",
    "adx+aroon+bb+rsi",
    "adx+aroon+bb+sr",
    "adx+aroon+bb+rsi+sr",
    "adx+aroon+rsi+sr",
    "adx+aroon+stoch",
    "adx+aroon",
    "adx+aroon+rsi",
    "adx+fib",
    "adx+fib+macd+rsi",
    "adx+rsi",
    "adx+rsi+sr",
    "adx+macd+rsi",
    "adx+macd+rsi+sr",
    "aroon+bb+rsi",
    "aroon",
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

# State file: list of lots (entry_price, atr_at_entry, high_water_mark, shares), symbol; max MAX_POSITIONS lots
BOT_STATE_FILE = Path(__file__).resolve().parent / ".bot_state.json"
CAPITAL_PCT = 0.30
# Crypto-friendly: wider TP and stop so volatile moves don’t get cut early
TAKE_PROFIT_PCT = 0.5   # 10%
STOP_ATR_MULT = 2.5


def _tp_trail_stop():
    """Take profit %, trail ATR mult, stop ATR mult for current mode."""
    if BOT_MODE == "scalping":
        return SCALP_TP_PCT, SCALP_TRAIL_ATR_MULT, SCALP_STOP_ATR_MULT
    return TAKE_PROFIT_PCT, DEFAULT_TRAIL_ATR, STOP_ATR_MULT


def _check_interval():
    """Seconds to sleep between cycles."""
    return SCALP_CHECK_INTERVAL if BOT_MODE == "scalping" else CHECK_INTERVAL_SECONDS


def load_state() -> dict:
    """Load persisted state: {"lots": [...], "symbol": str, "last_entry_date": "YYYY-MM-DD"}."""
    if not BOT_STATE_FILE.exists():
        return {}
    try:
        with open(BOT_STATE_FILE) as f:
            data = json.load(f)
        if "lots" in data:
            return data
        # Backward compat: old single-lot format
        if "entry_price" in data and "atr_at_entry" in data:
            return {
                "lots": [{
                    "entry_price": data["entry_price"],
                    "atr_at_entry": data["atr_at_entry"],
                    "high_water_mark": data.get("high_water_mark", data["entry_price"]),
                    "shares": data.get("shares", 1),
                }],
                "symbol": data.get("symbol", ""),
            }
        return data
    except Exception:
        return {}


def save_state_lots(lots: list, symbol: str, last_entry_date: Optional[str] = None) -> None:
    """Save list of lots and metadata (each lot: entry_price, atr_at_entry, high_water_mark, shares, ...)."""
    data = {"lots": lots, "symbol": symbol}
    if last_entry_date is not None:
        data["last_entry_date"] = last_entry_date
    with open(BOT_STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def clear_state() -> None:
    if BOT_STATE_FILE.exists():
        BOT_STATE_FILE.unlink(missing_ok=True)


def _is_crypto_symbol(symbol: str) -> bool:
    """True if symbol is crypto (24/7 market). Alpaca: BTCUSD; Binance: BTCUSDT."""
    if not symbol:
        return False
    s = symbol.upper().strip()
    if s in ("BTCUSD", "ETHUSD", "LTCUSD", "DOGEUSD", "XRPUSD", "SOLUSD", "ADAUSD", "AVAXUSD", "LINKUSD", "MATICUSD", "DOTUSD", "UNIUSD", "BCHUSD", "LTCUSD"):
        return True
    if s.endswith("USD") and len(s) >= 6:
        return True
    if s.endswith("USDT") and len(s) >= 6:
        return True
    return False


def run_once(client: _Client) -> None:
    """Fetch bars, check entry/exit, place orders if needed."""
    # Crypto trades 24/7: skip stock market-hours check for crypto symbols or when TRADE_CRYPTO=true
    if not TRADE_CRYPTO and not _is_crypto_symbol(client.symbol):
        clock = client.get_clock()
        if not clock.is_open:
            logger.info("Market closed, skipping cycle.")
            return

    bars = client.get_bars(timeframe=BAR_TIMEFRAME, limit=BAR_LIMIT)
    n_bars = 0 if bars.empty else len(bars)
    if bars.empty or n_bars < MIN_BARS:
        logger.warning("Not enough bars for strategy: got %d (need %d+).", n_bars, MIN_BARS)
        return

    position = client.get_position(client.symbol)
    has_position = position is not None and int(float(position.qty)) != 0
    current_price = float(position.avg_entry_price) if has_position else None
    if current_price is None:
        try:
            trade = client.get_latest_trade(client.symbol)
            current_price = float(trade.price) if trade else None
        except Exception:
            current_price = None
    if current_price is None and not bars.empty:
        current_price = float(bars["close"].iloc[-1])

    state = load_state()
    lots = state.get("lots") or []
    symbol = state.get("symbol") or client.symbol
    last_entry_date = state.get("last_entry_date")

    # Reconcile: if we have position but lots empty or single lot with unknown shares, sync from broker
    if has_position and position is not None:
        total_qty = float(position.qty)
        if not lots:
            # Restarted with open position: treat whole position as one lot (use current ATR from bars)
            close = bars["close"].astype(float)
            high = bars["high"].astype(float) if "high" in bars.columns else close
            low = bars["low"].astype(float) if "low" in bars.columns else close
            atr_val = atr_fn(high, low, close, 14).iloc[-1]
            atr_entry = 0.0 if (atr_val is None or (isinstance(atr_val, float) and math.isnan(atr_val))) else float(atr_val)
            new_lot = {
                "entry_price": float(position.avg_entry_price),
                "atr_at_entry": atr_entry,
                "high_water_mark": current_price or float(position.avg_entry_price),
                "shares": total_qty,
            }
            if USE_DYNAMIC_TP_EMA_ADX:
                new_lot["current_target_pct"] = DYNAMIC_TP_FIRST_PCT
                new_lot["raised_stop"] = None
            lots = [new_lot]
            save_state_lots(lots, client.symbol, last_entry_date)
        elif len(lots) == 1 and lots[0].get("shares", 1) == 1 and total_qty != 1:
            lots[0]["shares"] = total_qty
            save_state_lots(lots, client.symbol, last_entry_date)

    # --- Exit: check each lot; close first that triggers TP/trail/SL or dynamic EMA/ADX ---
    tp_pct, trail_atr, stop_atr = _tp_trail_stop()
    breakeven_trigger = SCALP_BREAKEVEN_TRIGGER_PCT if BOT_MODE == "scalping" else None
    kwargs_exit = dict(take_profit_pct=tp_pct, stop_atr_mult=stop_atr, trail_atr_mult=trail_atr)
    if breakeven_trigger is not None:
        kwargs_exit["breakeven_trigger_pct"] = breakeven_trigger

    if lots and current_price is not None:
        current_date = bars.index[-1].date()
        for idx, lot in enumerate(lots):
            hwm = max(float(lot["high_water_mark"]), current_price)
            lot["high_water_mark"] = hwm
            # Max holding period: force exit after 3 full days from entry_date
            should_exit = False
            reason = ""
            entry_date_str = lot.get("entry_date")
            if entry_date_str:
                try:
                    entry_date = datetime.fromisoformat(entry_date_str).date()
                    age_days = (current_date - entry_date).days
                except Exception:
                    age_days = 0
                if age_days >= 3:
                    should_exit = True
                    reason = "max_hold"

            if not should_exit:
                if USE_DYNAMIC_TP_EMA_ADX:
                    tp_first = lot.get("tp_first_pct", DYNAMIC_TP_FIRST_PCT)
                    tp_step = lot.get("tp_step_pct", DYNAMIC_TP_STEP_PCT)
                    target_pct = lot.get("current_target_pct", tp_first)
                    raised = lot.get("raised_stop")
                    should_exit, reason, new_target, new_raised = should_exit_dynamic_ema_adx(
                        current_price,
                        float(lot["entry_price"]),
                        float(lot["atr_at_entry"]),
                        hwm,
                        target_pct,
                        raised,
                        bars,
                        tp_first_pct=tp_first,
                        tp_step_pct=tp_step,
                    )
                    lot["current_target_pct"] = new_target
                    lot["raised_stop"] = new_raised
                else:
                    should_exit, reason = should_exit_tp_sl_trailing(
                        current_price,
                        float(lot["entry_price"]),
                        float(lot["atr_at_entry"]),
                        hwm,
                        **kwargs_exit,
                    )
            if should_exit:
                sell_qty = lot["shares"]
                if sell_qty >= 1e-9:
                    # Alpaca stocks need int qty; crypto can be fractional
                    qty_to_sell = min(qty_to_sell, available_btc)
                    client.submit_market_order(side="sell", qty=qty_to_sell)
                    if _db_available:
                        realized_pnl = (current_price - float(lot["entry_price"])) * sell_qty
                        insert_trade(
                            datetime.utcnow().isoformat(),
                            client.symbol,
                            "sell",
                            float(sell_qty),
                            current_price,
                            realized_pnl=realized_pnl,
                        )
                        logger.info(
                            "Exiting lot: %s (price=%.2f, qty=%s) | total profit (DB): %.2f",
                            reason, current_price, qty_to_sell, get_total_profit(),
                        )
                    else:
                        logger.info("Exiting lot: %s (price=%.2f, qty=%s)", reason, current_price, qty_to_sell)
                lots.pop(idx)
                save_state_lots(lots, client.symbol, last_entry_date)
                if not lots:
                    clear_state()
                return
        save_state_lots(lots, client.symbol, last_entry_date)

    # --- Entry: when 2 indicators met and we have room for another lot (max 4) ---
    if len(lots) < MAX_POSITIONS:
        result = compute_signal(bars)
        logger.info("Signal: %s (%s) | lots=%d/%d", result.signal, result.reason, len(lots), MAX_POSITIONS)
        if result.signal == "buy" and result.atr_at_entry is not None:
            # Prevent bad / low-performing indicator combos and same-indicator stacking
            entry_reason = result.reason or ""
            entry_tags = {t for t in entry_reason.split("+") if t}
            if entry_tags:
                # 0) Skip known-bad combinations
                combo_key = "+".join(sorted(entry_tags))
                if combo_key in BAD_ENTRY_COMBOS:
                    logger.info("Skip entry: bad combo %s", combo_key)
                    return
                # Prevent same-indicator stacking: if an open lot was created from any of these indicator tags, skip
                active_tags = set()
                for lot in lots:
                    active_tags.update(lot.get("indicator_tags", []))
                if entry_tags & active_tags:
                    logger.info("Skip entry: indicators already active (%s)", ",".join(entry_tags & active_tags))
                    return
            # Only one new lot per day: if we already opened a lot today, wait until next day
            entry_date = bars.index[-1].date()
            if last_entry_date is not None and entry_date.isoformat() == last_entry_date:
                logger.info("Skip entry: already opened lot on %s", last_entry_date)
                return

            # Position sizing: 60% capital for good combos, else default 30%
            combo_key = "+".join(sorted(entry_tags)) if entry_tags else ""
            capital_pct = 0.70 if combo_key in GOOD_ENTRY_COMBOS else CAPITAL_PCT

            qty = client.shares_from_pct_buying_power(pct=capital_pct, price=current_price)
            # Crypto: allow fractional (e.g. 0.001 BTC); stocks: need at least 1 share
            min_qty = 0.0001 if _is_crypto_symbol(client.symbol) else 1
            if qty < min_qty:
                logger.warning("%.0f%% of buying power gives %.6s shares at price %.2f; skip.", capital_pct * 100, qty, current_price or 0)
            else:
                entry_price_est = current_price or float(bars["close"].iloc[-1])
                client.submit_market_order(side="buy", qty=qty)
                if _db_available:
                    insert_trade(
                        datetime.utcnow().isoformat(),
                        client.symbol,
                        "buy",
                        float(qty),
                        entry_price_est,
                    )
                new_lot = {
                    "entry_price": entry_price_est,
                    "atr_at_entry": result.atr_at_entry,
                    "high_water_mark": entry_price_est,
                    "shares": float(qty),
                    "indicator_tags": list(entry_tags),
                    "entry_date": entry_date.isoformat(),
                }
                if USE_DYNAMIC_TP_EMA_ADX:
                    # Per-combo dynamic TP: good combos 5%/5%, others 4%/3%
                    if combo_key in GOOD_ENTRY_COMBOS:
                        tp_first = DYNAMIC_TP_FIRST_PCT
                        tp_step = DYNAMIC_TP_STEP_PCT
                    else:
                        tp_first = 0.04
                        tp_step = 0.03
                    new_lot["tp_first_pct"] = tp_first
                    new_lot["tp_step_pct"] = tp_step
                    new_lot["current_target_pct"] = tp_first
                    new_lot["raised_stop"] = None
                lots.append(new_lot)
                last_entry_date = entry_date.isoformat()
                save_state_lots(lots, client.symbol, last_entry_date)
                logger.info("Long entry: qty=%s, entry~%.2f, ATR=%.4f, lots=%d/%d",
                            qty, entry_price_est, result.atr_at_entry, len(lots), MAX_POSITIONS)


def run_bot() -> None:
    """Run the bot loop forever (Ctrl+C to stop)."""
    if _db_available:
        init_db()
    client = _Client()
    account = client.get_account()
    tp_pct, trail_atr, stop_atr = _tp_trail_stop()
    interval = _check_interval()
    logger.info(
        "Bot started. broker=%s, mode=%s, timeframe=%s, status=%s, symbol=%s, paper=%s, capital_pct=%.0f%%, TP=%.2f%%, trail=%.1fxATR, SL=%.1fxATR, check_every=%ds",
        BROKER,
        BOT_MODE,
        BAR_TIMEFRAME,
        account.status,
        client.symbol,
        getattr(client, "is_paper", False),
        CAPITAL_PCT * 100,
        tp_pct * 100,
        trail_atr,
        stop_atr,
        interval,
    )

    while True:
        try:
            run_once(client)
        except Exception as e:
            logger.exception("Cycle error: %s", e)
        logger.info("Sleeping %s seconds...", interval)
        time.sleep(interval)




