"""Binance USD-M Futures API client (demo/paper trading with leverage)."""
import logging
import time
from typing import Optional

import pandas as pd

from config import (
    BINANCE_API_KEY,
    BINANCE_LEVERAGE,
    BINANCE_SECRET_KEY,
    BINANCE_TESTNET,
    POSITION_SIZE,
    TRADE_SYMBOL,
)
from config import validate_config as _validate_config

logger = logging.getLogger(__name__)

# Simple namespace for position and account (same interface as Alpaca for bot)
class _Position:
    __slots__ = ("qty", "avg_entry_price")
    def __init__(self, qty: float, avg_entry_price: float):
        self.qty = qty
        self.avg_entry_price = avg_entry_price

class _Account:
    __slots__ = ("buying_power", "status")
    def __init__(self, buying_power: float, status: str = "ACTIVE"):
        self.buying_power = buying_power
        self.status = status

class _Trade:
    __slots__ = ("price",)
    def __init__(self, price: float):
        self.price = price


def _symbol_to_binance(symbol: str) -> str:
    """Map TRADE_SYMBOL to Binance Futures symbol (e.g. BTCUSD -> BTCUSDT)."""
    s = (symbol or "").strip().upper()
    if s.endswith("USDT"):
        return s
    if s.endswith("USD") and len(s) >= 6:
        return s[:-3] + "USDT"
    if s == "BTC":
        return "BTCUSDT"
    return s + "USDT" if len(s) >= 2 else "BTCUSDT"


def _timeframe_to_binance(timeframe: str) -> str:
    """Map our timeframe to Binance kline interval (1m, 5m, 15m, 1h, 1d)."""
    s = (timeframe or "1Day").strip().lower()
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


class BinanceClient:
    """Thin wrapper around Binance USD-M Futures (ccxt) for trading and data."""

    def __init__(self) -> None:
        _validate_config()
        try:
            import ccxt
        except ImportError:
            raise ImportError("Install ccxt: pip install ccxt")
        self._exchange = ccxt.binanceusdm({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET_KEY,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if BINANCE_TESTNET:
            # Use demo trading (sandbox/testnet deprecated for futures in ccxt)
            self._exchange.enable_demo_trading(True)
            logger.info("Binance Futures demo (paper) enabled.")
        self.symbol = _symbol_to_binance(TRADE_SYMBOL)
        self.position_size = POSITION_SIZE
        self.is_paper = BINANCE_TESTNET
        self._set_leverage()

    def _set_leverage(self) -> None:
        """Set leverage for the trading symbol."""
        try:
            self._exchange.set_leverage(BINANCE_LEVERAGE, self.symbol)
            logger.info("Leverage set to %dx for %s", BINANCE_LEVERAGE, self.symbol)
        except Exception as e:
            logger.warning("set_leverage failed (may be ok): %s", e)

    def get_account(self) -> _Account:
        """Get account info (USDT balance as buying power)."""
        try:
            balance = self._exchange.fetch_balance()
            # USD-M Futures: free/usdt or total
            usdt = balance.get("USDT") or {}
            free = float(usdt.get("free") or 0)
            total = float(usdt.get("total") or 0)
            # Use available (free) as buying power for new orders
            bp = free if free > 0 else total
            return _Account(buying_power=bp, status="ACTIVE")
        except Exception as e:
            logger.warning("fetch_balance failed: %s", e)
            return _Account(buying_power=0.0, status="UNKNOWN")

    def get_position(self, symbol: Optional[str] = None) -> Optional[_Position]:
        """Get current long position for symbol, or None if flat."""
        sym = symbol or self.symbol
        try:
            positions = self._exchange.fetch_positions([sym])
            for p in positions:
                if p.get("symbol") == sym:
                    side = p.get("side")
                    contracts = float(p.get("contracts") or 0)
                    if side == "long" and contracts > 0:
                        entry = float(p.get("entryPrice") or 0)
                        return _Position(qty=contracts, avg_entry_price=entry)
                    if side == "short" and contracts > 0:
                        # Bot is long-only; treat short as no position for entry logic
                        return None
            return None
        except Exception as e:
            logger.warning("fetch_positions failed: %s", e)
            return None

    def get_latest_trade(self, symbol: Optional[str] = None) -> Optional[_Trade]:
        """Latest price (from ticker) for symbol."""
        sym = symbol or self.symbol
        try:
            ticker = self._exchange.fetch_ticker(sym)
            last = ticker.get("last")
            if last is not None:
                return _Trade(price=float(last))
        except Exception as e:
            logger.debug("fetch_ticker failed: %s", e)
        return None

    def get_bars(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1Day",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get OHLCV bars. Returns DataFrame with close, high, low and time index. Retries on failure (demo API can be flaky)."""
        sym = symbol or self.symbol
        tf = _timeframe_to_binance(timeframe)
        last_error = None
        for attempt in range(3):
            try:
                ohlcv = self._exchange.fetch_ohlcv(sym, tf, limit=limit)
                break
            except Exception as e:
                last_error = e
                logger.warning("fetch_ohlcv attempt %d/3 failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2)
        else:
            logger.warning("fetch_ohlcv failed after 3 attempts: %s", last_error)
            return pd.DataFrame()
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        for c in ("high", "low"):
            if c not in df.columns:
                df[c] = df["close"]
        return df[["open", "high", "low", "close"]].astype(float)

    def shares_from_pct_buying_power(
        self,
        pct: float = 0.30,
        price: float = 0.0,
        symbol: Optional[str] = None,
    ) -> float:
        """Compute quantity (in base asset, e.g. BTC) for pct of USDT balance, scaled by futures leverage.

        Example: pct=0.30 and BINANCE_LEVERAGE=10 -> ~300% notional exposure (30% margin at 10x).
        """
        if price <= 0:
            trade = self.get_latest_trade(symbol or self.symbol)
            price = float(trade.price) if trade else 0
        if price <= 0:
            return 0.0
        account = self.get_account()
        # Use pct of balance as margin, scaled by futures leverage to get notional size
        value = account.buying_power * pct * BINANCE_LEVERAGE
        qty = value / price
        # Binance Futures: round to symbol's lot size; use 6 decimals for BTC
        return max(0.0, round(qty, 6))

    def submit_market_order(
        self,
        symbol: Optional[str] = None,
        qty: Optional[float] = None,
        side: str = "buy",
    ):
        """Submit market order. side: buy (open long) or sell (close long). qty in base asset (e.g. BTC)."""
        sym = symbol or self.symbol
        amount = float(qty) if qty is not None else self.position_size
        try:
            if side.lower() == "buy":
                order = self._exchange.create_market_buy_order(sym, amount)
            else:
                order = self._exchange.create_market_sell_order(sym, amount)
            logger.info("Order submitted: %s %s qty=%s id=%s", side, sym, amount, order.get("id"))
            return order
        except Exception as e:
            logger.exception("submit_market_order failed: %s", e)
            raise

    def close_position(self, symbol: Optional[str] = None) -> None:
        """Close full position for symbol (market sell)."""
        pos = self.get_position(symbol or self.symbol)
        if pos is None or float(pos.qty) <= 0:
            logger.info("No position to close for %s", symbol or self.symbol)
            return
        self.submit_market_order(symbol=symbol, qty=pos.qty, side="sell")
        logger.info("Closed position: %s", symbol or self.symbol)

    def get_clock(self) -> object:
        """Futures trade 24/7; return object with is_open=True."""
        class Clock:
            is_open = True
        return Clock()

