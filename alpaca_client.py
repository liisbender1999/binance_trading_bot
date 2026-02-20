"""Alpaca API client wrapper for market data and orders."""
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional, Union

import alpaca_trade_api as tradeapi
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit

from config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    POSITION_SIZE,
    TRADE_SYMBOL,
)
from config import validate_config as _validate_config

logger = logging.getLogger(__name__)


def _is_crypto_symbol(symbol: str) -> bool:
    """True if symbol is crypto. Alpaca uses BTCUSD or BTC/USD."""
    if not symbol:
        return False
    s = symbol.upper().strip().replace("/", "")
    if s.endswith("USD") and len(s) >= 6:
        return True
    return False


def _symbol_to_crypto_pair(symbol: str) -> str:
    """BTCUSD -> BTC/USD, BTC/USD -> BTC/USD."""
    s = (symbol or "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USD") and len(s) > 3:
        return f"{s[:-3]}/USD"
    return s


def _timeframe_to_data_api(timeframe: str) -> str:
    """Map our timeframe string to Alpaca Data API (e.g. 1Day, 5Min)."""
    s = (timeframe or "1Day").strip()
    if s.upper() in ("1D", "1DAY", "DAY"):
        return "1Day"
    if s.upper() in ("5M", "5MIN"):
        return "5Min"
    if s.upper() in ("1H", "1HOUR", "HOUR"):
        return "1Hour"
    if s.upper() in ("1M", "1MIN", "MINUTE"):
        return "1Min"
    if s.upper() in ("15M", "15MIN"):
        return "15Min"
    return s if s else "1Day"


def _timeframe_to_yf(timeframe: str) -> tuple[str, str]:
    """Map our timeframe to (period, interval) for yfinance."""
    s = (timeframe or "1Day").strip().lower()
    if s in ("1d", "1day", "day"):
        return "180d", "1d"
    if s in ("1m", "1min", "minute"):
        return "7d", "1m"
    if s in ("5m", "5min"):
        return "30d", "5m"
    if s in ("15m", "15min"):
        return "60d", "15m"
    if s in ("1h", "1hour", "hour"):
        return "180d", "1h"
    # default: daily
    return "180d", "1d"


def _resolve_timeframe(timeframe: Union[str, TimeFrame]) -> TimeFrame:
    """Map string (e.g. '5Min', '1Day') to Alpaca TimeFrame for get_bars."""
    if isinstance(timeframe, TimeFrame):
        return timeframe
    s = (timeframe or "1Day").strip().lower()
    if s in ("1d", "1day", "day"):
        return TimeFrame.Day
    if s in ("1m", "1min", "minute"):
        return TimeFrame.Minute
    if s in ("1h", "1hour", "hour"):
        return TimeFrame.Hour
    if s in ("5m", "5min"):
        return TimeFrame(5, TimeFrameUnit.Minute)
    if s in ("15m", "15min"):
        return TimeFrame(15, TimeFrameUnit.Minute)
    return TimeFrame.Day


class AlpacaClient:
    """Thin wrapper around Alpaca REST API for trading and data."""

    def __init__(self) -> None:
        _validate_config()
        self._api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL,
            api_version="v2",
        )
        self.symbol = TRADE_SYMBOL
        self.position_size = POSITION_SIZE
        self.is_paper = "paper" in (ALPACA_BASE_URL or "").lower()

    def get_account(self):
        """Get account info (buying power, etc.)."""
        return self._api.get_account()

    def get_position(self, symbol: Optional[str] = None) -> Optional[object]:
        """Get current position for symbol, or None if flat."""
        sym = symbol or self.symbol
        try:
            return self._api.get_position(sym)
        except Exception:
            return None

    def get_latest_trade(self, symbol: Optional[str] = None):
        """Latest trade (price) for symbol."""
        sym = symbol or self.symbol
        return self._api.get_latest_trade(sym)

    def _get_crypto_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch crypto bars from Alpaca Data API (stocks API does not return crypto history)."""
        base = "https://data.sandbox.alpaca.markets" if "paper" in (ALPACA_BASE_URL or "").lower() else "https://data.alpaca.markets"
        pair = _symbol_to_crypto_pair(symbol)
        tf_param = _timeframe_to_data_api(timeframe)
        url = f"{base}/v1beta3/crypto/us/bars?symbols={urllib.parse.quote(pair)}&timeframe={urllib.parse.quote(tf_param)}&limit={limit}"
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
        except Exception as e:
            logger.warning("Crypto bars request failed: %s", e)
            return pd.DataFrame()
        bars_by_sym = data.get("bars") or {}
        # Response key may be "BTC/USD"
        series_list = bars_by_sym.get(pair) or list(bars_by_sym.values())[0] if bars_by_sym else []
        if not series_list:
            return pd.DataFrame()
        rows = []
        for b in series_list:
            t = b.get("t")
            c = b.get("c")
            if t is None or c is None:
                continue
            rows.append({
                "timestamp": t,
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": float(c),
                "volume": b.get("v"),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        return df.sort_index()

    def _get_crypto_bars_yf(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fallback: fetch crypto bars from yfinance when Alpaca data API is unavailable."""
        pair = _symbol_to_crypto_pair(symbol)
        # yfinance uses BTC-USD format
        yf_symbol = pair.replace("/", "-")
        period, interval = _timeframe_to_yf(timeframe)
        try:
            hist = yf.Ticker(yf_symbol).history(period=period, interval=interval)
        except Exception as e:
            logger.warning("yfinance crypto bars request failed for %s: %s", yf_symbol, e)
            return pd.DataFrame()
        if hist is None or hist.empty:
            return pd.DataFrame()
        df = hist.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )
        if "close" not in df.columns and "Close" in hist.columns:
            df["close"] = hist["Close"]
        df = df[["close", "high", "low"]].copy()
        df = df.tail(limit)
        return df

    def get_bars(
        self,
        symbol: Optional[str] = None,
        timeframe: str = "1Day",
        limit: int = 100,
    ):
        """Get historical bars (OHLCV). timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day.
        For crypto symbols uses Alpaca Data API; for stocks uses trading API.
        Returns DataFrame with 'close' and time index for strategy.
        """
        sym = symbol or self.symbol
        tf = _resolve_timeframe(timeframe)
        if _is_crypto_symbol(sym):
            # 1) Try Alpaca crypto data API
            df_crypto = self._get_crypto_bars(sym, timeframe, limit)
            if not df_crypto.empty:
                return df_crypto
            # 2) Fallback to yfinance
            logger.warning("Crypto data API returned no bars (or unauthorized), falling back to yfinance for %s", sym)
            df_yf = self._get_crypto_bars_yf(sym, timeframe, limit)
            if not df_yf.empty:
                return df_yf
            # 3) Last resort: trading get_bars (may be empty for crypto)
            logger.warning("yfinance also returned no bars, falling back to trading API get_bars for %s", sym)
        raw = self._api.get_bars(sym, tf, limit=limit)
        return self._bars_to_df(raw)

    def get_bars_range(
        self,
        symbol: Optional[str] = None,
        start: Optional[Union[datetime, str]] = None,
        end: Optional[Union[datetime, str]] = None,
        limit: Optional[int] = None,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Get historical bars for backtesting.
        Prefer limit= (most recent N bars); otherwise uses start/end date strings.
        """
        sym = symbol or self.symbol
        if _is_crypto_symbol(sym):
            df_crypto = self._get_crypto_bars(sym, timeframe, limit=limit or 1000)
            if not df_crypto.empty:
                return df_crypto
            logger.warning("Crypto data API returned no bars (or unauthorized), falling back to yfinance for %s", sym)
            df_yf = self._get_crypto_bars_yf(sym, timeframe, limit or 1000)
            if not df_yf.empty:
                return df_yf
            logger.warning("yfinance also returned no bars, falling back to trading API get_bars_range for %s", sym)
        tf = _resolve_timeframe(timeframe)
        if limit is not None:
            raw = self._api.get_bars(sym, tf, limit=limit)
        else:
            start_str = (start.date().isoformat() if isinstance(start, datetime) else start) if start else None
            end_str = (end.date().isoformat() if isinstance(end, datetime) else end) if end else None
            try:
                if start_str and end_str:
                    raw = self._api.get_bars(sym, tf, start=start_str, end=end_str)
                else:
                    raw = self._api.get_bars(sym, tf, limit=1000)
            except Exception:
                raw = self._api.get_bars(sym, tf, limit=1000)
        return self._bars_to_df(raw)

    def shares_from_pct_buying_power(
        self,
        pct: float = 0.30,
        price: float = 0.0,
        symbol: Optional[str] = None,
    ):
        """Compute share quantity to use pct of buying power at given price. Uses latest trade if price is 0.
        For crypto: returns fractional qty (float). For stocks: returns int."""
        sym = symbol or self.symbol
        if price <= 0:
            trade = self.get_latest_trade(sym)
            price = float(trade.price) if trade else 0
        if price <= 0:
            return 0
        account = self.get_account()
        bp = float(account.buying_power)
        value = bp * pct
        qty = value / price
        if _is_crypto_symbol(sym):
            # Crypto: fractional allowed (e.g. 0.001 BTC); round to 6 decimals
            return max(0.0, round(qty, 6))
        return max(0, int(qty))

    def submit_market_order(
        self,
        symbol: Optional[str] = None,
        qty: Optional[Union[int, float]] = None,
        side: str = "buy",
    ):
        """Submit a market order. side: 'buy' or 'sell'. For crypto, qty can be fractional; time_in_force must be gtc."""
        sym = symbol or self.symbol
        q = qty if qty is not None else self.position_size
        # Crypto does not support time_in_force="day"; use gtc (good till cancelled)
        tif = "gtc" if _is_crypto_symbol(sym) else "day"
        order = self._api.submit_order(
            symbol=sym,
            qty=q,
            side=side,
            type="market",
            time_in_force=tif,
        )
        logger.info("Order submitted: %s %s %s qty=%s", side, sym, order.id, q)
        return order

    def close_position(self, symbol: Optional[str] = None):
        """Close full position for symbol (market)."""
        sym = symbol or self.symbol
        self._api.close_position(sym)
        logger.info("Closed position: %s", sym)

    def get_clock(self):
        """Market open/close times and current state."""
        return self._api.get_clock()

    def _bars_to_df(self, raw) -> pd.DataFrame:
        """Convert API bars response (BarsV2 or list) to DataFrame with 'close' and time index."""
        df = None
        if hasattr(raw, "df"):
            df = raw.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            # Build from list of bar objects (e.g. BarsV2 is list of Bar)
            if hasattr(raw, "__iter__") and hasattr(raw, "__len__") and len(raw) > 0:
                rows = []
                for b in raw:
                    r = getattr(b, "_raw", {})
                    ts = getattr(b, "timestamp", None) or r.get("t")
                    c = getattr(b, "close", None) or r.get("c")
                    if ts is not None and c is not None:
                        row = {"timestamp": ts, "close": float(c)}
                        h, l_ = getattr(b, "high", None), getattr(b, "low", None)
                        if h is not None:
                            row["high"] = float(h)
                        if l_ is not None:
                            row["low"] = float(l_)
                        rows.append(row)
                if rows:
                    df = pd.DataFrame(rows).set_index("timestamp")
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        if "timestamp" in df.columns and df.index.name != "timestamp":
            df = df.set_index("timestamp")
        if "close" not in df.columns:
            for col in ("close", "c"):
                if col in df.columns:
                    df = df.rename(columns={col: "close"})
                    break
            if "close" not in df.columns and len(df.columns) >= 4:
                df = df.rename(columns={df.columns[3]: "close"})
        if "close" not in df.columns:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        # Ensure high/low exist for ATR (strategy); use close if missing
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        return df
