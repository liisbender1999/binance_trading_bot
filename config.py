"""Load configuration from environment (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env: first from this file's folder, then from current working directory
_project_dir = Path(__file__).resolve().parent
_env_file = _project_dir / ".env"
load_dotenv(_env_file)
if not _env_file.exists():
    load_dotenv(Path.cwd() / ".env")

# Binance USD-M Futures API. Use demo/testnet for paper trading.
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", os.getenv("BINANCE_TESTNET_API_KEY", "")).strip()
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", os.getenv("BINANCE_TESTNET_SECRET", "")).strip()
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").strip().lower() in ("true", "1", "yes")
BINANCE_LEVERAGE = int(os.getenv("BINANCE_LEVERAGE", "3"))  # e.g. 3x for long

# Trading parameters
TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "BTCUSD").strip()
# When True, skip market-hours check (crypto trades 24/7)
TRADE_CRYPTO = os.getenv("TRADE_CRYPTO", "true").strip().lower() in ("true", "1", "yes")
POSITION_SIZE = int(os.getenv("POSITION_SIZE", "1"))
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))
# Entry strategy: "rsi" (good for crypto) or "macd" (used when ENTRY_MODE=single)
TRADE_STRATEGY = os.getenv("TRADE_STRATEGY", "rsi").strip().lower()
# Entry mode: "single" = only TRADE_STRATEGY; "or" = enter when RSI OR MACD OR BB signals
ENTRY_MODE = os.getenv("ENTRY_MODE", "or").strip().lower()
# When ENTRY_MODE=or: require this many indicators to agree (1=any one, 2=at least two of RSI/MACD/BB/Volume/SR/ADX/Aroon). Higher = fewer trades, usually higher win rate.
ENTRY_MIN_INDICATORS = int(os.getenv("ENTRY_MIN_INDICATORS", "2"))
# When True (and ENTRY_MODE=or): only enter when Support/Resistance is among the firing indicators (SR confluence). Default false = higher return and more trades.
REQUIRE_SR_CONFLUENCE = os.getenv("REQUIRE_SR_CONFLUENCE", "false").strip().lower() in ("true", "1", "yes")
# Max open positions (lots) at once; new entry allowed when current lots < this (e.g. 4 = up to 4 entries)
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "4"))
# Volume indicator: period for average volume (current bar volume must be above this MA to count as "volume" signal)
VOLUME_MA_PERIOD = int(os.getenv("VOLUME_MA_PERIOD", "25"))
# Support/Resistance: lookback bars for recent low (support) and recent high (resistance)
SR_LOOKBACK = int(os.getenv("SR_LOOKBACK", "20"))
# Tolerance: price within this fraction of support/resistance counts as "at level" (e.g. 0.01 = 1%)
SR_TOLERANCE_PCT = float(os.getenv("SR_TOLERANCE_PCT", "0.01"))
# ADX: period and threshold; signal = "buy" when ADX > threshold (strong trend)
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", "25"))
# Aroon: period; signal = "buy" when Aroon Up > Aroon Down (uptrend)
AROON_PERIOD = int(os.getenv("AROON_PERIOD", "25"))
# Stochastic: K period, D period; signal when %K > %D and 20 < %K < 80 (momentum, not overbought)
STOCH_K_PERIOD = int(os.getenv("STOCH_K_PERIOD", "14"))
STOCH_D_PERIOD = int(os.getenv("STOCH_D_PERIOD", "3"))
# Fibonacci retracements: lookback for swing high/low; buy when price near 0.382/0.5/0.618 support
FIB_LOOKBACK = int(os.getenv("FIB_LOOKBACK", "20"))
FIB_TOLERANCE_PCT = float(os.getenv("FIB_TOLERANCE_PCT", "0.01"))  # 1% of price

# Exit: fixed 1.5% TP / 1.5% SL (straight) or dynamic TP/SL with EMA/ADX
USE_FIXED_TP_SL = os.getenv("USE_FIXED_TP_SL", "true").strip().lower() in ("true", "1", "yes")
FIXED_TP_PCT = float(os.getenv("FIXED_TP_PCT", "0.015"))   # 1.5% take profit
FIXED_SL_PCT = float(os.getenv("FIXED_SL_PCT", "0.015"))   # 1.5% stop loss
USE_DYNAMIC_TP_EMA_ADX = os.getenv("USE_DYNAMIC_TP_EMA_ADX", "false").strip().lower() in ("true", "1", "yes")

# Mode: "swing" = daily bars, fewer trades; "scalping" = intraday bars, more trades per day
BOT_MODE = os.getenv("BOT_MODE", "swing").strip().lower()
# Bar timeframe: 1Day (swing), 1Hour, 5Min or 1Min (scalping). More bars per day = more signals.
BAR_TIMEFRAME = os.getenv("BAR_TIMEFRAME", "").strip() or ("5Min" if BOT_MODE == "scalping" else "1Day")
# How many bars to fetch (need MIN_BARS+ for RSI/ATR). 5Min: 100 bars = ~8h; 1Min: 100 = ~1.7h
BAR_LIMIT = int(os.getenv("BAR_LIMIT", "100" if BOT_MODE == "scalping" else "80"))
# Minimum bars required to run strategy (lower for crypto if data is limited, e.g. 20)
MIN_BARS = int(os.getenv("MIN_BARS", "40"))

# Scalping: tighter TP/SL and faster checks (used when BOT_MODE=scalping unless overridden)
SCALP_TP_PCT = float(os.getenv("SCALP_TP_PCT", "0.01"))   # 1% take profit
SCALP_TRAIL_ATR_MULT = float(os.getenv("SCALP_TRAIL_ATR_MULT", "1.5"))
SCALP_STOP_ATR_MULT = float(os.getenv("SCALP_STOP_ATR_MULT", "1.0"))
SCALP_BREAKEVEN_TRIGGER_PCT = float(os.getenv("SCALP_BREAKEVEN_TRIGGER_PCT", "0.005"))  # 0.5% up -> lock breakeven
SCALP_CHECK_INTERVAL = int(os.getenv("SCALP_CHECK_INTERVAL", "30"))   # seconds


def validate_config() -> None:
    """Raise if required Binance env vars are missing."""
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        raise ValueError(
            "Missing BINANCE_API_KEY or BINANCE_SECRET_KEY.\n"
            f"Create a .env file in: {_project_dir}\n"
            "With: BINANCE_API_KEY=... and BINANCE_SECRET_KEY=...\n"
            "Demo keys: https://testnet.binancefuture.com (Futures Testnet)"
        )
