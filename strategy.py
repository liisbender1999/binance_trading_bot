"""
Crypto-friendly default: enter when RSI > 30 only (more trades). Optional: MACD, BB, SMA50.

Set RSI_REQUIRE_MACD / RSI_REQUIRE_BB / RSI_REQUIRE_SMA50 = True to add filters (fewer, stricter entries).
Exit: TP, trailing stop, SL — use wider TP/SL for crypto (see bot/backtest constants).
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from config import (
    TRADE_STRATEGY,
    ENTRY_MODE,
    ENTRY_MIN_INDICATORS,
    REQUIRE_SR_CONFLUENCE,
    VOLUME_MA_PERIOD as CONFIG_VOLUME_MA_PERIOD,
    SR_LOOKBACK as CONFIG_SR_LOOKBACK,
    SR_TOLERANCE_PCT as CONFIG_SR_TOLERANCE_PCT,
    ADX_PERIOD as CONFIG_ADX_PERIOD,
    ADX_THRESHOLD as CONFIG_ADX_THRESHOLD,
    AROON_PERIOD as CONFIG_AROON_PERIOD,
    STOCH_K_PERIOD as CONFIG_STOCH_K_PERIOD,
    STOCH_D_PERIOD as CONFIG_STOCH_D_PERIOD,
    FIB_LOOKBACK as CONFIG_FIB_LOOKBACK,
    FIB_TOLERANCE_PCT as CONFIG_FIB_TOLERANCE_PCT,
)

logger = logging.getLogger(__name__)

# RSI
RSI_PERIOD = 14
# Buy when RSI is above this (30=oversold bounce, 50=momentum)
RSI_ENTRY_LEVEL = 30
# Optional: only enter when price above this EMA (set to 0 to disable)
RSI_EMA_FILTER_PERIOD = 0
# Optional filters (set True to require; all False = RSI > 30 only, more crypto-friendly)
RSI_REQUIRE_MACD = True  # momentum confirmation: improved 4y backtest return + drawdown
RSI_REQUIRE_BB = False
RSI_REQUIRE_SMA50 = False
BB_PERIOD = 15   # optimized for 2y BTC backtest
BB_STD = 2.5     # optimized for 2y BTC backtest
SMA50_PERIOD = 50
# MACD (used when RSI_REQUIRE_MACD or TRADE_STRATEGY=macd)
MACD_FAST = 8
MACD_SLOW = 17
MACD_SIGNAL = 9
# Min bars (40 enough for RSI+ATR; set 50 if using RSI_REQUIRE_SMA50)
MIN_BARS = 40
ATR_PERIOD = 14
# Volume: buy when current bar volume > MA(volume, period); overridden by config.VOLUME_MA_PERIOD
VOLUME_MA_PERIOD = CONFIG_VOLUME_MA_PERIOD
# Support/Resistance: buy when price at support (near recent low) or breaking above resistance (recent high)
SR_LOOKBACK = CONFIG_SR_LOOKBACK
SR_TOLERANCE_PCT = CONFIG_SR_TOLERANCE_PCT
# ADX: trend strength; signal when ADX > threshold (e.g. 25 = strong trend)
ADX_PERIOD = CONFIG_ADX_PERIOD
ADX_THRESHOLD = CONFIG_ADX_THRESHOLD
# Aroon: signal when Aroon Up > Aroon Down (uptrend)
AROON_PERIOD = CONFIG_AROON_PERIOD
# Stochastic: %K and %D periods; buy when %K > %D and 20 < %K < 80
STOCH_K_PERIOD = CONFIG_STOCH_K_PERIOD
STOCH_D_PERIOD = CONFIG_STOCH_D_PERIOD
# Fibonacci: lookback for swing high/low; buy when price near key retracement (0.382, 0.5, 0.618)
FIB_LOOKBACK = CONFIG_FIB_LOOKBACK
FIB_TOLERANCE_PCT = CONFIG_FIB_TOLERANCE_PCT


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index (0–100)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def macd_line_signal(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    signal_line = ema(macd_line, MACD_SIGNAL)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(close: pd.Series, period: int = BB_PERIOD, num_std: float = BB_STD) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Middle = SMA(period), upper = middle + num_std*std, lower = middle - num_std*std."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (exponential with alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    """Average Directional Index (0–100). Measures trend strength, not direction."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    plus_dm_raw = high - prev_high
    minus_dm_raw = prev_low - low
    plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0.0)
    minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0.0)
    tr_smooth = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / tr_smooth.replace(0, 1e-10)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / tr_smooth.replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    return _wilder_smooth(dx, period)


def aroon(high: pd.Series, low: pd.Series, period: int = AROON_PERIOD) -> Tuple[pd.Series, pd.Series]:
    """Aroon Up and Aroon Down (0–100). Up > Down = uptrend."""
    # Index of highest high in window (0 = current bar); Aroon Up = 100 * (period - idx) / period
    idx_high = high.rolling(window=period).apply(lambda x: x.argmax(), raw=True)
    idx_low = low.rolling(window=period).apply(lambda x: x.argmin(), raw=True)
    aroon_up = 100 * (period - idx_high) / period
    aroon_down = 100 * (period - idx_low) / period
    return aroon_up, aroon_down


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = STOCH_K_PERIOD,
    d_period: int = STOCH_D_PERIOD,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic oscillator: %K and %D (0–100). Fast stoch: raw %K, %D = SMA(%K, d_period)."""
    lowest = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()
    diff = highest - lowest
    raw_k = 100 * (close - lowest) / diff.replace(0, 1e-10)
    k_series = raw_k
    d_series = k_series.rolling(window=d_period).mean()
    return k_series, d_series


# Key Fibonacci retracement ratios (from high: pullback support levels in uptrend)
FIB_RATIOS = (0.382, 0.5, 0.618)


@dataclass
class StrategyResult:
    signal: str
    atr_at_entry: Optional[float] = None
    reason: str = ""


def _compute_rsi_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rsi_entry_level: Optional[int] = None,
    rsi_require_macd: Optional[bool] = None,
) -> StrategyResult:
    """Enter when RSI > entry level (default RSI_ENTRY_LEVEL)."""
    level = rsi_entry_level if rsi_entry_level is not None else RSI_ENTRY_LEVEL
    require_macd = rsi_require_macd if rsi_require_macd is not None else RSI_REQUIRE_MACD
    rsi_series = rsi(close, RSI_PERIOD)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_rsi = rsi_series.iloc[i]
    curr_atr = atr_series.iloc[i]
    curr_close = close.iloc[i]
    if pd.isna(curr_rsi) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="nan_indicator")
    if not (curr_rsi > level):
        return StrategyResult("hold", reason=f"rsi_below_{level}")
    # Optional: price above short EMA
    if RSI_EMA_FILTER_PERIOD > 0:
        ema_series = ema(close, RSI_EMA_FILTER_PERIOD)
        curr_ema = ema_series.iloc[i]
        if pd.isna(curr_ema):
            return StrategyResult("hold", reason="ema_nan")
        if curr_close <= curr_ema:
            return StrategyResult("hold", reason="below_ema")
    # Optional: MACD above signal (momentum confirmation)
    if require_macd:
        macd_line, signal_line, _ = macd_line_signal(close)
        curr_macd = macd_line.iloc[i]
        curr_signal = signal_line.iloc[i]
        if pd.isna(curr_macd) or pd.isna(curr_signal):
            return StrategyResult("hold", reason="macd_nan")
        if not (curr_macd > curr_signal):
            return StrategyResult("hold", reason="macd_below_signal")
    # Optional: price above lower Bollinger Band
    if RSI_REQUIRE_BB:
        _, _, lower_bb = bollinger_bands(close, BB_PERIOD, BB_STD)  # no overrides in filter
        curr_lower = lower_bb.iloc[i]
        if pd.isna(curr_lower):
            return StrategyResult("hold", reason="bb_nan")
        if curr_close <= curr_lower:
            return StrategyResult("hold", reason="below_lower_bb")
    # Optional: price above 50-day SMA (medium-term trend)
    if RSI_REQUIRE_SMA50:
        sma50 = close.rolling(window=SMA50_PERIOD).mean()
        curr_sma50 = sma50.iloc[i]
        if pd.isna(curr_sma50):
            return StrategyResult("hold", reason="sma50_nan")
        if curr_close <= curr_sma50:
            return StrategyResult("hold", reason="below_sma50")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="entry_ok")


def _compute_macd_signal(bars: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series) -> StrategyResult:
    """Enter when MACD is above signal."""
    macd_line, signal_line, _ = macd_line_signal(close)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_macd = macd_line.iloc[i]
    curr_signal = signal_line.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_macd) or pd.isna(curr_signal) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="nan_indicator")
    if not (curr_macd > curr_signal):
        return StrategyResult("hold", reason="macd_below_signal")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="macd_ok")


def _compute_bb_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    bb_period: Optional[int] = None,
    bb_std: Optional[float] = None,
) -> StrategyResult:
    """Enter when price at or below lower Bollinger Band (oversold bounce)."""
    period = bb_period if bb_period is not None else BB_PERIOD
    num_std = bb_std if bb_std is not None else BB_STD
    _, _, lower_bb = bollinger_bands(close, period, num_std)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_close = close.iloc[i]
    curr_lower = lower_bb.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_lower) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="nan_indicator")
    if curr_close > curr_lower:
        return StrategyResult("hold", reason="above_lower_bb")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="bb_oversold")


def _compute_volume_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume_ma_period: Optional[int] = None,
) -> StrategyResult:
    """Enter when current bar volume is above the average volume (confirms interest)."""
    if "volume" not in bars.columns:
        return StrategyResult("hold", reason="no_volume_data")
    vol = bars["volume"].astype(float)
    if vol.isna().all() or (vol <= 0).all():
        return StrategyResult("hold", reason="volume_nan")
    period = volume_ma_period if volume_ma_period is not None else VOLUME_MA_PERIOD
    vol_ma = vol.rolling(window=period).mean()
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_vol = vol.iloc[i]
    curr_vol_ma = vol_ma.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_vol) or pd.isna(curr_vol_ma) or curr_vol_ma <= 0 or pd.isna(curr_atr):
        return StrategyResult("hold", reason="volume_nan")
    if curr_vol <= curr_vol_ma:
        return StrategyResult("hold", reason="volume_below_avg")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="volume_above_avg")


def _compute_sr_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    sr_lookback: Optional[int] = None,
    sr_tolerance_pct: Optional[float] = None,
) -> StrategyResult:
    """Enter when price is at support (near recent low) or breaking above resistance (recent high)."""
    lookback = sr_lookback if sr_lookback is not None else SR_LOOKBACK
    tolerance = sr_tolerance_pct if sr_tolerance_pct is not None else SR_TOLERANCE_PCT
    if len(bars) < lookback:
        return StrategyResult("hold", reason="sr_insufficient_bars")
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    support = low.rolling(window=lookback).min().iloc[i]
    resistance = high.rolling(window=lookback).max().iloc[i]
    curr_close = close.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(support) or pd.isna(resistance) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="sr_nan")
    # At support: close within tolerance above the recent low (buy the dip)
    at_support = curr_close <= support * (1 + tolerance)
    # Breakout: close at or above the recent high (buy the breakout)
    at_resistance = curr_close >= resistance * (1 - tolerance)
    if at_support:
        return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="at_support")
    if at_resistance:
        return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="breakout_resistance")
    return StrategyResult("hold", reason="sr_no_level")


def _compute_adx_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    adx_period: Optional[int] = None,
    adx_threshold: Optional[int] = None,
) -> StrategyResult:
    """Enter when ADX is above threshold (strong trend = favorable to trade)."""
    period = adx_period if adx_period is not None else ADX_PERIOD
    threshold = adx_threshold if adx_threshold is not None else ADX_THRESHOLD
    adx_series = adx(high, low, close, period)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_adx = adx_series.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_adx) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="adx_nan")
    if curr_adx <= threshold:
        return StrategyResult("hold", reason=f"adx_below_{threshold}")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="adx_strong_trend")


def _compute_aroon_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    aroon_period: Optional[int] = None,
) -> StrategyResult:
    """Enter when Aroon Up > Aroon Down (uptrend: recent highs more recent than recent lows)."""
    period = aroon_period if aroon_period is not None else AROON_PERIOD
    if len(bars) < period:
        return StrategyResult("hold", reason="aroon_insufficient_bars")
    aroon_up_series, aroon_down_series = aroon(high, low, period)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_up = aroon_up_series.iloc[i]
    curr_down = aroon_down_series.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_up) or pd.isna(curr_down) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="aroon_nan")
    if curr_up <= curr_down:
        return StrategyResult("hold", reason="aroon_downtrend")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="aroon_uptrend")


def _compute_stoch_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    stoch_k_period: Optional[int] = None,
    stoch_d_period: Optional[int] = None,
) -> StrategyResult:
    """Enter when %K > %D and 20 < %K < 80 (bullish momentum, not overbought)."""
    k_period = stoch_k_period if stoch_k_period is not None else STOCH_K_PERIOD
    d_period = stoch_d_period if stoch_d_period is not None else STOCH_D_PERIOD
    if len(bars) < k_period + d_period:
        return StrategyResult("hold", reason="stoch_insufficient_bars")
    k_series, d_series = stochastic(high, low, close, k_period, d_period)
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    curr_k = k_series.iloc[i]
    curr_d = d_series.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_k) or pd.isna(curr_d) or pd.isna(curr_atr):
        return StrategyResult("hold", reason="stoch_nan")
    if curr_k <= curr_d:
        return StrategyResult("hold", reason="stoch_k_below_d")
    if curr_k <= 20 or curr_k >= 80:
        return StrategyResult("hold", reason="stoch_outside_zone")
    return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="stoch_bullish")


def _compute_fib_signal(
    bars: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    fib_lookback: Optional[int] = None,
    fib_tolerance_pct: Optional[float] = None,
) -> StrategyResult:
    """Enter when price is near a Fibonacci retracement support (0.382, 0.5, 0.618) in an uptrend."""
    lookback = fib_lookback if fib_lookback is not None else FIB_LOOKBACK
    tolerance = fib_tolerance_pct if fib_tolerance_pct is not None else FIB_TOLERANCE_PCT
    if len(bars) < lookback:
        return StrategyResult("hold", reason="fib_insufficient_bars")
    atr_series = atr(high, low, close, ATR_PERIOD)
    i = len(bars) - 1
    swing_high = high.iloc[i - lookback + 1 : i + 1].max()
    swing_low = low.iloc[i - lookback + 1 : i + 1].min()
    curr_close = close.iloc[i]
    curr_atr = atr_series.iloc[i]
    if pd.isna(curr_atr) or pd.isna(swing_high) or pd.isna(swing_low):
        return StrategyResult("hold", reason="fib_nan")
    range_ = swing_high - swing_low
    if range_ <= 0:
        return StrategyResult("hold", reason="fib_flat_range")
    if swing_high <= swing_low:
        return StrategyResult("hold", reason="fib_no_uptrend")
    for ratio in FIB_RATIOS:
        level = swing_high - ratio * range_
        if level <= 0:
            continue
        dist_pct = abs(curr_close - level) / level
        if dist_pct <= tolerance:
            return StrategyResult("buy", atr_at_entry=float(curr_atr), reason="fib_at_support")
    return StrategyResult("hold", reason="fib_no_level")


def compute_signal(bars: pd.DataFrame, **indicator_overrides) -> StrategyResult:
    """
    Entry: single strategy (rsi or macd) or OR mode: enter when at least ENTRY_MIN_INDICATORS
    of RSI, MACD, BB, Volume, Support/Resistance, ADX, Aroon signal. indicator_overrides: optional
    rsi_entry_level, bb_period, bb_std, volume_ma_period, sr_lookback, sr_tolerance_pct,
    adx_period, adx_threshold, aroon_period, entry_min_indicators for backtest/optimization.
    """
    if bars is None or len(bars) < MIN_BARS or "close" not in bars.columns:
        return StrategyResult("hold", reason="not_enough_data")

    close = bars["close"].astype(float)
    high = bars["high"].astype(float) if "high" in bars.columns else close
    low = bars["low"].astype(float) if "low" in bars.columns else close
    rsi_level = indicator_overrides.get("rsi_entry_level")
    rsi_require_macd = indicator_overrides.get("rsi_require_macd")
    bb_period = indicator_overrides.get("bb_period")
    bb_std = indicator_overrides.get("bb_std")
    volume_ma_period = indicator_overrides.get("volume_ma_period")
    sr_lookback = indicator_overrides.get("sr_lookback")
    sr_tolerance_pct = indicator_overrides.get("sr_tolerance_pct")
    adx_period = indicator_overrides.get("adx_period")
    adx_threshold = indicator_overrides.get("adx_threshold")
    aroon_period = indicator_overrides.get("aroon_period")
    stoch_k_period = indicator_overrides.get("stoch_k_period")
    stoch_d_period = indicator_overrides.get("stoch_d_period")
    fib_lookback = indicator_overrides.get("fib_lookback")
    fib_tolerance_pct = indicator_overrides.get("fib_tolerance_pct")
    min_indicators_override = indicator_overrides.get("entry_min_indicators")

    if ENTRY_MODE == "or":
        indicators = [
            (_compute_rsi_signal, "rsi", {"rsi_entry_level": rsi_level, "rsi_require_macd": rsi_require_macd}),
            (_compute_macd_signal, "macd", {}),
            (_compute_bb_signal, "bb", {"bb_period": bb_period, "bb_std": bb_std}),
            (_compute_volume_signal, "volume", {"volume_ma_period": volume_ma_period}),
            (_compute_sr_signal, "sr", {"sr_lookback": sr_lookback, "sr_tolerance_pct": sr_tolerance_pct}),
            (_compute_adx_signal, "adx", {"adx_period": adx_period, "adx_threshold": adx_threshold}),
            (_compute_aroon_signal, "aroon", {"aroon_period": aroon_period}),
            (_compute_stoch_signal, "stoch", {"stoch_k_period": stoch_k_period, "stoch_d_period": stoch_d_period}),
            (_compute_fib_signal, "fib", {"fib_lookback": fib_lookback, "fib_tolerance_pct": fib_tolerance_pct}),
        ]
        buys = []
        for fn, name, kwargs in indicators:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            res = fn(bars, close, high, low, **kwargs) if kwargs else fn(bars, close, high, low)
            if res.signal == "buy":
                buys.append((name, res))
        num_indicators = len(indicators)
        min_required = max(1, min(num_indicators, int(min_indicators_override if min_indicators_override is not None else ENTRY_MIN_INDICATORS)))
        if len(buys) >= min_required:
            firing_names = [b[0] for b in buys]
            if REQUIRE_SR_CONFLUENCE and "sr" not in firing_names:
                return StrategyResult("hold", reason="no_sr_confluence")
            name, res = buys[0]
            res.reason = "+".join(b[0] for b in buys[:num_indicators])
            return res
        return StrategyResult("hold", reason="no_indicator_signal")

    if TRADE_STRATEGY == "macd":
        return _compute_macd_signal(bars, close, high, low)
    return _compute_rsi_signal(bars, close, high, low, rsi_entry_level=rsi_level, rsi_require_macd=rsi_require_macd)


def should_exit_tp_sl(
    current_price: float,
    entry_price: float,
    atr_at_entry: float,
    take_profit_pct: float = 0.02,
    stop_atr_mult: float = 2.0,
) -> Tuple[bool, str]:
    if entry_price <= 0:
        return False, ""
    take_profit_price = entry_price * (1 + take_profit_pct)
    stop_price = entry_price - stop_atr_mult * atr_at_entry
    if current_price >= take_profit_price:
        return True, "take_profit"
    if current_price <= stop_price:
        return True, "stop_loss"
    return False, ""


# Trailing stop (3× ATR = optimized for crypto in backtest)
TRAIL_ATR_MULT = 3.0

# Tiered TP: first 5%, then +5% each step (10%, 15%, …); take profit when RSI ≤ 65 at that level
TP_STEP_PCT = 0.05           # first target 5%, then add 5% each time (10%, 15%, 20%, …)
RSI_TAKE_AT_TP = 65          # take profit when RSI ≤ 65 at current target; if RSI > 65 hold for next +5%
MAX_TP_PCT = 0.50             # cap target at 50%

# Dynamic TP/SL with EMA and ADX: first target 5%, +5% steps; stop = fixed % ladder
DYNAMIC_TP_FIRST_PCT = 0.05      # first take profit level 5%
DYNAMIC_TP_STEP_PCT = 0.03       # then +3% each step (8%, 11%, 14%, …)
INITIAL_STOP_PCT = 0.03          # initial stop 3% below entry; then 0%, then previous targets as we advance
EMA_TP_PERIOD = 20            # EMA period for "hold or take profit" at target
ADX_TP_THRESHOLD = 25         # ADX >= this = strong trend → hold; below = take profit
ADX_TP_FALLING_BARS = 3       # ADX falling over this many bars → take profit (trend weakening)


def should_exit_dynamic(
    current_price: float,
    entry_price: float,
    atr_at_entry: float,
    high_water_mark: float,
    current_target_pct: float,
    raised_stop: Optional[float],
    bars_for_rsi: Optional[pd.DataFrame],
    tp_step_pct: float = TP_STEP_PCT,
    stop_atr_mult: float = 2.5,
    trail_atr_mult: float = TRAIL_ATR_MULT,
    max_tp_pct: float = MAX_TP_PCT,
    rsi_take_threshold: float = RSI_TAKE_AT_TP,
) -> Tuple[bool, str, float, Optional[float]]:
    """
    Tiered exit: targets 5%, 10%, 15%, … At each level, if RSI ≤ 65 take profit; else raise stop and aim for +5% more.
    Returns (should_exit, reason, new_current_target_pct, new_raised_stop).
    """
    if entry_price <= 0:
        return False, "", current_target_pct, raised_stop

    initial_stop = entry_price - stop_atr_mult * atr_at_entry
    current_tp_price = entry_price * (1 + current_target_pct)
    raised_stop_value = raised_stop  # use existing raised stop when we've already passed a level

    # Stop / trail: exit if price falls to effective stop
    trail_stop = high_water_mark - trail_atr_mult * atr_at_entry
    effective_stop = max(initial_stop, trail_stop)
    if raised_stop is not None:
        effective_stop = max(effective_stop, raised_stop)
    if current_price <= effective_stop:
        reason = "stop_loss" if current_price < entry_price else "raised_stop"
        return True, reason, current_target_pct, raised_stop

    # At or past current target level?
    if current_price >= current_tp_price:
        # Check RSI: take profit when RSI ≤ 65 (momentum fading), else hold for next +5%
        take_here = False
        if bars_for_rsi is not None and len(bars_for_rsi) >= RSI_PERIOD + 1 and "close" in bars_for_rsi.columns:
            close = bars_for_rsi["close"].astype(float)
            rsi_series = rsi(close, RSI_PERIOD)
            curr_rsi = rsi_series.iloc[-1]
            if not pd.isna(curr_rsi) and curr_rsi <= rsi_take_threshold:
                take_here = True  # RSI ≤ 65 → take profit at this level
        if take_here:
            return True, "take_profit", current_target_pct, raised_stop
        # Hold: lock in current level as new stop, next target = current + 5%
        new_raised = entry_price * (1 + current_target_pct)
        next_target = min(current_target_pct + tp_step_pct, max_tp_pct)
        return False, "", next_target, new_raised

    return False, "", current_target_pct, raised_stop


def should_exit_dynamic_ema_adx(
    current_price: float,
    entry_price: float,
    atr_at_entry: float,
    high_water_mark: float,
    current_target_pct: float,
    raised_stop: Optional[float],
    bars_for_indicators: Optional[pd.DataFrame],
    tp_first_pct: float = DYNAMIC_TP_FIRST_PCT,
    tp_step_pct: float = DYNAMIC_TP_STEP_PCT,
    initial_stop_pct: float = INITIAL_STOP_PCT,
    max_tp_pct: float = MAX_TP_PCT,
    ema_period: int = EMA_TP_PERIOD,
    adx_threshold: int = ADX_TP_THRESHOLD,
    adx_falling_bars: int = ADX_TP_FALLING_BARS,
) -> Tuple[bool, str, float, Optional[float]]:
    """
    Dynamic TP/SL: first target 5%, then +5% steps (EMA/ADX decide hold or take profit).
    Stop = ATR-based, recalculated every bar from current bars; floor = previous target when raised.
    Returns (should_exit, reason, new_current_target_pct, new_raised_stop).
    """
    if entry_price <= 0:
        logger.debug("exit[ema_adx]: skip (entry_price<=0)")
        return False, "", current_target_pct, raised_stop

    # Manual, stepped stop ladder:
    # - Initial stop 5% below entry
    # - Once we decide to hold at first target, stop moves to entry (breakeven)
    # - On later holds, stop moves to previous targets (7%, 12%, 17%, …)
    initial_stop = entry_price * (1 - initial_stop_pct)
    effective_stop = initial_stop
    if raised_stop is not None:
        effective_stop = max(effective_stop, raised_stop)

    if current_price <= effective_stop:
        # Outcome-based: stop_loss only when actual loss; raised_stop when we lock in profit or breakeven
        reason = "stop_loss" if current_price < entry_price else "raised_stop"
        logger.info(
            "exit[ema_adx]: %s at stop %.4f (entry=%.4f, price=%.4f, target=%.2f%%, raised_stop=%s)",
            reason,
            effective_stop,
            entry_price,
            current_price,
            current_target_pct * 100,
            f\"{raised_stop:.4f}\" if raised_stop is not None else \"None\",
        )
        return True, reason, current_target_pct, raised_stop

    current_tp_price = entry_price * (1 + current_target_pct)
    if current_price < current_tp_price:
        logger.debug(
            "exit[ema_adx]: hold below target (price=%.4f, tp_price=%.4f, target=%.2f%%)",
            current_price,
            current_tp_price,
            current_target_pct * 100,
        )
        return False, "", current_target_pct, raised_stop

    # At or past current target: use EMA and ADX to hold or take profit
    take_profit_here = True  # default: take profit if we can't compute indicators
    if bars_for_indicators is not None and len(bars_for_indicators) >= max(ema_period, ADX_PERIOD) + adx_falling_bars:
        close = bars_for_indicators["close"].astype(float)
        high = bars_for_indicators["high"].astype(float) if "high" in bars_for_indicators.columns else close
        low = bars_for_indicators["low"].astype(float) if "low" in bars_for_indicators.columns else close
        ema_series = ema(close, ema_period)
        adx_series = adx(high, low, close, ADX_PERIOD)
        curr_ema = ema_series.iloc[-1]
        curr_adx = adx_series.iloc[-1]
        adx_prev = adx_series.iloc[-1 - adx_falling_bars] if len(adx_series) > adx_falling_bars else curr_adx
        if not pd.isna(curr_ema) and not pd.isna(curr_adx):
            # Take profit when momentum fading (price below EMA) or trend weak/weakening (ADX low or falling)
            if current_price <= curr_ema:
                take_profit_here = True   # price below EMA → take profit
                logger.info(
                    "exit[ema_adx]: take_profit (price<=EMA) price=%.4f ema=%.4f adx=%.2f prev_adx=%.2f target=%.2f%%",
                    current_price,
                    curr_ema,
                    curr_adx,
                    adx_prev,
                    current_target_pct * 100,
                )
            elif curr_adx < adx_threshold:
                take_profit_here = True   # weak trend → take profit
                logger.info(
                    "exit[ema_adx]: take_profit (ADX<threshold) price=%.4f ema=%.4f adx=%.2f<th=%.2f target=%.2f%%",
                    current_price,
                    curr_ema,
                    curr_adx,
                    float(adx_threshold),
                    current_target_pct * 100,
                )
            elif curr_adx < adx_prev:
                take_profit_here = True   # ADX falling → take profit
                logger.info(
                    "exit[ema_adx]: take_profit (ADX falling) price=%.4f ema=%.4f adx=%.2f<prev=%.2f target=%.2f%%",
                    current_price,
                    curr_ema,
                    curr_adx,
                    adx_prev,
                    current_target_pct * 100,
                )
            else:
                take_profit_here = False  # hold: price above EMA, ADX strong and rising
                logger.info(
                    "exit[ema_adx]: hold (trend strong) price=%.4f ema=%.4f adx=%.2f>=max(th=%.2f, prev=%.2f) target=%.2f%%",
                    current_price,
                    curr_ema,
                    curr_adx,
                    float(adx_threshold),
                    adx_prev,
                    current_target_pct * 100,
                )

    if take_profit_here:
        logger.info(
            "exit[ema_adx]: TAKE PROFIT at target %.2f%% (price=%.4f entry=%.4f raised_stop=%s)",
            current_target_pct * 100,
            current_price,
            entry_price,
            f\"{raised_stop:.4f}\" if raised_stop is not None else \"None\",
        )
        return True, "take_profit", current_target_pct, raised_stop

    # Hold: move stop to previous target (first hit -> breakeven, then each step locks previous level)
    if current_target_pct <= tp_first_pct:
        new_raised_stop = entry_price  # breakeven when first target (5%) hit
    else:
        prev_target_pct = current_target_pct - tp_step_pct
        new_raised_stop = entry_price * (1 + prev_target_pct)
    next_target = min(current_target_pct + tp_step_pct, max_tp_pct)
    logger.info(
        "exit[ema_adx]: HOLD, move stop to %.4f and next target to %.2f%% (current=%.2f%%)",
        new_raised_stop,
        next_target * 100,
        current_target_pct * 100,
    )
    return False, "", next_target, new_raised_stop


# Stop-loss behaviour
BREAKEVEN_TRIGGER_PCT = 0.03   # once price was 3% above entry, never exit below entry
USE_INITIAL_STOP = False       # if False, only trailing stop (no "entry - N*ATR") — fewer shake-outs in crypto


def should_exit_tp_sl_trailing(
    current_price: float,
    entry_price: float,
    atr_at_entry: float,
    high_water_mark: float,
    take_profit_pct: float = 0.02,
    stop_atr_mult: float = 2.0,
    trail_atr_mult: float = TRAIL_ATR_MULT,
    breakeven_trigger_pct: float = BREAKEVEN_TRIGGER_PCT,
    use_initial_stop: bool = USE_INITIAL_STOP,
) -> Tuple[bool, str]:
    """TP at +X%, then trailing stop; optional initial stop; breakeven once up a bit."""
    if entry_price <= 0:
        return False, ""
    take_profit_price = entry_price * (1 + take_profit_pct)
    initial_stop = entry_price - stop_atr_mult * atr_at_entry
    trail_stop = high_water_mark - trail_atr_mult * atr_at_entry
    effective_stop = trail_stop
    if use_initial_stop:
        effective_stop = max(effective_stop, initial_stop)
    # Breakeven: once we were up by breakeven_trigger_pct, don't exit below entry
    if high_water_mark >= entry_price * (1 + breakeven_trigger_pct):
        effective_stop = max(effective_stop, entry_price)
    if current_price >= take_profit_price:
        return True, "take_profit"
    if current_price <= effective_stop:
        if effective_stop >= entry_price:
            return True, "trailing_stop"
        return True, "stop_loss"
    return False, ""

