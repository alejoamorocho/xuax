"""
SIGNAL-BASED FEATURES - Features that PREDICT, not describe

The problem with current features:
- RSI = 65 (so what?)
- MACD = 0.002 (means nothing)

What traders actually use:
- RSI crossed ABOVE 30 (oversold reversal signal)
- MACD crossed bullish (momentum shift)
- Price broke above Bollinger upper (breakout)

This module creates SIGNAL features that distinguish winners from losers.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_crossover(fast: pd.Series, slow: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Detect crossover signals.

    Returns:
        cross_up: 1 when fast crosses above slow
        cross_down: 1 when fast crosses below slow
    """
    prev_diff = (fast.shift(1) - slow.shift(1))
    curr_diff = (fast - slow)

    cross_up = ((prev_diff <= 0) & (curr_diff > 0)).astype(float)
    cross_down = ((prev_diff >= 0) & (curr_diff < 0)).astype(float)

    return cross_up, cross_down


def compute_ma_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Moving Average SIGNAL features (not values).

    Signals:
    1. Golden Cross (MA20 crosses above MA50)
    2. Death Cross (MA20 crosses below MA50)
    3. Price above/below MAs
    4. MA alignment (all MAs aligned = strong trend)
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']

    # Calculate MAs
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # Signal 1-2: Golden/Death Cross
    cross_up, cross_down = compute_crossover(ma20, ma50)
    result[f'{prefix}ma_golden_cross'] = cross_up
    result[f'{prefix}ma_death_cross'] = cross_down

    # Signal 3: Price position relative to MAs
    result[f'{prefix}price_above_ma20'] = (close > ma20).astype(float)
    result[f'{prefix}price_above_ma50'] = (close > ma50).astype(float)
    result[f'{prefix}price_above_ma200'] = (close > ma200).astype(float)

    # Signal 4: MA Alignment Score (-3 to +3)
    # +3 = strong bullish (price > ma10 > ma20 > ma50)
    # -3 = strong bearish
    alignment = (
        (close > ma10).astype(float) +
        (ma10 > ma20).astype(float) +
        (ma20 > ma50).astype(float) +
        (close > ma200).astype(float)
    ) - 2  # Center around 0
    result[f'{prefix}ma_alignment'] = alignment / 2  # Normalize to -1 to +1

    # Signal 5: Recent Golden/Death cross (within last 5 bars)
    result[f'{prefix}recent_golden_cross'] = cross_up.rolling(5).max()
    result[f'{prefix}recent_death_cross'] = cross_down.rolling(5).max()

    return result.fillna(0)


def compute_macd_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    MACD SIGNAL features.

    Signals:
    1. MACD Bullish Crossover (MACD crosses above signal)
    2. MACD Bearish Crossover
    3. MACD Histogram direction change
    4. MACD above/below zero
    5. Bullish/Bearish Divergence
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']

    # Calculate MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    # Signal 1-2: MACD Crossovers
    cross_up, cross_down = compute_crossover(macd_line, signal_line)
    result[f'{prefix}macd_bullish_cross'] = cross_up
    result[f'{prefix}macd_bearish_cross'] = cross_down

    # Signal 3: Histogram direction change
    hist_up = ((histogram > histogram.shift(1)) & (histogram.shift(1) <= histogram.shift(2))).astype(float)
    hist_down = ((histogram < histogram.shift(1)) & (histogram.shift(1) >= histogram.shift(2))).astype(float)
    result[f'{prefix}macd_hist_turn_up'] = hist_up
    result[f'{prefix}macd_hist_turn_down'] = hist_down

    # Signal 4: MACD above/below zero
    result[f'{prefix}macd_above_zero'] = (macd_line > 0).astype(float)

    # Signal 5: Recent crossover (within last 3 bars)
    result[f'{prefix}recent_macd_bullish'] = cross_up.rolling(3).max()
    result[f'{prefix}recent_macd_bearish'] = cross_down.rolling(3).max()

    # Signal 6: MACD Divergence (price makes new high but MACD doesn't)
    price_high_20 = close.rolling(20).max()
    macd_high_20 = macd_line.rolling(20).max()

    new_price_high = (close >= price_high_20 * 0.999)  # Within 0.1% of high
    macd_not_high = (macd_line < macd_high_20 * 0.95)  # MACD not confirming

    result[f'{prefix}macd_bearish_divergence'] = (new_price_high & macd_not_high).astype(float)

    return result.fillna(0)


def compute_rsi_signals(df: pd.DataFrame, period: int = 14, prefix: str = '') -> pd.DataFrame:
    """
    RSI SIGNAL features.

    Signals:
    1. Oversold reversal (RSI crosses above 30)
    2. Overbought reversal (RSI crosses below 70)
    3. RSI zone (oversold/neutral/overbought)
    4. RSI divergence
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']

    # Calculate RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Signal 1: Oversold reversal (RSI crosses above 30)
    oversold_exit = ((rsi > 30) & (rsi.shift(1) <= 30)).astype(float)
    result[f'{prefix}rsi_oversold_reversal'] = oversold_exit

    # Signal 2: Overbought reversal (RSI crosses below 70)
    overbought_exit = ((rsi < 70) & (rsi.shift(1) >= 70)).astype(float)
    result[f'{prefix}rsi_overbought_reversal'] = overbought_exit

    # Signal 3: RSI zones
    result[f'{prefix}rsi_oversold'] = (rsi < 30).astype(float)
    result[f'{prefix}rsi_overbought'] = (rsi > 70).astype(float)
    result[f'{prefix}rsi_neutral_bullish'] = ((rsi >= 50) & (rsi < 70)).astype(float)

    # Signal 4: RSI turning up from low / down from high
    rsi_turn_up = ((rsi > rsi.shift(1)) & (rsi.shift(1) < 40)).astype(float)
    rsi_turn_down = ((rsi < rsi.shift(1)) & (rsi.shift(1) > 60)).astype(float)
    result[f'{prefix}rsi_turn_bullish'] = rsi_turn_up
    result[f'{prefix}rsi_turn_bearish'] = rsi_turn_down

    return result.fillna(0)


def compute_bollinger_signals(df: pd.DataFrame, period: int = 20, std: float = 2.0, prefix: str = '') -> pd.DataFrame:
    """
    Bollinger Band SIGNAL features.

    Signals:
    1. Breakout above upper band
    2. Breakdown below lower band
    3. Squeeze (low volatility = pending breakout)
    4. Band touch and reversal
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']

    # Calculate Bollinger Bands
    ma = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)

    # Band width (for squeeze detection)
    band_width = (upper - lower) / ma
    avg_width = band_width.rolling(50).mean()

    # Signal 1: Breakout above upper band
    breakout_up = ((close > upper) & (close.shift(1) <= upper.shift(1))).astype(float)
    result[f'{prefix}bb_breakout_up'] = breakout_up

    # Signal 2: Breakdown below lower band
    breakdown = ((close < lower) & (close.shift(1) >= lower.shift(1))).astype(float)
    result[f'{prefix}bb_breakdown'] = breakdown

    # Signal 3: Squeeze (band width below average = volatility contraction)
    result[f'{prefix}bb_squeeze'] = (band_width < avg_width * 0.75).astype(float)

    # Signal 4: Touch and reversal
    # Touched lower band and now moving up
    touched_lower = (close.shift(1) <= lower.shift(1))
    moving_up = (close > close.shift(1))
    result[f'{prefix}bb_lower_reversal'] = (touched_lower & moving_up).astype(float)

    # Touched upper band and now moving down
    touched_upper = (close.shift(1) >= upper.shift(1))
    moving_down = (close < close.shift(1))
    result[f'{prefix}bb_upper_reversal'] = (touched_upper & moving_down).astype(float)

    # Signal 5: Price position within bands (0 to 1)
    bb_position = (close - lower) / (upper - lower + 1e-10)
    result[f'{prefix}bb_position'] = bb_position.clip(0, 1)

    return result.fillna(0)


def compute_support_resistance_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Support/Resistance SIGNAL features.

    Signals:
    1. Breakout above recent high
    2. Breakdown below recent low
    3. Near support/resistance
    4. Failed breakout (breakout then reversal)
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']

    # Recent highs and lows
    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    high_50 = high.rolling(50).max()
    low_50 = low.rolling(50).min()

    # Signal 1: Breakout above recent high
    breakout_20 = ((close > high_20.shift(1)) & (close.shift(1) <= high_20.shift(2))).astype(float)
    breakout_50 = ((close > high_50.shift(1)) & (close.shift(1) <= high_50.shift(2))).astype(float)
    result[f'{prefix}breakout_high_20'] = breakout_20
    result[f'{prefix}breakout_high_50'] = breakout_50

    # Signal 2: Breakdown below recent low
    breakdown_20 = ((close < low_20.shift(1)) & (close.shift(1) >= low_20.shift(2))).astype(float)
    breakdown_50 = ((close < low_50.shift(1)) & (close.shift(1) >= low_50.shift(2))).astype(float)
    result[f'{prefix}breakdown_low_20'] = breakdown_20
    result[f'{prefix}breakdown_low_50'] = breakdown_50

    # Signal 3: Near resistance (within 0.2%)
    near_resistance = (close >= high_20 * 0.998).astype(float)
    near_support = (close <= low_20 * 1.002).astype(float)
    result[f'{prefix}near_resistance'] = near_resistance
    result[f'{prefix}near_support'] = near_support

    # Signal 4: Failed breakout (broke high then closed below)
    was_breakout = (breakout_20.shift(1).fillna(0) + breakout_20.shift(2).fillna(0) + breakout_20.shift(3).fillna(0)) > 0
    now_below = (close < high_20.shift(3))
    result[f'{prefix}failed_breakout'] = (was_breakout & now_below).astype(float)

    return result.fillna(0)


def compute_volume_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Volume SIGNAL features.

    Signals:
    1. Volume spike (unusual volume)
    2. Volume confirmation (price move with volume)
    3. Volume divergence (price move without volume)
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']
    volume = df['volume']

    # Average volume
    avg_vol_20 = volume.rolling(20).mean()

    # Signal 1: Volume spike (>2x average)
    result[f'{prefix}volume_spike'] = (volume > avg_vol_20 * 2).astype(float)
    result[f'{prefix}high_volume'] = (volume > avg_vol_20 * 1.5).astype(float)

    # Signal 2: Volume confirmation
    # Price up + high volume = bullish confirmation
    price_up = (close > close.shift(1))
    high_vol = (volume > avg_vol_20)
    result[f'{prefix}bullish_volume_confirm'] = (price_up & high_vol).astype(float)

    # Price down + high volume = bearish confirmation
    price_down = (close < close.shift(1))
    result[f'{prefix}bearish_volume_confirm'] = (price_down & high_vol).astype(float)

    # Signal 3: Volume divergence (price moves but volume doesn't confirm)
    low_vol = (volume < avg_vol_20 * 0.7)
    result[f'{prefix}weak_move_up'] = (price_up & low_vol).astype(float)
    result[f'{prefix}weak_move_down'] = (price_down & low_vol).astype(float)

    return result.fillna(0)


def compute_momentum_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Momentum SIGNAL features.

    Signals:
    1. Momentum shift (acceleration/deceleration)
    2. Consecutive up/down bars
    3. Strong move (large single bar)
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']

    # Returns
    ret = close.pct_change()

    # Signal 1: Momentum acceleration
    # Positive returns getting larger = acceleration
    accel = ret - ret.shift(1)
    result[f'{prefix}momentum_accelerating'] = (accel > 0).astype(float)

    # Signal 2: Consecutive bars
    up_bar = (ret > 0).astype(float)
    down_bar = (ret < 0).astype(float)

    consec_up = up_bar.rolling(3).sum()
    consec_down = down_bar.rolling(3).sum()

    result[f'{prefix}three_up_bars'] = (consec_up == 3).astype(float)
    result[f'{prefix}three_down_bars'] = (consec_down == 3).astype(float)

    # Signal 3: Strong single bar move (>1 std of returns)
    ret_std = ret.rolling(20).std()
    result[f'{prefix}strong_up_bar'] = (ret > ret_std * 1.5).astype(float)
    result[f'{prefix}strong_down_bar'] = (ret < -ret_std * 1.5).astype(float)

    # Signal 4: Trend strength (ADX-like)
    high = df['high']
    low = df['low']
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    # Directional movement
    plus_dm = ((high - high.shift(1)) > (low.shift(1) - low)) * (high - high.shift(1))
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = ((low.shift(1) - low) > (high - high.shift(1))) * (low.shift(1) - low)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()

    result[f'{prefix}strong_trend'] = (adx > 25).astype(float)
    result[f'{prefix}bullish_trend'] = ((adx > 20) & (plus_di > minus_di)).astype(float)
    result[f'{prefix}bearish_trend'] = ((adx > 20) & (minus_di > plus_di)).astype(float)

    return result.fillna(0)


def compute_candle_pattern_signals(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Candlestick Pattern SIGNAL features.

    Common reversal patterns that traders use.
    """
    result = pd.DataFrame(index=df.index)

    open_p = df['open']
    high = df['high']
    low = df['low']
    close = df['close']

    body = close - open_p
    body_abs = abs(body)
    upper_wick = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_p, close], axis=1).min(axis=1) - low

    # Average body size for reference
    avg_body = body_abs.rolling(20).mean()

    # Signal 1: Hammer (bullish reversal at bottom)
    # Small body, long lower wick, small upper wick
    is_hammer = (
        (lower_wick > body_abs * 2) &
        (upper_wick < body_abs * 0.5) &
        (body_abs < avg_body)
    )
    result[f'{prefix}hammer'] = is_hammer.astype(float)

    # Signal 2: Shooting Star (bearish reversal at top)
    is_shooting_star = (
        (upper_wick > body_abs * 2) &
        (lower_wick < body_abs * 0.5) &
        (body_abs < avg_body)
    )
    result[f'{prefix}shooting_star'] = is_shooting_star.astype(float)

    # Signal 3: Engulfing patterns
    prev_body = body.shift(1)

    # Bullish engulfing: previous red candle fully engulfed by current green
    bullish_engulf = (
        (prev_body < 0) &  # Previous was red
        (body > 0) &  # Current is green
        (close > open_p.shift(1)) &  # Current close > prev open
        (open_p < close.shift(1))  # Current open < prev close
    )
    result[f'{prefix}bullish_engulfing'] = bullish_engulf.astype(float)

    # Bearish engulfing
    bearish_engulf = (
        (prev_body > 0) &  # Previous was green
        (body < 0) &  # Current is red
        (close < open_p.shift(1)) &  # Current close < prev open
        (open_p > close.shift(1))  # Current open > prev close
    )
    result[f'{prefix}bearish_engulfing'] = bearish_engulf.astype(float)

    # Signal 4: Doji (indecision)
    is_doji = (body_abs < avg_body * 0.1)
    result[f'{prefix}doji'] = is_doji.astype(float)

    # Signal 5: Large bullish/bearish candle
    result[f'{prefix}large_bullish_candle'] = ((body > 0) & (body_abs > avg_body * 2)).astype(float)
    result[f'{prefix}large_bearish_candle'] = ((body < 0) & (body_abs > avg_body * 2)).astype(float)

    return result.fillna(0)


def compute_all_signal_features(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Compute ALL signal-based features for a single timeframe.

    Returns DataFrame with ~60 signal features.
    """
    logger.info(f"Computing signal features for {prefix or 'base'} timeframe...")

    all_signals = []

    # MA signals (~8 features)
    all_signals.append(compute_ma_signals(df, prefix))

    # MACD signals (~7 features)
    all_signals.append(compute_macd_signals(df, prefix))

    # RSI signals (~7 features)
    all_signals.append(compute_rsi_signals(df, prefix=prefix))

    # Bollinger signals (~7 features)
    all_signals.append(compute_bollinger_signals(df, prefix=prefix))

    # Support/Resistance signals (~8 features)
    all_signals.append(compute_support_resistance_signals(df, prefix))

    # Volume signals (~6 features)
    all_signals.append(compute_volume_signals(df, prefix))

    # Momentum signals (~10 features)
    all_signals.append(compute_momentum_signals(df, prefix))

    # Candle patterns (~7 features)
    all_signals.append(compute_candle_pattern_signals(df, prefix))

    result = pd.concat(all_signals, axis=1)

    logger.info(f"   Generated {result.shape[1]} signal features")

    return result


def compute_multi_timeframe_signals(tf_data: Dict[str, pd.DataFrame], base_tf: str = 'M5') -> pd.DataFrame:
    """
    Compute signal features across multiple timeframes and combine.

    Key concept: Signals from higher timeframes are more significant.
    """
    logger.info("="*70)
    logger.info("COMPUTING MULTI-TIMEFRAME SIGNAL FEATURES")
    logger.info("="*70)

    base_index = tf_data[base_tf].index
    all_features = []

    # Priority timeframes for signals
    timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']

    for tf in timeframes:
        if tf in tf_data:
            logger.info(f"\nProcessing {tf}...")
            df = tf_data[tf]

            # Compute signals for this timeframe
            signals = compute_all_signal_features(df, prefix=f'{tf}_')

            # Align to base timeframe
            signals = signals.reindex(base_index, method='ffill')
            all_features.append(signals)

    # Combine all timeframe signals
    result = pd.concat(all_features, axis=1)
    result = result.fillna(0)

    # Add multi-timeframe confluence signals
    logger.info("\nComputing confluence signals...")

    # Bullish confluence: multiple timeframes showing bullish signals
    bullish_cols = [c for c in result.columns if any(x in c for x in ['bullish', 'golden', 'breakout_high', 'oversold_reversal'])]
    if bullish_cols:
        result['mtf_bullish_confluence'] = result[bullish_cols].sum(axis=1) / len(bullish_cols)

    # Bearish confluence
    bearish_cols = [c for c in result.columns if any(x in c for x in ['bearish', 'death', 'breakdown', 'overbought_reversal'])]
    if bearish_cols:
        result['mtf_bearish_confluence'] = result[bearish_cols].sum(axis=1) / len(bearish_cols)

    # Trend alignment across timeframes
    trend_cols = [c for c in result.columns if 'ma_alignment' in c or 'bullish_trend' in c]
    if trend_cols:
        result['mtf_trend_alignment'] = result[trend_cols].mean(axis=1)

    logger.info(f"\n Total signal features: {result.shape[1]}")

    return result


if __name__ == "__main__":
    # Test
    logger.info("Testing signal features...")

    import pandas as pd

    # Load test data
    df = pd.read_csv('data/xauusd_m5.csv')
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()

    # Test single timeframe
    signals = compute_all_signal_features(df[:10000], prefix='M5_')

    logger.info(f"\nGenerated {signals.shape[1]} signal features")
    logger.info(f"Sample:\n{signals.head()}")

    # Show non-zero signal counts
    logger.info("\nSignal occurrence counts:")
    for col in signals.columns[:20]:
        count = (signals[col] != 0).sum()
        logger.info(f"  {col}: {count} signals")
