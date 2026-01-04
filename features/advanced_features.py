# features/advanced_features.py
"""
Advanced Features for Gold Trading

Section 4 of PPO Gold Trading Improvements Checklist:
- 4.1 Session features (trading hours)
- 4.2 Position state features (for environment)
- 4.3 TIPS 10Y yield (via yfinance TIP ETF)
- 4.5 VIX with regimes (via yfinance ^VIX)
- 4.6 Economic calendar features
- 4.7 Price action features
- 4.8 Multi-timeframe context features

Note: 4.4 COT data excluded (requires paid data source)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


# ============================================================================
# 4.1 Session Features (Trading Hours)
# ============================================================================
def calculate_session_features(timestamp: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate features related to trading session.

    Gold has very different behavior in each session:
    - Asian: Consolidation, 40-60% less volatility
    - London: Breakouts, trend initiation
    - New York: Continuation or reversal, highest volume
    - London-NY Overlap: Highest liquidity and volatility
    """
    # Convert to UTC if not already
    if timestamp.tz is None:
        # Assume UTC if no timezone
        hour = timestamp.hour
        day = timestamp.dayofweek
    else:
        hour = timestamp.tz_convert('UTC').hour
        day = timestamp.tz_convert('UTC').dayofweek

    features = pd.DataFrame(index=timestamp)

    # Session definitions (in UTC)
    # Asian: 23:00 - 08:00 UTC
    # London: 08:00 - 17:00 UTC
    # New York: 13:00 - 22:00 UTC
    # Overlap: 13:00 - 17:00 UTC

    # Session indicators (one-hot encoded)
    features['session_asian'] = ((hour >= 23) | (hour < 8)).astype(float)
    features['session_london'] = ((hour >= 8) & (hour < 17)).astype(float)
    features['session_ny'] = ((hour >= 13) & (hour < 22)).astype(float)
    features['session_overlap'] = ((hour >= 13) & (hour < 17)).astype(float)

    # Session as single value (for embedding)
    features['session_id'] = 0  # Default Asian
    features.loc[features['session_london'] == 1, 'session_id'] = 1
    features.loc[features['session_ny'] == 1, 'session_id'] = 2
    features.loc[features['session_overlap'] == 1, 'session_id'] = 3

    # Cyclic encoding of hour (better for neural networks)
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclic) - 5 trading days
    features['day_sin'] = np.sin(2 * np.pi * day / 5)
    features['day_cos'] = np.cos(2 * np.pi * day / 5)

    # Is Friday (different behavior due to weekend positioning)
    features['is_friday'] = (day == 4).astype(float)

    # First/last hour of session (higher volatility)
    features['session_open_hour'] = (
        (hour == 8) |   # London open
        (hour == 13) |  # NY open
        (hour == 23)    # Asian open
    ).astype(float)

    features['session_close_hour'] = (
        (hour == 7) |   # Asian close
        (hour == 16) |  # London close
        (hour == 21)    # NY close
    ).astype(float)

    return features


# ============================================================================
# 4.2 Position State Features (for Environment)
# ============================================================================
def calculate_position_state_features(
    current_position: float,
    entry_price: float,
    current_price: float,
    entry_step: Optional[int],
    current_step: int,
    account_equity: float,
    unrealized_pnl: float,
    current_drawdown: float,
    max_drawdown_limit: float = 0.30,
    max_position_duration: int = 480,  # 480 hours = 20 days in H1
) -> Dict[str, float]:
    """
    Features describing current position state.

    The model NEEDS to know:
    1. Does it have an open position?
    2. Is it winning or losing?
    3. How long has the position been open?
    4. How much accumulated risk does it have?
    """
    features = {}

    # === Current Position ===
    features['current_position'] = current_position
    features['has_position'] = float(abs(current_position) > 0.01)
    features['is_long'] = float(current_position > 0.01)
    features['is_short'] = float(current_position < -0.01)
    features['is_flat'] = float(abs(current_position) <= 0.01)

    # === Position P&L ===
    if abs(current_position) > 0.01 and entry_price > 0:
        if current_position > 0:  # Long
            unrealized_return = (current_price - entry_price) / entry_price
        else:  # Short
            unrealized_return = (entry_price - current_price) / entry_price

        features['unrealized_return'] = unrealized_return
        features['unrealized_pnl_normalized'] = unrealized_pnl / (account_equity + 1e-10)
        features['is_winner'] = float(unrealized_pnl > 0)

        # P&L magnitude (buckets)
        pnl_pct = unrealized_pnl / (account_equity + 1e-10)
        features['pnl_bucket'] = np.clip(pnl_pct * 10, -5, 5)
    else:
        features['unrealized_return'] = 0.0
        features['unrealized_pnl_normalized'] = 0.0
        features['is_winner'] = 0.0
        features['pnl_bucket'] = 0.0

    # === Position Duration ===
    if abs(current_position) > 0.01 and entry_step is not None:
        duration = current_step - entry_step
        features['position_duration_normalized'] = duration / max_position_duration
        features['position_duration_log'] = np.log1p(duration) / np.log1p(max_position_duration)
    else:
        features['position_duration_normalized'] = 0.0
        features['position_duration_log'] = 0.0

    # === Accumulated Risk ===
    features['drawdown_utilization'] = current_drawdown / max_drawdown_limit
    features['risk_budget_remaining'] = 1.0 - features['drawdown_utilization']

    # Drawdown warning level (escalating)
    features['dd_warning_level'] = 0.0
    if features['drawdown_utilization'] > 0.5:
        features['dd_warning_level'] = 1.0
    if features['drawdown_utilization'] > 0.7:
        features['dd_warning_level'] = 2.0
    if features['drawdown_utilization'] > 0.9:
        features['dd_warning_level'] = 3.0

    return features


# ============================================================================
# 4.3 TIPS Yield Features (via yfinance TIP ETF)
# ============================================================================
def fetch_tips_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch TIPS (Treasury Inflation-Protected Securities) data via yfinance.

    Uses TIP ETF as proxy for real interest rates.
    Correlation with gold: -0.82 to -0.93 (very strong negative)
    - TIPS yield up -> Gold down
    - TIPS yield down -> Gold up
    """
    try:
        import yfinance as yf

        # TIP: iShares TIPS Bond ETF (proxy for real rates)
        tip = yf.download('TIP', start=start_date, end=end_date, progress=False)

        if tip.empty:
            return None

        return tip
    except Exception as e:
        print(f"Error fetching TIPS data: {e}")
        return None


def calculate_tips_features(tips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features from TIPS ETF data.
    """
    features = pd.DataFrame(index=tips_df.index)

    close = tips_df['Close'] if 'Close' in tips_df.columns else tips_df['Adj Close']

    # Price-based proxy for yield (inverse relationship)
    # When TIP price goes up, yields go down
    features['tips_return'] = close.pct_change().fillna(0)
    features['tips_return_5d'] = close.pct_change(5).fillna(0)
    features['tips_return_20d'] = close.pct_change(20).fillna(0)

    # Yield proxy (inverted returns)
    features['tips_yield_proxy'] = -features['tips_return_20d']

    # Normalized level
    features['tips_normalized'] = (close - close.rolling(252, min_periods=20).mean()) / \
                                   (close.rolling(252, min_periods=20).std() + 1e-10)

    # Momentum
    features['tips_momentum'] = close.pct_change(10) / (close.pct_change(10).rolling(50).std() + 1e-10)

    return features.fillna(0)


# ============================================================================
# 4.5 VIX Features with Regimes (via yfinance ^VIX)
# ============================================================================
def fetch_vix_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch VIX data via yfinance.

    VIX is the "fear index":
    - VIX high -> Market panic -> Flight to safety -> Bullish gold
    - VIX low -> Complacency -> Risk-on -> Neutral/Bearish gold
    """
    try:
        import yfinance as yf

        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)

        if vix.empty:
            return None

        return vix
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return None


def calculate_vix_features(vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced VIX features for gold trading.
    """
    features = pd.DataFrame(index=vix_df.index)

    vix = vix_df['Close'] if 'Close' in vix_df.columns else vix_df['Adj Close']

    # Level
    features['vix_level'] = vix
    features['vix_normalized'] = (vix - vix.rolling(252, min_periods=20).mean()) / \
                                  (vix.rolling(252, min_periods=20).std() + 1e-10)

    # Volatility regime
    features['vix_regime'] = 0  # Normal
    features.loc[vix < 15, 'vix_regime'] = -1       # Very low (complacency)
    features.loc[(vix >= 15) & (vix < 20), 'vix_regime'] = 0  # Normal
    features.loc[(vix >= 20) & (vix < 25), 'vix_regime'] = 1  # Elevated
    features.loc[(vix >= 25) & (vix < 30), 'vix_regime'] = 2  # High
    features.loc[vix >= 30, 'vix_regime'] = 3       # Crisis

    # Spike detection (abrupt change)
    vix_change = vix.pct_change()
    features['vix_spike'] = (vix_change > 0.20).astype(float)  # +20% in one day
    features['vix_crash'] = (vix_change < -0.15).astype(float)  # -15% in one day

    # Mean reversion potential
    features['vix_mean_distance'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-10)

    # Momentum
    features['vix_momentum_5d'] = vix.pct_change(5).fillna(0)
    features['vix_momentum_20d'] = vix.pct_change(20).fillna(0)

    # Binary features
    features['is_high_vix'] = (vix > 25).astype(float)
    features['is_crisis_vix'] = (vix > 30).astype(float)
    features['is_low_vix'] = (vix < 15).astype(float)

    return features.fillna(0)


# ============================================================================
# 4.6 Economic Calendar Features
# ============================================================================
def get_fomc_dates(year: int) -> list:
    """
    Get FOMC meeting dates for a given year.
    These are approximate - actual dates should be verified.
    """
    # Typical FOMC schedule (8 meetings per year)
    # Usually Tuesday-Wednesday of third week of:
    # Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]

    dates = []
    for month in fomc_months:
        # Approximate: third Wednesday of the month
        first_day = datetime(year, month, 1)
        # Find first Wednesday
        days_until_wed = (2 - first_day.weekday()) % 7
        first_wed = first_day + timedelta(days=days_until_wed)
        # Third Wednesday
        third_wed = first_wed + timedelta(weeks=2)
        dates.append(third_wed.date())

    return dates


def get_nfp_dates(year: int) -> list:
    """
    Get NFP (Non-Farm Payrolls) release dates.
    Usually first Friday of each month.
    """
    dates = []
    for month in range(1, 13):
        first_day = datetime(year, month, 1)
        # Find first Friday
        days_until_fri = (4 - first_day.weekday()) % 7
        first_fri = first_day + timedelta(days=days_until_fri)
        dates.append(first_fri.date())

    return dates


def calculate_calendar_features(timestamp: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate features based on economic calendar.
    """
    features = pd.DataFrame(index=timestamp)

    # Get unique years in the data
    years = timestamp.year.unique()

    # Collect all important dates
    fomc_dates = []
    nfp_dates = []
    for year in years:
        fomc_dates.extend(get_fomc_dates(year))
        nfp_dates.extend(get_nfp_dates(year))

    fomc_dates = set(fomc_dates)
    nfp_dates = set(nfp_dates)

    # Convert timestamp to date for comparison
    dates = pd.Series(timestamp.date, index=timestamp)

    # Is FOMC day
    features['is_fomc_day'] = dates.apply(lambda x: x in fomc_dates).astype(float)

    # Is NFP day
    features['is_nfp_day'] = dates.apply(lambda x: x in nfp_dates).astype(float)

    # Days to next FOMC
    def days_to_next_event(date, event_dates):
        future_dates = [d for d in event_dates if d > date]
        if future_dates:
            return (min(future_dates) - date).days
        return 30  # Default if no future dates

    features['days_to_fomc'] = dates.apply(lambda x: days_to_next_event(x, fomc_dates))
    features['days_to_fomc'] = features['days_to_fomc'].clip(0, 30) / 30  # Normalize

    features['days_to_nfp'] = dates.apply(lambda x: days_to_next_event(x, nfp_dates))
    features['days_to_nfp'] = features['days_to_nfp'].clip(0, 30) / 30  # Normalize

    # High impact window (day before and day of)
    features['is_high_impact_day'] = (
        features['is_fomc_day'] | features['is_nfp_day']
    ).astype(float)

    # Pre-event caution (1-2 days before major event)
    features['pre_fomc'] = (features['days_to_fomc'] * 30 <= 2).astype(float)
    features['pre_nfp'] = (features['days_to_nfp'] * 30 <= 1).astype(float)

    # Month end effect (last 3 days of month)
    features['is_month_end'] = (timestamp.day >= 28).astype(float)

    # Quarter end effect
    features['is_quarter_end'] = (
        (timestamp.month.isin([3, 6, 9, 12])) & (timestamp.day >= 28)
    ).astype(float)

    return features


# ============================================================================
# 4.7 Price Action Features
# ============================================================================
def calculate_price_action_features(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Calculate price action features (candlestick patterns and structure).
    """
    features = pd.DataFrame(index=close.index)

    # === Candle Body Features ===
    body = close - open_
    body_size = abs(body)
    candle_range = high - low

    features['body_ratio'] = body_size / (candle_range + 1e-10)
    features['is_bullish_candle'] = (body > 0).astype(float)
    features['is_doji'] = (body_size < candle_range * 0.1).astype(float)

    # Upper/Lower shadow
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low

    features['upper_shadow_ratio'] = upper_shadow / (candle_range + 1e-10)
    features['lower_shadow_ratio'] = lower_shadow / (candle_range + 1e-10)

    # === Support/Resistance ===
    # Distance to recent swing high/low
    rolling_high_20 = high.rolling(20).max()
    rolling_low_20 = low.rolling(20).min()
    atr = (high - low).rolling(14).mean()

    features['distance_to_resistance'] = (rolling_high_20 - close) / (atr + 1e-10)
    features['distance_to_support'] = (close - rolling_low_20) / (atr + 1e-10)

    # === Momentum Patterns ===
    # Higher highs, higher lows (uptrend)
    hh = high > high.shift(1)
    hl = low > low.shift(1)
    features['uptrend_structure'] = (hh & hl).rolling(3).mean().fillna(0)

    # Lower highs, lower lows (downtrend)
    lh = high < high.shift(1)
    ll = low < low.shift(1)
    features['downtrend_structure'] = (lh & ll).rolling(3).mean().fillna(0)

    # === Range Analysis ===
    features['range_percentile'] = candle_range.rolling(20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) > 0 else 0.5
    ).fillna(0.5)

    # Wide range bar (breakout potential)
    features['is_wide_range_bar'] = (
        candle_range > candle_range.rolling(20).quantile(0.9)
    ).astype(float).fillna(0)

    # Narrow range bar (consolidation)
    features['is_narrow_range_bar'] = (
        candle_range < candle_range.rolling(20).quantile(0.1)
    ).astype(float).fillna(0)

    # === Candle Patterns ===
    # Hammer (long lower shadow, small body at top)
    features['is_hammer'] = (
        (lower_shadow > body_size * 2) &
        (upper_shadow < body_size * 0.5) &
        (body > 0)
    ).astype(float)

    # Shooting star (long upper shadow, small body at bottom)
    features['is_shooting_star'] = (
        (upper_shadow > body_size * 2) &
        (lower_shadow < body_size * 0.5) &
        (body < 0)
    ).astype(float)

    # Engulfing patterns
    features['bullish_engulfing'] = (
        (body.shift(1) < 0) &  # Previous bearish
        (body > 0) &  # Current bullish
        (close > open_.shift(1)) &  # Current close > prev open
        (open_ < close.shift(1))  # Current open < prev close
    ).astype(float)

    features['bearish_engulfing'] = (
        (body.shift(1) > 0) &  # Previous bullish
        (body < 0) &  # Current bearish
        (close < open_.shift(1)) &  # Current close < prev open
        (open_ > close.shift(1))  # Current open > prev close
    ).astype(float)

    return features.fillna(0)


# ============================================================================
# 4.8 Multi-Timeframe Context Features
# ============================================================================
def calculate_mtf_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    base_timeframe: str = 'H1',
) -> pd.DataFrame:
    """
    Calculate multi-timeframe context features.

    Uses the base timeframe data to simulate higher timeframes.
    """
    features = pd.DataFrame(index=close.index)

    # === Trend Alignment Across Timeframes ===
    # EMA 21 at different effective timeframes
    ema21_h1 = close.ewm(span=21).mean()
    ema21_h4 = close.ewm(span=21*4).mean()  # Simulate H4
    ema21_d1 = close.ewm(span=21*24).mean()  # Simulate D1

    # Price vs EMA at each timeframe
    features['h1_above_ema21'] = (close > ema21_h1).astype(float)
    features['h4_above_ema21'] = (close > ema21_h4).astype(float)
    features['d1_above_ema21'] = (close > ema21_d1).astype(float)

    # Trend alignment score (-3 to +3)
    features['trend_alignment'] = (
        features['h1_above_ema21'] +
        features['h4_above_ema21'] +
        features['d1_above_ema21']
    ) * 2 - 3

    # === Higher Timeframe RSI ===
    def rsi(price, period=21):
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - 100 / (1 + gain / (loss + 1e-10))

    # H4 RSI (using 4x period)
    features['h4_rsi'] = (rsi(close, period=21*4) - 50) / 50

    # D1 RSI (using 24x period for hourly data)
    features['d1_rsi'] = (rsi(close, period=21*24) - 50) / 50

    # === Higher Timeframe Volatility ===
    # ATR at different timeframes
    tr_h1 = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    atr_h1 = tr_h1.rolling(14).mean()
    atr_h4 = tr_h1.rolling(14*4).mean()
    atr_d1 = tr_h1.rolling(14*24).mean()

    features['atr_h1_normalized'] = atr_h1 / close
    features['atr_h4_normalized'] = atr_h4 / close
    features['atr_d1_normalized'] = atr_d1 / close

    # Volatility contraction/expansion across timeframes
    features['vol_h1_vs_d1'] = (atr_h1 / (atr_d1 + 1e-10)).fillna(1)

    # === Support/Resistance from Higher Timeframes ===
    # D1 high/low (using 24-hour rolling)
    d1_high = high.rolling(24).max()
    d1_low = low.rolling(24).min()

    features['above_d1_high'] = (close > d1_high.shift(1)).astype(float)
    features['below_d1_low'] = (close < d1_low.shift(1)).astype(float)

    # Weekly high/low (using 120-hour rolling for 5 trading days)
    w1_high = high.rolling(120).max()
    w1_low = low.rolling(120).min()

    features['above_w1_high'] = (close > w1_high.shift(1)).astype(float)
    features['below_w1_low'] = (close < w1_low.shift(1)).astype(float)

    # Distance from weekly range
    w1_range = w1_high - w1_low
    features['position_in_weekly_range'] = (
        (close - w1_low) / (w1_range + 1e-10)
    ).fillna(0.5)

    return features.fillna(0)


# ============================================================================
# Main Function: Calculate All Advanced Features
# ============================================================================
def calculate_all_advanced_features(
    df: pd.DataFrame,
    include_external_data: bool = True,
) -> pd.DataFrame:
    """
    Calculate all advanced features from Section 4.

    Args:
        df: DataFrame with OHLC data and timestamp index
        include_external_data: If True, try to fetch VIX and TIPS data

    Returns:
        DataFrame with all advanced features
    """
    features = pd.DataFrame(index=df.index)

    # 4.1 Session Features
    if isinstance(df.index, pd.DatetimeIndex):
        session_features = calculate_session_features(df.index)
        for col in session_features.columns:
            features[col] = session_features[col]

    # 4.6 Calendar Features
    if isinstance(df.index, pd.DatetimeIndex):
        calendar_features = calculate_calendar_features(df.index)
        for col in calendar_features.columns:
            features[col] = calendar_features[col]

    # 4.7 Price Action Features
    price_action_features = calculate_price_action_features(
        df['open'], df['high'], df['low'], df['close']
    )
    for col in price_action_features.columns:
        features[col] = price_action_features[col]

    # 4.8 MTF Features
    mtf_features = calculate_mtf_features(
        df['close'], df['high'], df['low']
    )
    for col in mtf_features.columns:
        features[col] = mtf_features[col]

    # 4.3 & 4.5 External Data (VIX, TIPS)
    if include_external_data and isinstance(df.index, pd.DatetimeIndex):
        try:
            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')

            # Fetch VIX
            vix_df = fetch_vix_data(start_date, end_date)
            if vix_df is not None and not vix_df.empty:
                vix_features = calculate_vix_features(vix_df)
                # Resample to match main data frequency
                vix_features = vix_features.reindex(df.index, method='ffill')
                for col in vix_features.columns:
                    features[f'vix_{col}' if not col.startswith('vix') else col] = vix_features[col]

            # Fetch TIPS
            tips_df = fetch_tips_data(start_date, end_date)
            if tips_df is not None and not tips_df.empty:
                tips_features = calculate_tips_features(tips_df)
                # Resample to match main data frequency
                tips_features = tips_features.reindex(df.index, method='ffill')
                for col in tips_features.columns:
                    features[col] = tips_features[col]

        except Exception as e:
            print(f"Warning: Could not fetch external data: {e}")

    return features.fillna(0)
