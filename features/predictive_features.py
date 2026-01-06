"""
PREDICTIVE FEATURES - Features designed to distinguish winners from losers

This replaces the old 152-feature system with features that actually predict.

Key changes:
1. Signal-based features (crossovers, breakouts) instead of continuous values
2. Multi-timeframe confirmation (higher TF signals = stronger)
3. Regime features (trend vs range, high vs low volatility)
4. Cross-asset leading indicators (DXY moves first, gold follows)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_tz_naive(index):
    """Convert DatetimeIndex to timezone-naive."""
    if hasattr(index, 'tz') and index.tz is not None:
        return index.tz_convert('UTC').tz_localize(None)
    return index


def load_ohlcv(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    return df


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market regime features.

    Regimes are CRITICAL for distinguishing winners from losers:
    - In trending markets: trend-following wins
    - In ranging markets: mean-reversion wins
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']

    # === TREND VS RANGE REGIME ===

    # ADX for trend strength
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()

    plus_dm = ((high - high.shift(1)) > (low.shift(1) - low)) * (high - high.shift(1))
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = ((low.shift(1) - low) > (high - high.shift(1))) * (low.shift(1) - low)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(14).mean()

    # Trend regime: ADX > 25 = trending
    result['regime_trending'] = (adx > 25).astype(float)
    result['regime_strong_trend'] = (adx > 40).astype(float)
    result['regime_ranging'] = (adx < 20).astype(float)

    # Trend direction in trending regime
    result['regime_bullish_trend'] = ((adx > 25) & (plus_di > minus_di)).astype(float)
    result['regime_bearish_trend'] = ((adx > 25) & (minus_di > plus_di)).astype(float)

    # === VOLATILITY REGIME ===

    ret = close.pct_change()
    vol = ret.rolling(20).std()
    vol_ma = vol.rolling(50).mean()

    result['regime_high_volatility'] = (vol > vol_ma * 1.5).astype(float)
    result['regime_low_volatility'] = (vol < vol_ma * 0.5).astype(float)
    result['regime_volatility_expanding'] = (vol > vol.shift(5)).astype(float)

    # === MOMENTUM REGIME ===

    mom_20 = close.pct_change(20)
    mom_50 = close.pct_change(50)

    result['regime_bullish_momentum'] = ((mom_20 > 0) & (mom_50 > 0)).astype(float)
    result['regime_bearish_momentum'] = ((mom_20 < 0) & (mom_50 < 0)).astype(float)
    result['regime_momentum_divergence'] = ((mom_20 > 0) != (mom_50 > 0)).astype(float)

    return result.fillna(0)


def compute_cross_asset_signals(df_gold: pd.DataFrame, macro_dict: dict) -> pd.DataFrame:
    """
    Compute cross-asset LEADING signals.

    Key insight: DXY and yields often move BEFORE gold.
    """
    result = pd.DataFrame(index=df_gold.index)
    gold_index = make_tz_naive(df_gold.index)

    # === DXY SIGNALS ===
    if 'dxy' in macro_dict:
        dxy = macro_dict['dxy']
        dxy.index = make_tz_naive(dxy.index)
        dxy_aligned = dxy.reindex(gold_index, method='ffill')

        # DXY momentum (leads gold inversely)
        dxy_mom = dxy_aligned.pct_change(5)
        result['dxy_weakening'] = (dxy_mom < -0.005).astype(float)  # DXY down = gold up
        result['dxy_strengthening'] = (dxy_mom > 0.005).astype(float)

        # DXY trend
        dxy_ma20 = dxy_aligned.rolling(20).mean()
        result['dxy_below_ma'] = (dxy_aligned < dxy_ma20).astype(float)

        # DXY reversal signal
        dxy_high = dxy_aligned.rolling(20).max()
        dxy_low = dxy_aligned.rolling(20).min()
        result['dxy_at_resistance'] = (dxy_aligned >= dxy_high * 0.998).astype(float)
        result['dxy_at_support'] = (dxy_aligned <= dxy_low * 1.002).astype(float)

    # === VIX SIGNALS ===
    if 'vix' in macro_dict:
        vix = macro_dict['vix']
        vix.index = make_tz_naive(vix.index)
        vix_aligned = vix.reindex(gold_index, method='ffill')

        # VIX levels (fear = gold up)
        result['vix_elevated'] = (vix_aligned > 20).astype(float)
        result['vix_extreme'] = (vix_aligned > 30).astype(float)
        result['vix_complacent'] = (vix_aligned < 15).astype(float)

        # VIX spike (sudden fear)
        vix_change = vix_aligned.pct_change(5)
        result['vix_spiking'] = (vix_change > 0.2).astype(float)

    # === YIELDS SIGNALS ===
    if 'us10y' in macro_dict:
        us10y = macro_dict['us10y']
        us10y.index = make_tz_naive(us10y.index)
        us10y_aligned = us10y.reindex(gold_index, method='ffill')

        # Yield direction (yields down = gold up)
        yield_mom = us10y_aligned.diff(5)
        result['yields_falling'] = (yield_mom < -0.05).astype(float)
        result['yields_rising'] = (yield_mom > 0.05).astype(float)

    # === SPX SIGNALS ===
    if 'spx' in macro_dict:
        spx = macro_dict['spx']
        spx.index = make_tz_naive(spx.index)
        spx_aligned = spx.reindex(gold_index, method='ffill')

        # Risk-off signal (SPX down = gold safe haven)
        spx_ret = spx_aligned.pct_change(5)
        result['risk_off'] = (spx_ret < -0.02).astype(float)
        result['risk_on'] = (spx_ret > 0.02).astype(float)

    return result.fillna(0)


def compute_session_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trading session signals.

    Gold has specific session patterns that affect winners/losers.
    """
    result = pd.DataFrame(index=df.index)

    # Extract hour (assuming UTC)
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour

    # Session definitions (UTC)
    # Asian: 23:00-07:00 UTC
    # London: 07:00-16:00 UTC
    # NY: 12:00-21:00 UTC
    # London-NY overlap: 12:00-16:00 UTC (best liquidity)

    result['session_asian'] = ((hour >= 23) | (hour < 7)).astype(float)
    result['session_london'] = ((hour >= 7) & (hour < 16)).astype(float)
    result['session_ny'] = ((hour >= 12) & (hour < 21)).astype(float)
    result['session_overlap'] = ((hour >= 12) & (hour < 16)).astype(float)

    # Avoid low liquidity periods
    result['low_liquidity'] = ((hour >= 21) & (hour < 23)).astype(float)

    # Session open signals (first hour of session often sets direction)
    result['london_open'] = ((hour >= 7) & (hour < 8)).astype(float)
    result['ny_open'] = ((hour >= 12) & (hour < 13)).astype(float)

    return result.fillna(0)


def compute_entry_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features that indicate trade ENTRY quality.

    Winners tend to have better entries (near support, after pullback, etc.)
    """
    result = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']

    # === PULLBACK FEATURES ===
    # Entry after pullback in uptrend is higher quality

    # Trend (MA50)
    ma50 = close.rolling(50).mean()
    in_uptrend = (close > ma50)

    # Pullback: price moved down recently
    recent_high = high.rolling(10).max()
    pullback_pct = (recent_high - close) / recent_high

    # Quality entry: uptrend + pullback to support
    result['quality_entry_pullback'] = (in_uptrend & (pullback_pct > 0.005) & (pullback_pct < 0.02)).astype(float)

    # === SUPPORT/RESISTANCE PROXIMITY ===

    # Distance to recent low (potential support)
    low_20 = low.rolling(20).min()
    dist_to_support = (close - low_20) / close
    result['near_support'] = (dist_to_support < 0.005).astype(float)

    # Distance to recent high (potential resistance)
    high_20 = high.rolling(20).max()
    dist_to_resistance = (high_20 - close) / close
    result['near_resistance'] = (dist_to_resistance < 0.005).astype(float)

    # === CONSOLIDATION BREAKOUT ===
    # Tight range then breakout = quality entry

    # Range of last 20 bars
    range_20 = (high.rolling(20).max() - low.rolling(20).min()) / close
    range_5 = (high.rolling(5).max() - low.rolling(5).min()) / close

    # Consolidation: tight recent range
    consolidation = (range_5 < range_20 * 0.5)

    # Breakout after consolidation
    breakout = (close > high.rolling(5).max().shift(1))
    result['consolidation_breakout'] = (consolidation & breakout).astype(float)

    # === VOLUME CONFIRMATION AT ENTRY ===

    volume = df['volume']
    avg_vol = volume.rolling(20).mean()

    # High volume on move = confirmation
    result['volume_confirmed_up'] = ((close > close.shift(1)) & (volume > avg_vol * 1.5)).astype(float)

    return result.fillna(0)


def make_predictive_features(base_timeframe: str = 'M5', data_dir: str = 'data') -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create complete predictive feature set.

    This replaces make_ultimate_features() with signal-based features.

    Returns:
        features (ndarray): Shape (N, ~100), dtype float32
        returns (ndarray): Shape (N,), target returns
        timestamps (DatetimeIndex): Shape (N,)
    """
    logger.info("="*70)
    logger.info("PREDICTIVE FEATURE SYSTEM")
    logger.info("="*70)

    data_dir = Path(data_dir)

    # === STEP 1: LOAD BASE DATA ===
    logger.info("\n1. Loading base data...")

    base_file = data_dir / f'xauusd_{base_timeframe.lower()}.csv'
    df_base = load_ohlcv(base_file)
    logger.info(f"   Loaded {len(df_base):,} bars")

    base_index = make_tz_naive(df_base.index)

    # === STEP 2: LOAD ALL TIMEFRAMES FOR SIGNALS ===
    logger.info("\n2. Loading timeframe data...")

    tf_files = {
        'M5': 'xauusd_m5.csv',
        'M15': 'xauusd_m15.csv',
        'H1': 'xauusd_h1.csv',
        'H4': 'xauusd_h4.csv',
        'D1': 'xauusd_d1.csv',
    }

    tf_data = {}
    for tf, filename in tf_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            tf_data[tf] = load_ohlcv(filepath)
            logger.info(f"   {tf}: {len(tf_data[tf]):,} bars")

    # === STEP 3: COMPUTE SIGNAL FEATURES ===
    logger.info("\n3. Computing signal features...")

    from features.signal_features import compute_all_signal_features

    all_features = []

    # Signals from each timeframe
    for tf in ['M5', 'H1', 'H4', 'D1']:
        if tf in tf_data:
            df_tf = tf_data[tf].copy()
            df_tf.index = make_tz_naive(df_tf.index)
            signals = compute_all_signal_features(df_tf, prefix=f'{tf}_')

            # Align to base timeframe
            signals.index = make_tz_naive(signals.index)
            signals_aligned = signals.reindex(base_index, method='ffill')
            signals_aligned = signals_aligned.fillna(0)
            all_features.append(signals_aligned)

    # === STEP 4: COMPUTE REGIME FEATURES ===
    logger.info("\n4. Computing regime features...")

    # Use H1 for regime detection (smoother)
    if 'H1' in tf_data:
        regime_df = tf_data['H1'].copy()
    else:
        regime_df = df_base.copy()

    regime_df.index = make_tz_naive(regime_df.index)
    regime_features = compute_regime_features(regime_df)
    regime_features.index = make_tz_naive(regime_features.index)
    regime_aligned = regime_features.reindex(base_index, method='ffill')
    regime_aligned = regime_aligned.fillna(0)
    all_features.append(regime_aligned)

    logger.info(f"   Regime features: {regime_aligned.shape[1]}")

    # === STEP 5: COMPUTE CROSS-ASSET SIGNALS ===
    logger.info("\n5. Computing cross-asset signals...")

    # Load macro data
    macro_files = {
        'dxy': 'dxy_daily.csv',
        'vix': 'vix_daily.csv',
        'us10y': 'us10y_daily.csv',
        'spx': 'spx_daily.csv',
    }

    macro_dict = {}
    for name, filename in macro_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            df_macro = pd.read_csv(filepath)
            df_macro['time'] = pd.to_datetime(df_macro['time'], utc=True).dt.tz_localize(None)
            df_macro = df_macro.set_index('time').sort_index()

            if 'close' in df_macro.columns:
                macro_dict[name] = df_macro['close']
            elif 'Close' in df_macro.columns:
                macro_dict[name] = df_macro['Close']

            logger.info(f"   {name}: {len(df_macro):,} bars")

    df_base_naive = df_base.copy()
    df_base_naive.index = make_tz_naive(df_base_naive.index)
    cross_asset = compute_cross_asset_signals(df_base_naive, macro_dict)
    cross_asset.index = make_tz_naive(cross_asset.index)
    cross_asset = cross_asset.reindex(base_index, method='ffill').fillna(0)
    all_features.append(cross_asset)

    logger.info(f"   Cross-asset features: {cross_asset.shape[1]}")

    # === STEP 6: COMPUTE SESSION SIGNALS ===
    logger.info("\n6. Computing session signals...")

    session_features = compute_session_signals(df_base_naive)
    session_features.index = base_index
    session_features = session_features.fillna(0)
    all_features.append(session_features)

    logger.info(f"   Session features: {session_features.shape[1]}")

    # === STEP 7: COMPUTE ENTRY QUALITY FEATURES ===
    logger.info("\n7. Computing entry quality features...")

    entry_features = compute_entry_quality_features(df_base_naive)
    entry_features.index = base_index
    entry_features = entry_features.fillna(0)
    all_features.append(entry_features)

    logger.info(f"   Entry quality features: {entry_features.shape[1]}")

    # === STEP 8: COMBINE ALL FEATURES ===
    logger.info("\n8. Combining all features...")

    combined = pd.concat(all_features, axis=1)
    combined = combined.fillna(0)

    # Remove duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # === STEP 9: COMPUTE RETURNS ===
    logger.info("\n9. Computing returns...")

    returns = df_base['close'].pct_change().fillna(0).values.astype(np.float32)

    # === FINAL SUMMARY ===
    logger.info("\n" + "="*70)
    logger.info("PREDICTIVE FEATURES CREATED!")
    logger.info("="*70)

    logger.info(f"\nTotal features: {combined.shape[1]}")
    logger.info(f"Total samples: {len(combined):,}")
    logger.info(f"Date range: {base_index[0]} to {base_index[-1]}")

    # Count signal vs continuous features
    binary_cols = [c for c in combined.columns if combined[c].isin([0, 1]).all()]
    logger.info(f"\nBinary signal features: {len(binary_cols)}")
    logger.info(f"Continuous features: {combined.shape[1] - len(binary_cols)}")

    X = combined.values.astype(np.float32)

    return X, returns, combined.index


if __name__ == "__main__":
    # Test
    logger.info("Testing predictive features...")

    X, returns, timestamps = make_predictive_features(base_timeframe='M5')

    logger.info(f"\nFeatures shape: {X.shape}")
    logger.info(f"Returns shape: {returns.shape}")
    logger.info(f"Timestamps: {len(timestamps)}")

    # Check for NaNs/Infs
    logger.info(f"\nNaN count: {np.isnan(X).sum()}")
    logger.info(f"Inf count: {np.isinf(X).sum()}")
