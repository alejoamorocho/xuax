# features/make_features.py
"""
Feature Engineering for Gold Trading

Uses gold-optimized technical indicators:
- RSI(21) with 75/25 thresholds
- MACD(16,34,13)
- ATR-based volatility regime
- Bollinger Bands(13)

Advanced Features (Section 4):
- Session features (trading hours)
- Calendar features (FOMC, NFP)
- Price action features (candlestick patterns)
- Multi-timeframe context features
- VIX and TIPS features (via yfinance)
"""

import numpy as np
import pandas as pd
from data.load_data import load_ohlc_csv
from features.gold_indicators import (
    calculate_gold_rsi,
    get_rsi_signals,
    get_macd_signals,
    get_atr_features,
    get_bollinger_features,
    calculate_all_gold_indicators,
)
from features.advanced_features import (
    calculate_session_features,
    calculate_calendar_features,
    calculate_price_action_features,
    calculate_mtf_features,
    calculate_all_advanced_features,
)


def compute_features(
    df: pd.DataFrame,
    use_optimized_indicators: bool = True,
    use_advanced_features: bool = True,
    include_external_data: bool = False,
) -> tuple:
    """
    Compute features for gold trading.

    Args:
        df: DataFrame with OHLC data
        use_optimized_indicators: If True, use gold-optimized indicators
        use_advanced_features: If True, include session, calendar, price action, MTF features
        include_external_data: If True, try to fetch VIX and TIPS data from yfinance

    Returns:
        df, features, returns
    """
    df = df.copy()

    # =========================================================================
    # 1. Basic Price Features
    # =========================================================================
    df["ret"] = np.log(df["close"]).diff().fillna(0.0)
    df["vol"] = df["ret"].rolling(24).std().fillna(0.0)
    df["mom"] = df["close"].pct_change(24).fillna(0.0)

    # Moving Averages
    df["ma_fast"] = df["close"].rolling(24).mean()
    df["ma_slow"] = df["close"].rolling(120).mean()
    df["ma_diff"] = ((df["ma_fast"] - df["ma_slow"]) / df["close"]).fillna(0.0)

    # =========================================================================
    # 2. Technical Indicators (Optimized for Gold)
    # =========================================================================
    if use_optimized_indicators:
        # 3.1 RSI Optimizado (21, umbrales 75/25)
        rsi = calculate_gold_rsi(df["close"], period=21)
        df["rsi"] = rsi / 100.0  # Normalizado [0, 1]

        rsi_signals = get_rsi_signals(rsi, overbought=75, oversold=25)
        df["rsi_normalized"] = rsi_signals["rsi_normalized"]
        df["rsi_oversold"] = rsi_signals["rsi_oversold"]
        df["rsi_overbought"] = rsi_signals["rsi_overbought"]

        # 3.2 MACD Optimizado (16/34/13)
        macd_signals = get_macd_signals(
            df["close"],
            fast_period=16,
            slow_period=34,
            signal_period=13
        )
        df["macd_diff"] = macd_signals["macd_normalized"]
        df["macd_histogram"] = macd_signals["macd_histogram"]
        df["macd_cross_up"] = macd_signals["macd_cross_up"]
        df["macd_cross_down"] = macd_signals["macd_cross_down"]
        df["macd_above_zero"] = macd_signals["macd_above_zero"]

        # 3.3 ATR Features
        atr_features = get_atr_features(
            df["high"],
            df["low"],
            df["close"],
            period=14
        )
        df["atr_normalized"] = atr_features["atr_normalized"]
        df["atr_regime"] = atr_features["atr_regime"]
        df["atr_percentile"] = atr_features["atr_percentile"]
        df["atr_expanding"] = atr_features["atr_expanding"]

        # 3.4 Bollinger Bands Optimizados (13)
        bb_features = get_bollinger_features(df["close"], period=13, num_std=2.0)
        df["bb_percent_b"] = bb_features["bb_percent_b"]
        df["bb_bandwidth"] = bb_features["bb_bandwidth"]
        df["bb_squeeze"] = bb_features["bb_squeeze"]
        df["bb_position"] = bb_features["bb_position"]
        df["bb_overbought"] = bb_features["bb_overbought"]
        df["bb_oversold"] = bb_features["bb_oversold"]

    else:
        # Legacy indicators (for backward compatibility)
        # RSI Standard (14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = (100 - (100 / (1 + rs))) / 100.0
        df["rsi"] = df["rsi"].fillna(0.5)

        # MACD Standard (12/26/9)
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd_diff"] = ((macd - signal) / df["close"]).fillna(0.0)

    # =========================================================================
    # 3. Macro Features (if available)
    # =========================================================================
    if "dxy_close" in df.columns:
        # DXY Returns
        df["dxy_ret"] = np.log(df["dxy_close"]).diff().fillna(0.0)
        # SPX Returns
        df["spx_ret"] = np.log(df["spx_close"]).diff().fillna(0.0)
        # US10Y Change
        df["us10y_chg"] = df["us10y_close"].diff().fillna(0.0)

        # Correlations
        df["corr_dxy"] = df["ret"].rolling(24).corr(df["dxy_ret"]).fillna(0.0)
        df["corr_spx"] = df["ret"].rolling(24).corr(df["spx_ret"]).fillna(0.0)

    # =========================================================================
    # 4. Advanced Features (Section 4 of Checklist)
    # =========================================================================
    advanced_feature_cols = []

    if use_advanced_features:
        # Check if index is datetime for session/calendar features
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

        # 4.1 Session Features
        if has_datetime_index:
            try:
                session_feats = calculate_session_features(df.index)
                for col in session_feats.columns:
                    df[col] = session_feats[col].values
                    advanced_feature_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not calculate session features: {e}")

        # 4.6 Calendar Features
        if has_datetime_index:
            try:
                calendar_feats = calculate_calendar_features(df.index)
                for col in calendar_feats.columns:
                    df[col] = calendar_feats[col].values
                    advanced_feature_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not calculate calendar features: {e}")

        # 4.7 Price Action Features
        try:
            price_action_feats = calculate_price_action_features(
                df["open"], df["high"], df["low"], df["close"]
            )
            for col in price_action_feats.columns:
                df[col] = price_action_feats[col].values
                advanced_feature_cols.append(col)
        except Exception as e:
            print(f"Warning: Could not calculate price action features: {e}")

        # 4.8 MTF Context Features
        try:
            mtf_feats = calculate_mtf_features(df["close"], df["high"], df["low"])
            for col in mtf_feats.columns:
                df[col] = mtf_feats[col].values
                advanced_feature_cols.append(col)
        except Exception as e:
            print(f"Warning: Could not calculate MTF features: {e}")

        # 4.3 & 4.5 External Data (VIX, TIPS) - optional
        if include_external_data and has_datetime_index:
            try:
                external_feats = calculate_all_advanced_features(
                    df, include_external_data=True
                )
                # Only add VIX and TIPS columns (others already added)
                vix_tips_cols = [c for c in external_feats.columns
                                 if c.startswith('vix_') or c.startswith('tips_')]
                for col in vix_tips_cols:
                    if col not in df.columns:
                        df[col] = external_feats[col].values
                        advanced_feature_cols.append(col)
            except Exception as e:
                print(f"Warning: Could not fetch external data: {e}")

    # =========================================================================
    # 5. Build Feature Matrix
    # =========================================================================
    if use_optimized_indicators:
        # Full feature set with optimized indicators
        feature_cols = [
            # Basic
            "ret", "vol", "mom", "ma_diff",
            # RSI optimizado (21, 75/25)
            "rsi", "rsi_normalized", "rsi_oversold", "rsi_overbought",
            # MACD optimizado (16/34/13)
            "macd_diff", "macd_histogram", "macd_cross_up", "macd_cross_down", "macd_above_zero",
            # ATR features
            "atr_normalized", "atr_regime", "atr_percentile", "atr_expanding",
            # Bollinger Bands (13)
            "bb_percent_b", "bb_bandwidth", "bb_squeeze", "bb_position", "bb_overbought", "bb_oversold",
        ]
    else:
        # Legacy feature set
        feature_cols = ["ret", "vol", "mom", "ma_diff", "rsi", "macd_diff"]

    # Add macro features if available
    if "dxy_close" in df.columns:
        macro_cols = ["dxy_ret", "spx_ret", "us10y_chg", "corr_dxy", "corr_spx"]
        feature_cols.extend(macro_cols)

    # Add advanced features if enabled
    if use_advanced_features and advanced_feature_cols:
        # Only add columns that exist in df
        valid_advanced_cols = [c for c in advanced_feature_cols if c in df.columns]
        feature_cols.extend(valid_advanced_cols)

    # Clean data - skip warmup period
    warmup = 120
    df = df.iloc[warmup:].reset_index(drop=True)

    # Extract features and returns
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    rets = df["ret"].to_numpy(dtype=np.float32)

    # Force cleanup
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    mu = feats.mean(axis=0, keepdims=True)
    sig = feats.std(axis=0, keepdims=True) + 1e-8
    feats = (feats - mu) / sig

    return df, feats, rets


def make_features(
    csv_path: str,
    window: int = 64,
    use_optimized_indicators: bool = True,
    use_advanced_features: bool = True,
    include_external_data: bool = False,
):
    """
    Load data and compute features.

    Args:
        csv_path: Path to OHLC CSV file
        window: Lookback window (not used in computation, for compatibility)
        use_optimized_indicators: If True, use gold-optimized indicators
        use_advanced_features: If True, include session, calendar, price action, MTF features
        include_external_data: If True, try to fetch VIX and TIPS data from yfinance

    Returns:
        df, features, returns
    """
    df = load_ohlc_csv(csv_path)
    return compute_features(
        df,
        use_optimized_indicators=use_optimized_indicators,
        use_advanced_features=use_advanced_features,
        include_external_data=include_external_data,
    )


# ============================================================================
# Legacy function for backward compatibility
# ============================================================================
def compute_rsi(series: pd.Series, period: int = 21) -> pd.Series:
    """Compute RSI using gold-optimized period (21)."""
    return calculate_gold_rsi(series, period=period)
