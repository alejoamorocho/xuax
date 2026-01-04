# features/normalizers.py
"""
Feature Normalization for Trading Models

Section 5.3 of PPO Gold Trading Improvements Checklist:
- Rolling z-score normalization (no lookahead bias)
- Rolling min-max normalization
- Percentile ranking normalization

IMPORTANT: Never use global normalization (mean/std of entire dataset)
as it causes lookahead bias. Always use rolling window normalization.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


class FeatureNormalizer:
    """
    Rolling normalization for trading features.

    CRITICAL: Global normalization (mean/std of entire dataset) causes lookahead bias.
    Use rolling z-score with 252-period window (1 year in daily data).
    """

    def __init__(self, lookback: int = 252, min_periods: int = 20):
        """
        Initialize the normalizer.

        Args:
            lookback: Rolling window size for normalization
            min_periods: Minimum periods required for calculation
        """
        self.lookback = lookback
        self.min_periods = min_periods

    def normalize_zscore(self, series: pd.Series) -> pd.Series:
        """
        Z-score rolling normalization.

        Returns values with mean=0, std=1 using rolling window.
        Best for: returns, MACD, price changes.

        Args:
            series: Input series

        Returns:
            Normalized series with rolling z-score
        """
        mean = series.rolling(self.lookback, min_periods=self.min_periods).mean()
        std = series.rolling(self.lookback, min_periods=self.min_periods).std()
        return (series - mean) / (std + 1e-10)

    def normalize_minmax(self, series: pd.Series) -> pd.Series:
        """
        Min-max rolling normalization to [0, 1].

        Best for: bounded indicators, volume.

        Args:
            series: Input series

        Returns:
            Normalized series in range [0, 1]
        """
        min_val = series.rolling(self.lookback, min_periods=self.min_periods).min()
        max_val = series.rolling(self.lookback, min_periods=self.min_periods).max()
        return (series - min_val) / (max_val - min_val + 1e-10)

    def normalize_percentile(self, series: pd.Series) -> pd.Series:
        """
        Percentile ranking rolling [0, 1].

        Best for: volatility, ATR, comparing relative values.

        Args:
            series: Input series

        Returns:
            Normalized series with percentile rank
        """
        def pct_rank(x):
            if len(x) < 2:
                return 0.5
            return (x.argsort().argsort()[-1] + 1) / len(x)

        return series.rolling(self.lookback, min_periods=self.min_periods).apply(pct_rank)

    def normalize_robust(self, series: pd.Series) -> pd.Series:
        """
        Robust normalization using median and IQR.

        Less sensitive to outliers than z-score.
        Best for: price data with occasional spikes.

        Args:
            series: Input series

        Returns:
            Robustly normalized series
        """
        median = series.rolling(self.lookback, min_periods=self.min_periods).median()
        q25 = series.rolling(self.lookback, min_periods=self.min_periods).quantile(0.25)
        q75 = series.rolling(self.lookback, min_periods=self.min_periods).quantile(0.75)
        iqr = q75 - q25
        return (series - median) / (iqr + 1e-10)


def normalize_features_for_lstm(
    df: pd.DataFrame,
    feature_types: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize features appropriately for LSTM input.

    Different features need different normalization:
    - Returns and changes: z-score
    - Bounded indicators (RSI): already normalized, adjust to [-1, 1]
    - Volatility (ATR): percentile ranking
    - Prices: log returns, not absolute prices

    Args:
        df: DataFrame with features
        feature_types: Optional dict mapping column names to normalization types
                      ("zscore", "minmax", "percentile", "robust", "bounded", "none")

    Returns:
        Normalized DataFrame
    """
    normalizer = FeatureNormalizer(lookback=252, min_periods=20)
    normalized_df = df.copy()

    # Default feature types based on common patterns
    default_types = {
        # Returns and changes -> z-score
        "ret": "zscore",
        "return": "zscore",
        "log_return": "zscore",
        "pct_change": "zscore",
        "macd": "zscore",
        "macd_diff": "zscore",
        "macd_histogram": "zscore",
        "mom": "zscore",
        "momentum": "zscore",

        # Already bounded [0, 1] or similar
        "rsi": "bounded",
        "bb_percent_b": "bounded",
        "bb_bandwidth": "minmax",

        # Volatility -> percentile
        "atr": "percentile",
        "vol": "percentile",
        "volatility": "percentile",
        "atr_normalized": "none",  # Already normalized
        "atr_percentile": "none",  # Already normalized

        # Binary features -> no normalization
        "is_": "none",
        "session_": "none",
        "macd_cross": "none",
        "rsi_overbought": "none",
        "rsi_oversold": "none",
    }

    # Merge with user-provided types
    if feature_types:
        default_types.update(feature_types)

    for col in df.columns:
        # Skip non-numeric columns
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        # Find matching normalization type
        norm_type = "zscore"  # Default
        for pattern, ntype in default_types.items():
            if pattern in col.lower():
                norm_type = ntype
                break

        # Apply normalization
        if norm_type == "zscore":
            normalized_df[col] = normalizer.normalize_zscore(df[col])
        elif norm_type == "minmax":
            normalized_df[col] = normalizer.normalize_minmax(df[col])
        elif norm_type == "percentile":
            normalized_df[col] = normalizer.normalize_percentile(df[col])
        elif norm_type == "robust":
            normalized_df[col] = normalizer.normalize_robust(df[col])
        elif norm_type == "bounded":
            # Assume bounded [0, 100] or [0, 1], convert to [-1, 1]
            if df[col].max() > 1.5:  # Likely [0, 100] range (like RSI)
                normalized_df[col] = (df[col] - 50) / 50
            else:  # Likely [0, 1] range
                normalized_df[col] = df[col] * 2 - 1
        # "none" -> keep as is

    # Fill NaN values (from rolling warmup) with 0
    normalized_df = normalized_df.fillna(0)

    return normalized_df


def create_lookback_windows(
    features: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """
    Create overlapping lookback windows for LSTM input.

    Args:
        features: 2D array of shape (n_samples, n_features)
        lookback: Number of time steps to look back

    Returns:
        3D array of shape (n_samples, lookback, n_features)
    """
    n_samples, n_features = features.shape

    # Create output array
    windows = np.zeros((n_samples, lookback, n_features), dtype=np.float32)

    for i in range(n_samples):
        start_idx = max(0, i - lookback + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx

        # Copy available data
        windows[i, -window_size:, :] = features[start_idx:end_idx, :]

        # Padding is already zeros (from np.zeros initialization)

    return windows


class OnlineNormalizer:
    """
    Online (streaming) normalizer for live trading.

    Maintains running statistics without storing full history.
    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, n_features: int, lookback: int = 252):
        """
        Initialize online normalizer.

        Args:
            n_features: Number of features to normalize
            lookback: Effective lookback for decay
        """
        self.n_features = n_features
        self.lookback = lookback
        self.decay = 1.0 - 1.0 / lookback

        # Running statistics (Welford's algorithm)
        self.count = np.zeros(n_features)
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)
        self.min_val = np.full(n_features, np.inf)
        self.max_val = np.full(n_features, -np.inf)

    def update(self, x: np.ndarray) -> np.ndarray:
        """
        Update statistics and return normalized value.

        Args:
            x: New observation (1D array of shape n_features)

        Returns:
            Normalized observation
        """
        assert x.shape[0] == self.n_features

        # Update min/max with decay
        self.min_val = np.minimum(self.min_val * self.decay + x * (1 - self.decay), x)
        self.max_val = np.maximum(self.max_val * self.decay + x * (1 - self.decay), x)

        # Welford's online algorithm for mean and variance
        self.count = self.count * self.decay + 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = x - self.mean
        self.M2 = self.M2 * self.decay + delta * delta2

        # Calculate variance and std
        variance = self.M2 / (self.count + 1e-10)
        std = np.sqrt(variance + 1e-10)

        # Return z-score normalized value
        normalized = (x - self.mean) / (std + 1e-10)

        # Clip extreme values
        normalized = np.clip(normalized, -5, 5)

        return normalized

    def reset(self):
        """Reset all statistics."""
        self.count = np.zeros(self.n_features)
        self.mean = np.zeros(self.n_features)
        self.M2 = np.zeros(self.n_features)
        self.min_val = np.full(self.n_features, np.inf)
        self.max_val = np.full(self.n_features, -np.inf)
