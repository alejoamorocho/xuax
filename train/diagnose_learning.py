"""
Deep Learning Diagnostic - Is the Model Learning from Winners?
==============================================================

This script investigates:
1. Is the model learning from winning trades?
2. Why is win rate only ~50%?
3. What patterns distinguish winners from losers?
4. Are the features predictive?

Usage:
    python train/diagnose_learning.py --steps 200000
"""

import os
import sys

# Path setup
if os.path.exists('/content/XAUX'):
    _project_root = '/content/XAUX'
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from datetime import datetime

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    sys.exit(1)

from env.gold_trading_env import GoldTradingEnv


class LearningDiagnosticCallback(BaseCallback):
    """
    Detailed callback to track learning signals.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Track rewards by trade outcome
        self.winner_rewards = []
        self.loser_rewards = []

        # Track action probabilities over time
        self.action_probs_history = []

        # Track value estimates
        self.value_estimates = []

        # Track trades
        self.all_trades = []

        # Track policy updates
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        # Feature tracking at entry
        self.winner_entry_features = []
        self.loser_entry_features = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])

        for info in infos:
            if 'closed_trades' in info:
                for trade in info['closed_trades']:
                    self.all_trades.append(trade)

                    # Separate winner/loser features
                    if trade.get('pnl_pct', 0) > 0:
                        if 'entry_features' in trade and 'raw_features' in trade['entry_features']:
                            self.winner_entry_features.append(trade['entry_features']['raw_features'])
                    else:
                        if 'entry_features' in trade and 'raw_features' in trade['entry_features']:
                            self.loser_entry_features.append(trade['entry_features']['raw_features'])

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout - capture policy metrics."""
        # Try to capture training losses
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # These are logged by SB3 during training
            pass


def load_data():
    """Load training data using the same feature engineering as training."""
    print("Loading features using ultimate_150_features...")

    # Determine data directory
    data_paths = [
        os.path.join(_project_root, 'data'),
        '/content/XAUX/data',
        '/content/drive/MyDrive/XAUX/data',
    ]

    data_dir = None
    for path in data_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it has the main data file
            if os.path.exists(os.path.join(path, 'xauusd_m5.csv')):
                data_dir = path
                print(f"Using data directory: {data_dir}")
                break

    if data_dir is None:
        raise FileNotFoundError(f"No valid data directory found. Checked: {data_paths}")

    try:
        # Use the NEW predictive signal features (same as training)
        from features.predictive_features import make_predictive_features

        X, returns, timestamps = make_predictive_features(base_timeframe='M5', data_dir=data_dir)
        print(f"Using PREDICTIVE SIGNAL features: {X.shape[1]} features")

        # Get feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Try to get actual feature names from the module
        try:
            from features.ultimate_150_features import get_feature_names
            feature_names = get_feature_names()
        except:
            pass

        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")

        # Load prices separately for the environment (use same data_dir)
        prices = None
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for pattern in ['XAUUSD', 'xauusd', 'gold', 'Gold']:
            matching = [f for f in csv_files if pattern in f.lower()]
            if matching:
                df = pd.read_csv(os.path.join(data_dir, matching[0]))
                if 'close' in df.columns:
                    prices = df['close'].values.astype(np.float32)
                    # Align prices with features
                    if len(prices) > len(X):
                        prices = prices[-len(X):]
                    elif len(prices) < len(X):
                        X = X[-len(prices):]
                        returns = returns[-len(prices):]
                break

        return X, returns, prices, feature_names

    except Exception as e:
        print(f"ERROR loading ultimate_150_features: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to raw CSV loading...")

        # Fallback to raw CSV (will have fewer features)
        data_path = os.path.join(_project_root, 'data')
        if not os.path.exists(data_path):
            data_path = '/content/XAUX/data'

        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        csv_file = None
        for pattern in ['XAUUSD', 'xauusd', 'gold']:
            matching = [f for f in csv_files if pattern in f.lower()]
            if matching:
                csv_file = matching[0]
                break

        if csv_file is None:
            raise FileNotFoundError("No data file found")

        df = pd.read_csv(os.path.join(data_path, csv_file))

        feature_cols = [c for c in df.columns if c not in
                       ['date', 'time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'return']]

        if not feature_cols:
            print("WARNING: No feature columns found! Creating basic features...")
            # Create basic features from OHLCV
            df['return'] = df['close'].pct_change().fillna(0)
            df['volatility'] = df['return'].rolling(20).std().fillna(0)
            df['momentum'] = df['close'].pct_change(10).fillna(0)
            df['rsi'] = 50  # Placeholder
            feature_cols = ['return', 'volatility', 'momentum']

        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change().fillna(0)

        X = df[feature_cols].values.astype(np.float32)
        r = df['return'].values.astype(np.float32)
        prices = df['close'].values.astype(np.float32) if 'close' in df.columns else None

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        return X, r, prices, feature_cols


def analyze_feature_predictiveness(winner_features, loser_features, feature_names):
    """
    Analyze if features can distinguish winners from losers.

    This is KEY - if features can't distinguish, the model CAN'T learn!
    """
    print("\n" + "="*60)
    print("FEATURE PREDICTIVENESS ANALYSIS")
    print("="*60)

    if len(winner_features) < 10 or len(loser_features) < 10:
        print("\nInsufficient data for analysis")
        return None

    winner_arr = np.array(winner_features)
    loser_arr = np.array(loser_features)

    print(f"\nSample sizes: {len(winner_features)} winners, {len(loser_features)} losers")

    # Calculate mean difference for each feature
    winner_means = np.mean(winner_arr, axis=0)
    loser_means = np.mean(loser_arr, axis=0)

    winner_stds = np.std(winner_arr, axis=0)
    loser_stds = np.std(loser_arr, axis=0)

    # Calculate effect size (Cohen's d) for each feature
    pooled_std = np.sqrt((winner_stds**2 + loser_stds**2) / 2)
    effect_sizes = np.abs(winner_means - loser_means) / (pooled_std + 1e-8)

    # Find features with largest effect size
    top_idx = np.argsort(effect_sizes)[-15:][::-1]

    print("\n--- TOP 15 PREDICTIVE FEATURES ---")
    print("(Effect size > 0.2 = small, > 0.5 = medium, > 0.8 = large)")
    print()

    predictive_features = []
    for i, idx in enumerate(top_idx):
        if idx < len(feature_names):
            fname = feature_names[idx]
        else:
            fname = f"feature_{idx}"

        es = effect_sizes[idx]
        direction = "HIGHER in winners" if winner_means[idx] > loser_means[idx] else "LOWER in winners"

        predictive_features.append({
            'name': fname,
            'effect_size': float(es),
            'winner_mean': float(winner_means[idx]),
            'loser_mean': float(loser_means[idx]),
            'direction': direction
        })

        # Color code effect size
        if es > 0.8:
            level = "LARGE"
        elif es > 0.5:
            level = "MEDIUM"
        elif es > 0.2:
            level = "SMALL"
        else:
            level = "NONE"

        print(f"{i+1:2}. {fname[:40]:<40} | d={es:.3f} ({level})")
        print(f"    Winner: {winner_means[idx]:+.4f} | Loser: {loser_means[idx]:+.4f} | {direction}")

    # Overall predictiveness score
    n_predictive = sum(1 for es in effect_sizes if es > 0.2)
    n_medium = sum(1 for es in effect_sizes if es > 0.5)
    n_large = sum(1 for es in effect_sizes if es > 0.8)

    print("\n--- PREDICTIVENESS SUMMARY ---")
    print(f"Total features: {len(effect_sizes)}")
    print(f"Small effect (d>0.2): {n_predictive} features")
    print(f"Medium effect (d>0.5): {n_medium} features")
    print(f"Large effect (d>0.8): {n_large} features")

    if n_medium < 5:
        print("\n⚠️  WARNING: Very few predictive features!")
        print("   The model may not have enough signal to learn from.")
        print("   Consider adding more discriminative features.")
    elif n_medium >= 10:
        print("\n✅ GOOD: Multiple predictive features found.")
        print("   The model should be able to learn patterns.")

    return {
        'predictive_features': predictive_features,
        'n_small_effect': n_predictive,
        'n_medium_effect': n_medium,
        'n_large_effect': n_large,
    }


def analyze_reward_signal(trades):
    """
    Analyze if reward signal properly distinguishes winners from losers.

    If winners don't get higher rewards than losers, the model can't learn!
    """
    print("\n" + "="*60)
    print("REWARD SIGNAL ANALYSIS")
    print("="*60)

    if len(trades) < 20:
        print("\nInsufficient trades for analysis")
        return None

    winners = [t for t in trades if t.get('pnl_pct', 0) > 0]
    losers = [t for t in trades if t.get('pnl_pct', 0) <= 0]

    print(f"\nTrades: {len(winners)} winners, {len(losers)} losers")
    print(f"Win rate: {len(winners)/len(trades)*100:.1f}%")

    # Analyze PnL distribution
    winner_pnls = [t['pnl_pct'] * 100 for t in winners]
    loser_pnls = [t['pnl_pct'] * 100 for t in losers]

    print("\n--- PnL DISTRIBUTION ---")
    print(f"Winners: avg={np.mean(winner_pnls):.3f}%, max={np.max(winner_pnls):.3f}%")
    print(f"Losers:  avg={np.mean(loser_pnls):.3f}%, min={np.min(loser_pnls):.3f}%")

    # Risk/Reward ratio
    avg_win = np.mean(winner_pnls)
    avg_loss = abs(np.mean(loser_pnls))
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    print(f"\nRisk/Reward: {rr:.2f}:1")

    if rr < 1:
        print("⚠️  WARNING: Average loss > average win!")
        print("   Even with 50% win rate, this loses money.")

    # Analyze durations
    winner_durations = [t.get('duration', 0) for t in winners]
    loser_durations = [t.get('duration', 0) for t in losers]

    print("\n--- DURATION ANALYSIS ---")
    print(f"Winners: avg={np.mean(winner_durations):.1f} bars ({np.mean(winner_durations)*5/60:.1f} hours)")
    print(f"Losers:  avg={np.mean(loser_durations):.1f} bars ({np.mean(loser_durations)*5/60:.1f} hours)")

    if np.mean(winner_durations) < np.mean(loser_durations):
        print("⚠️  WARNING: Holding losers longer than winners!")
        print("   This is opposite of 'let winners run, cut losers'")

    # Analyze exit reasons
    print("\n--- EXIT REASONS ---")
    winner_reasons = defaultdict(int)
    loser_reasons = defaultdict(int)

    for t in winners:
        winner_reasons[t.get('reason', 'unknown')] += 1
    for t in losers:
        loser_reasons[t.get('reason', 'unknown')] += 1

    print("Winners:")
    for r, c in sorted(winner_reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} ({c/len(winners)*100:.1f}%)")

    print("Losers:")
    for r, c in sorted(loser_reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} ({c/len(losers)*100:.1f}%)")

    # Check if model is using trailing stops properly
    trailing_winners = winner_reasons.get('trailing_stop', 0)
    manual_winners = winner_reasons.get('manual', 0)

    if manual_winners > trailing_winners and len(winners) > 10:
        print("\n⚠️  WARNING: More winners closed manually than via trailing!")
        print("   The model is cutting winners short instead of letting them run.")

    return {
        'win_rate': len(winners) / len(trades) * 100,
        'risk_reward': rr,
        'avg_winner_duration': np.mean(winner_durations),
        'avg_loser_duration': np.mean(loser_durations),
    }


def analyze_learning_progress(callback):
    """
    Analyze if the model is actually learning over time.
    """
    print("\n" + "="*60)
    print("LEARNING PROGRESS ANALYSIS")
    print("="*60)

    trades = callback.all_trades

    if len(trades) < 40:
        print("\nInsufficient trades for progress analysis")
        return None

    # Split trades into quarters
    n = len(trades)
    q = n // 4

    quarters = [
        trades[:q],
        trades[q:2*q],
        trades[2*q:3*q],
        trades[3*q:]
    ]

    print("\n--- WIN RATE BY QUARTER ---")
    print("(Should increase if learning)")

    win_rates = []
    avg_pnls = []
    avg_durations = []

    for i, qt in enumerate(quarters, 1):
        winners = [t for t in qt if t.get('pnl_pct', 0) > 0]
        wr = len(winners) / len(qt) * 100 if qt else 0
        avg_pnl = np.mean([t.get('pnl_pct', 0) * 100 for t in qt]) if qt else 0
        avg_dur = np.mean([t.get('duration', 0) for t in qt]) if qt else 0

        win_rates.append(wr)
        avg_pnls.append(avg_pnl)
        avg_durations.append(avg_dur)

        print(f"Q{i}: Win rate={wr:.1f}%, Avg PnL={avg_pnl:+.3f}%, Avg Duration={avg_dur:.1f} bars")

    # Check for improvement
    wr_trend = win_rates[-1] - win_rates[0]
    pnl_trend = avg_pnls[-1] - avg_pnls[0]
    dur_trend = avg_durations[-1] - avg_durations[0]

    print("\n--- LEARNING ASSESSMENT ---")

    if wr_trend > 5:
        print(f"✅ Win rate IMPROVING: +{wr_trend:.1f}%")
    elif wr_trend < -5:
        print(f"❌ Win rate DECLINING: {wr_trend:.1f}%")
    else:
        print(f"➖ Win rate FLAT: {wr_trend:+.1f}%")

    if pnl_trend > 0.01:
        print(f"✅ Average PnL IMPROVING: +{pnl_trend:.3f}%")
    elif pnl_trend < -0.01:
        print(f"❌ Average PnL DECLINING: {pnl_trend:.3f}%")
    else:
        print(f"➖ Average PnL FLAT: {pnl_trend:+.3f}%")

    if dur_trend > 5:
        print(f"✅ Hold duration INCREASING: +{dur_trend:.1f} bars")
    elif dur_trend < -5:
        print(f"❌ Hold duration DECREASING: {dur_trend:.1f} bars")
    else:
        print(f"➖ Hold duration FLAT: {dur_trend:+.1f} bars")

    # Overall assessment
    is_learning = (wr_trend > 2 or pnl_trend > 0.005)

    print("\n" + "="*40)
    if is_learning:
        print("ASSESSMENT: Model IS learning (some improvement)")
    else:
        print("ASSESSMENT: Model NOT learning (no improvement)")
        print("\nPossible causes:")
        print("1. Features not predictive enough")
        print("2. Reward signal not clear enough")
        print("3. Learning rate too low/high")
        print("4. Need more training steps")

    return {
        'win_rate_trend': wr_trend,
        'pnl_trend': pnl_trend,
        'duration_trend': dur_trend,
        'is_learning': is_learning,
    }


def run_diagnostic(args):
    """Run complete learning diagnostic."""
    print("="*60)
    print("DEEP LEARNING DIAGNOSTIC")
    print("="*60)
    print(f"Training steps: {args.steps}")
    print(f"Device: {args.device}")

    # Load data
    X, r, prices, feature_names = load_data()

    # Use subset
    subset = min(len(X), 80000)
    X = X[:subset]
    r = r[:subset]
    prices = prices[:subset] if prices is not None else None

    print(f"\nUsing {len(X)} samples")
    print(f"Features: {len(feature_names)}")

    # Create environment
    def make_env():
        env = GoldTradingEnv(
            features=X,
            returns=r,
            prices=prices,
            window=64,  # Smaller for faster diagnostic
            initial_balance=10000.0,
            max_drawdown=0.30,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create model with current config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        device=args.device,
    )

    # Setup callback
    callback = LearningDiagnosticCallback()

    # Train
    print("\n" + "="*60)
    print("TRAINING (collecting diagnostic data)...")
    print("="*60)

    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=True)

    # Analyze results
    print("\n")

    # 1. Feature predictiveness
    feature_analysis = analyze_feature_predictiveness(
        callback.winner_entry_features,
        callback.loser_entry_features,
        feature_names
    )

    # 2. Reward signal
    reward_analysis = analyze_reward_signal(callback.all_trades)

    # 3. Learning progress
    learning_analysis = analyze_learning_progress(callback)

    # Final summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    issues = []

    if feature_analysis and feature_analysis['n_medium_effect'] < 5:
        issues.append("LOW FEATURE PREDICTIVENESS - Features don't distinguish winners from losers")

    if reward_analysis:
        if reward_analysis['win_rate'] < 48:
            issues.append("WIN RATE BELOW 50% - Model has no edge")
        if reward_analysis['risk_reward'] < 1:
            issues.append("NEGATIVE RISK/REWARD - Losses bigger than wins")
        if reward_analysis['avg_loser_duration'] > reward_analysis['avg_winner_duration']:
            issues.append("HOLDING LOSERS TOO LONG - Cut losses faster")

    if learning_analysis and not learning_analysis['is_learning']:
        issues.append("NO LEARNING DETECTED - Model not improving over time")

    if issues:
        print("\n❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n📋 RECOMMENDATIONS:")
        if "FEATURE PREDICTIVENESS" in str(issues):
            print("   - Add more predictive features (momentum, trend indicators)")
            print("   - Check if features are properly normalized")
            print("   - Consider adding market regime indicators")

        if "WIN RATE" in str(issues) or "RISK/REWARD" in str(issues):
            print("   - Review entry conditions - what triggers trades?")
            print("   - Add filters (trend alignment, volatility)")
            print("   - Consider only trading during high-probability setups")

        if "HOLDING LOSERS" in str(issues):
            print("   - Increase penalty for holding losing positions")
            print("   - Tighten stop losses")
            print("   - Add time-based exit for stagnant trades")

        if "NO LEARNING" in str(issues):
            print("   - Increase training steps")
            print("   - Adjust learning rate")
            print("   - Simplify reward function")
    else:
        print("\n✅ No major issues detected!")

    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Diagnostic")
    parser.add_argument("--steps", type=int, default=200000, help="Training steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")

    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_diagnostic(args)
