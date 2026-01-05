"""
PPO Diagnostic Test - Analyze Entry/Exit Behavior and Feature Convergence
==========================================================================

This script runs a short training to diagnose:
1. Entry/Exit patterns - Is the model learning when to enter/exit?
2. Feature convergence - Are features being used effectively?
3. Reward distribution - Is the reward function guiding learning?
4. Trade quality - Are trades improving over time?

Usage:
    python train/diagnose_ppo.py --steps 100000 --window 64

The test will output detailed diagnostics about the model's behavior.
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
import torch
import json
from datetime import datetime
from collections import defaultdict

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    sys.exit(1)

from env.gold_trading_env import GoldTradingEnv


class DiagnosticCallback(BaseCallback):
    """
    Callback to collect detailed diagnostics during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.diagnostics = {
            'rewards': [],
            'actions': [],
            'positions': [],
            'trades_opened': [],
            'trades_closed': [],
            'daily_trades': [],
            'hold_durations': [],
            'pnl_per_trade': [],
            'step_rewards': defaultdict(list),  # Reward components
        }
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Get info from environment
        infos = self.locals.get('infos', [{}])
        actions = self.locals.get('actions', [])
        rewards = self.locals.get('rewards', [])

        for i, info in enumerate(infos):
            if i < len(rewards):
                self.diagnostics['rewards'].append(float(rewards[i]))
            if i < len(actions):
                self.diagnostics['actions'].append(int(actions[i]))

            self.diagnostics['positions'].append(info.get('num_positions', 0))
            self.diagnostics['daily_trades'].append(info.get('daily_trades', 0))

            # Track closed trades
            if 'closed_trades' in info:
                for trade in info['closed_trades']:
                    self.diagnostics['hold_durations'].append(trade.get('duration', 0))
                    self.diagnostics['pnl_per_trade'].append(trade.get('pnl_pct', 0))
                    self.diagnostics['trades_closed'].append({
                        'step': self.num_timesteps,
                        'pnl_pct': trade.get('pnl_pct', 0),
                        'duration': trade.get('duration', 0),
                        'reason': trade.get('reason', 'unknown'),
                        'efficiency': trade.get('efficiency', 0),
                    })

        return True

    def get_summary(self):
        """Generate diagnostic summary."""
        summary = {}

        # Action distribution
        actions = np.array(self.diagnostics['actions'])
        if len(actions) > 0:
            summary['action_distribution'] = {
                'flat_pct': float(np.mean(actions == 0) * 100),
                'long_pct': float(np.mean(actions == 1) * 100),
            }

        # Trade statistics
        trades = self.diagnostics['trades_closed']
        if len(trades) > 0:
            pnls = [t['pnl_pct'] for t in trades]
            durations = [t['duration'] for t in trades]
            reasons = [t['reason'] for t in trades]

            summary['trade_stats'] = {
                'total_trades': len(trades),
                'avg_pnl_pct': float(np.mean(pnls) * 100),
                'win_rate': float(np.mean([p > 0 for p in pnls]) * 100),
                'avg_duration_bars': float(np.mean(durations)),
                'avg_duration_hours': float(np.mean(durations) * 5 / 60),
                'exit_reasons': {
                    'manual': sum(1 for r in reasons if r == 'manual'),
                    'stop_loss': sum(1 for r in reasons if r == 'stop_loss'),
                    'trailing_stop': sum(1 for r in reasons if r == 'trailing_stop'),
                },
            }

            # Quality metrics
            if len(durations) >= 10:
                # Compare first half vs second half
                mid = len(durations) // 2
                summary['trade_quality_evolution'] = {
                    'first_half': {
                        'avg_duration': float(np.mean(durations[:mid])),
                        'avg_pnl': float(np.mean(pnls[:mid]) * 100),
                        'win_rate': float(np.mean([p > 0 for p in pnls[:mid]]) * 100),
                    },
                    'second_half': {
                        'avg_duration': float(np.mean(durations[mid:])),
                        'avg_pnl': float(np.mean(pnls[mid:]) * 100),
                        'win_rate': float(np.mean([p > 0 for p in pnls[mid:]]) * 100),
                    },
                }

        # Reward statistics
        rewards = np.array(self.diagnostics['rewards'])
        if len(rewards) > 0:
            summary['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            }

            # Reward evolution (split into 4 quarters)
            n = len(rewards)
            q = n // 4
            if q > 0:
                summary['reward_evolution'] = {
                    'q1_mean': float(np.mean(rewards[:q])),
                    'q2_mean': float(np.mean(rewards[q:2*q])),
                    'q3_mean': float(np.mean(rewards[2*q:3*q])),
                    'q4_mean': float(np.mean(rewards[3*q:])),
                }

        # Position statistics
        positions = np.array(self.diagnostics['positions'])
        if len(positions) > 0:
            summary['position_stats'] = {
                'time_in_market_pct': float(np.mean(positions > 0) * 100),
                'avg_positions': float(np.mean(positions)),
            }

        # Daily trade frequency
        daily = np.array(self.diagnostics['daily_trades'])
        if len(daily) > 0:
            summary['daily_trade_stats'] = {
                'avg_daily_trades': float(np.mean(daily)),
                'max_daily_trades': int(np.max(daily)),
            }

        return summary


def load_data(data_path):
    """Load and prepare data for training."""
    import pandas as pd

    print(f"Loading data from: {data_path}")

    # Find CSV file
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'XAUUSD' in f]
    if not csv_files:
        raise FileNotFoundError(f"No XAUUSD CSV found in {data_path}")

    csv_path = os.path.join(data_path, csv_files[0])
    print(f"Using: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Extract features and returns
    feature_cols = [c for c in df.columns if c not in ['date', 'time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'return']]

    if 'return' not in df.columns:
        if 'close' in df.columns:
            df['return'] = df['close'].pct_change().fillna(0)
        else:
            raise ValueError("No 'return' or 'close' column found")

    X = df[feature_cols].values.astype(np.float32)
    r = df['return'].values.astype(np.float32)

    prices = df['close'].values.astype(np.float32) if 'close' in df.columns else None

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Features: {X.shape}, Returns: {r.shape}")
    return X, r, prices, feature_cols


def analyze_features(model, feature_names, n_features=20):
    """
    Analyze which features the model is paying attention to.

    Uses the first layer weights to estimate feature importance.
    """
    try:
        # Get policy network
        policy = model.policy

        # Get first layer weights
        first_layer = None
        for name, param in policy.named_parameters():
            if 'mlp_extractor.policy_net.0.weight' in name or 'features_extractor' in name:
                first_layer = param.detach().cpu().numpy()
                break

        if first_layer is None:
            return None

        # Compute importance as mean absolute weight per input feature
        # Note: Input includes window * features + position state
        n_input = first_layer.shape[1]
        n_orig_features = len(feature_names)

        # Average across window dimension
        window_size = (n_input - 8) // n_orig_features  # 8 = position state features

        if window_size <= 0:
            return None

        # Reshape and average
        feature_weights = first_layer[:, :n_orig_features * window_size]
        feature_weights = feature_weights.reshape(first_layer.shape[0], window_size, n_orig_features)
        importance = np.mean(np.abs(feature_weights), axis=(0, 1))

        # Top features
        top_idx = np.argsort(importance)[-n_features:][::-1]
        top_features = [(feature_names[i], float(importance[i])) for i in top_idx]

        return {
            'top_features': top_features,
            'importance_std': float(np.std(importance)),
            'importance_mean': float(np.mean(importance)),
        }

    except Exception as e:
        print(f"Feature analysis error: {e}")
        return None


def run_diagnostic(args):
    """Run diagnostic training and analysis."""
    print("=" * 60)
    print("PPO DIAGNOSTIC TEST")
    print("=" * 60)

    # Load data
    data_path = os.path.join(_project_root, 'data')
    X, r, prices, feature_names = load_data(data_path)

    # Use subset for faster testing
    subset_size = min(len(X), 50000)  # Use at most 50k samples
    X = X[:subset_size]
    r = r[:subset_size]
    if prices is not None:
        prices = prices[:subset_size]

    print(f"\nUsing {len(X)} samples for diagnostic")
    print(f"Window: {args.window}")
    print(f"Training steps: {args.steps}")

    # Create environment
    def make_env():
        env = GoldTradingEnv(
            features=X,
            returns=r,
            prices=prices,
            window=args.window,
            initial_balance=10000.0,
            max_drawdown=0.30,
            max_trades_per_day=8,
            trade_cooldown=12,
        )
        return Monitor(env)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create model with diagnostic-friendly settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=64,
        gamma=0.6,
        ent_coef=0.02,
        clip_range=0.2,
        verbose=1,
        device=args.device,
    )

    # Setup callback
    diagnostic_callback = DiagnosticCallback()

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    start_time = datetime.now()

    model.learn(total_timesteps=args.steps, callback=diagnostic_callback, progress_bar=True)

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Get diagnostics
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)

    summary = diagnostic_callback.get_summary()

    # Print action distribution
    if 'action_distribution' in summary:
        print("\n--- ACTION DISTRIBUTION ---")
        ad = summary['action_distribution']
        print(f"  Flat (0):  {ad['flat_pct']:.1f}%")
        print(f"  Long (1):  {ad['long_pct']:.1f}%")

    # Print trade statistics
    if 'trade_stats' in summary:
        print("\n--- TRADE STATISTICS ---")
        ts = summary['trade_stats']
        print(f"  Total trades: {ts['total_trades']}")
        print(f"  Avg PnL: {ts['avg_pnl_pct']:.3f}%")
        print(f"  Win rate: {ts['win_rate']:.1f}%")
        print(f"  Avg duration: {ts['avg_duration_bars']:.1f} bars ({ts['avg_duration_hours']:.2f} hours)")
        print(f"  Exit reasons:")
        for reason, count in ts['exit_reasons'].items():
            print(f"    - {reason}: {count}")

    # Print trade quality evolution
    if 'trade_quality_evolution' in summary:
        print("\n--- TRADE QUALITY EVOLUTION ---")
        tqe = summary['trade_quality_evolution']
        print("  First half:")
        print(f"    Avg duration: {tqe['first_half']['avg_duration']:.1f} bars")
        print(f"    Avg PnL: {tqe['first_half']['avg_pnl']:.3f}%")
        print(f"    Win rate: {tqe['first_half']['win_rate']:.1f}%")
        print("  Second half:")
        print(f"    Avg duration: {tqe['second_half']['avg_duration']:.1f} bars")
        print(f"    Avg PnL: {tqe['second_half']['avg_pnl']:.3f}%")
        print(f"    Win rate: {tqe['second_half']['win_rate']:.1f}%")

        # Check for improvement
        improved = (
            tqe['second_half']['avg_duration'] > tqe['first_half']['avg_duration'] or
            tqe['second_half']['win_rate'] > tqe['first_half']['win_rate']
        )
        print(f"\n  LEARNING: {'YES - Model is improving!' if improved else 'NO - Model not converging'}")

    # Print reward evolution
    if 'reward_evolution' in summary:
        print("\n--- REWARD EVOLUTION ---")
        re = summary['reward_evolution']
        print(f"  Q1 mean: {re['q1_mean']:.4f}")
        print(f"  Q2 mean: {re['q2_mean']:.4f}")
        print(f"  Q3 mean: {re['q3_mean']:.4f}")
        print(f"  Q4 mean: {re['q4_mean']:.4f}")

        improving = re['q4_mean'] > re['q1_mean']
        print(f"\n  REWARD TREND: {'IMPROVING' if improving else 'DECLINING'}")

    # Print position statistics
    if 'position_stats' in summary:
        print("\n--- POSITION STATISTICS ---")
        ps = summary['position_stats']
        print(f"  Time in market: {ps['time_in_market_pct']:.1f}%")
        print(f"  Avg positions: {ps['avg_positions']:.2f}")

    # Print daily trade stats
    if 'daily_trade_stats' in summary:
        print("\n--- DAILY TRADE FREQUENCY ---")
        dts = summary['daily_trade_stats']
        print(f"  Avg daily trades: {dts['avg_daily_trades']:.1f}")
        print(f"  Max daily trades: {dts['max_daily_trades']}")
        print(f"  Over-trading: {'YES' if dts['avg_daily_trades'] > 8 else 'NO'}")

    # Feature importance analysis
    print("\n--- FEATURE IMPORTANCE ---")
    feature_analysis = analyze_features(model, feature_names)
    if feature_analysis:
        print("  Top 10 features by importance:")
        for i, (name, imp) in enumerate(feature_analysis['top_features'][:10]):
            print(f"    {i+1}. {name}: {imp:.4f}")
        print(f"\n  Importance std: {feature_analysis['importance_std']:.4f}")
        print(f"  (Higher std = model is differentiating features)")
    else:
        print("  Could not analyze features")

    # Save diagnostics
    output_dir = os.path.join(_project_root, 'train', 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'diagnostic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    # Add feature analysis to summary
    if feature_analysis:
        summary['feature_analysis'] = feature_analysis

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- DIAGNOSTICS SAVED ---")
    print(f"  {output_file}")

    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)

    issues = []
    if 'trade_stats' in summary:
        ts = summary['trade_stats']
        if ts['total_trades'] > args.steps / 100:
            issues.append("Over-trading detected")
        if ts['avg_duration_bars'] < 6:
            issues.append("Holding positions too short")
        if ts['win_rate'] < 45:
            issues.append("Win rate too low")

    if 'daily_trade_stats' in summary:
        if summary['daily_trade_stats']['avg_daily_trades'] > 10:
            issues.append("Daily trade limit exceeded")

    if 'action_distribution' in summary:
        ad = summary['action_distribution']
        if ad['flat_pct'] > 90 or ad['flat_pct'] < 10:
            issues.append("Action distribution imbalanced")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No major issues detected!")

    print("\n" + "=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Diagnostic Test")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--window", type=int, default=64, help="Window size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()
    run_diagnostic(args)
