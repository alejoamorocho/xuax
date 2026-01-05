"""
Trade Pattern Analyzer - Learn from Winners
============================================

This script analyzes completed trades to understand:
1. What feature patterns lead to WINNING trades
2. What feature patterns lead to LOSING trades
3. How to improve entry/exit decisions

The goal is to help the PPO model learn from successful patterns.

Usage:
    python train/analyze_trades.py --model path/to/model.zip
    python train/analyze_trades.py --trades path/to/trades.json
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# Path setup
if os.path.exists('/content/XAUX'):
    _project_root = '/content/XAUX'
else:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def analyze_trade_patterns(trades: list, feature_names: list = None):
    """
    Analyze winning vs losing trade patterns.

    Returns detailed analysis of what features correlate with success.
    """
    if not trades:
        return {"error": "No trades to analyze"}

    # Separate winners and losers
    winners = [t for t in trades if t.get('pnl_pct', 0) > 0]
    losers = [t for t in trades if t.get('pnl_pct', 0) <= 0]

    print(f"\n{'='*60}")
    print("TRADE PATTERN ANALYSIS")
    print(f"{'='*60}")
    print(f"\nTotal trades: {len(trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")

    analysis = {
        'total_trades': len(trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(trades) * 100 if trades else 0,
    }

    # === DURATION ANALYSIS ===
    print(f"\n{'='*60}")
    print("DURATION ANALYSIS")
    print(f"{'='*60}")

    winner_durations = [t.get('duration', 0) for t in winners]
    loser_durations = [t.get('duration', 0) for t in losers]

    if winner_durations:
        print(f"\nWinning trades:")
        print(f"  Avg duration: {np.mean(winner_durations):.1f} bars ({np.mean(winner_durations)*5/60:.2f} hours)")
        print(f"  Min duration: {np.min(winner_durations)} bars")
        print(f"  Max duration: {np.max(winner_durations)} bars")
        analysis['winner_avg_duration'] = float(np.mean(winner_durations))

    if loser_durations:
        print(f"\nLosing trades:")
        print(f"  Avg duration: {np.mean(loser_durations):.1f} bars ({np.mean(loser_durations)*5/60:.2f} hours)")
        print(f"  Min duration: {np.min(loser_durations)} bars")
        print(f"  Max duration: {np.max(loser_durations)} bars")
        analysis['loser_avg_duration'] = float(np.mean(loser_durations))

    # Duration insight
    if winner_durations and loser_durations:
        if np.mean(winner_durations) > np.mean(loser_durations):
            print(f"\n  INSIGHT: Winners are held {np.mean(winner_durations)/np.mean(loser_durations):.1f}x longer!")
            print(f"  RECOMMENDATION: The model should learn to HOLD winning positions longer.")
        else:
            print(f"\n  WARNING: Losers are held longer than winners - CUT LOSSES FASTER!")

    # === EXIT REASON ANALYSIS ===
    print(f"\n{'='*60}")
    print("EXIT REASON ANALYSIS")
    print(f"{'='*60}")

    winner_reasons = defaultdict(int)
    loser_reasons = defaultdict(int)

    for t in winners:
        winner_reasons[t.get('reason', 'unknown')] += 1
    for t in losers:
        loser_reasons[t.get('reason', 'unknown')] += 1

    print(f"\nWinning trade exits:")
    for reason, count in sorted(winner_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(winners) * 100 if winners else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print(f"\nLosing trade exits:")
    for reason, count in sorted(loser_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(losers) * 100 if losers else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    analysis['winner_exit_reasons'] = dict(winner_reasons)
    analysis['loser_exit_reasons'] = dict(loser_reasons)

    # Exit insight
    if winner_reasons.get('trailing_stop', 0) > winner_reasons.get('manual', 0):
        print(f"\n  GOOD: Most winners exit via trailing stop (letting winners run)")
    else:
        print(f"\n  ISSUE: Too many winners closed manually - should use trailing stops more")

    # === EFFICIENCY ANALYSIS ===
    print(f"\n{'='*60}")
    print("EFFICIENCY ANALYSIS (How much of max profit was captured)")
    print(f"{'='*60}")

    winner_efficiencies = [t.get('efficiency', 0) for t in winners]
    if winner_efficiencies:
        print(f"\nWinning trades efficiency:")
        print(f"  Avg efficiency: {np.mean(winner_efficiencies)*100:.1f}%")
        print(f"  >80% efficiency: {sum(1 for e in winner_efficiencies if e >= 0.8)} trades")
        print(f"  >60% efficiency: {sum(1 for e in winner_efficiencies if e >= 0.6)} trades")
        print(f"  <40% efficiency: {sum(1 for e in winner_efficiencies if e < 0.4)} trades")
        analysis['avg_efficiency'] = float(np.mean(winner_efficiencies))

        if np.mean(winner_efficiencies) < 0.5:
            print(f"\n  ISSUE: Low efficiency - closing winners too early!")
            print(f"  RECOMMENDATION: Use trailing stops more, don't exit manually")

    # === FEATURE PATTERN ANALYSIS ===
    print(f"\n{'='*60}")
    print("FEATURE PATTERN ANALYSIS")
    print(f"{'='*60}")

    # Compare entry features between winners and losers
    winner_entry_features = []
    loser_entry_features = []

    for t in winners:
        if 'entry_features' in t and 'raw_features' in t['entry_features']:
            winner_entry_features.append(t['entry_features']['raw_features'])

    for t in losers:
        if 'entry_features' in t and 'raw_features' in t['entry_features']:
            loser_entry_features.append(t['entry_features']['raw_features'])

    if winner_entry_features and loser_entry_features:
        winner_features = np.array(winner_entry_features)
        loser_features = np.array(loser_entry_features)

        # Compare mean feature values
        winner_means = np.mean(winner_features, axis=0)
        loser_means = np.mean(loser_features, axis=0)

        # Find features with biggest difference
        diff = np.abs(winner_means - loser_means)
        top_diff_idx = np.argsort(diff)[-10:][::-1]

        print(f"\nFeatures with BIGGEST difference between winners and losers:")
        print(f"(These features might be predictive of success)")
        print()

        for i, idx in enumerate(top_diff_idx):
            feature_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
            print(f"  {i+1}. {feature_name}:")
            print(f"      Winner avg: {winner_means[idx]:.4f}")
            print(f"      Loser avg:  {loser_means[idx]:.4f}")
            print(f"      Difference: {diff[idx]:.4f}")
            print()

        analysis['top_differentiating_features'] = [
            {
                'index': int(idx),
                'name': feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}",
                'winner_mean': float(winner_means[idx]),
                'loser_mean': float(loser_means[idx]),
                'difference': float(diff[idx]),
            }
            for idx in top_diff_idx
        ]

        # Feature importance suggestion
        print(f"\n  RECOMMENDATION: Focus on these features for entry signals.")
        print(f"  The model should learn to enter when these features have 'winner' patterns.")
    else:
        print("\n  No feature data available for analysis.")
        print("  Run training with the updated environment to collect feature snapshots.")

    # === PNL DISTRIBUTION ===
    print(f"\n{'='*60}")
    print("PNL DISTRIBUTION")
    print(f"{'='*60}")

    winner_pnls = [t.get('pnl_pct', 0) * 100 for t in winners]
    loser_pnls = [t.get('pnl_pct', 0) * 100 for t in losers]

    if winner_pnls:
        print(f"\nWinning trades PnL:")
        print(f"  Avg: +{np.mean(winner_pnls):.3f}%")
        print(f"  Max: +{np.max(winner_pnls):.3f}%")
        print(f"  Total: +{np.sum(winner_pnls):.3f}%")

    if loser_pnls:
        print(f"\nLosing trades PnL:")
        print(f"  Avg: {np.mean(loser_pnls):.3f}%")
        print(f"  Max loss: {np.min(loser_pnls):.3f}%")
        print(f"  Total: {np.sum(loser_pnls):.3f}%")

    # Risk/Reward ratio
    if winner_pnls and loser_pnls:
        avg_win = np.mean(winner_pnls)
        avg_loss = abs(np.mean(loser_pnls))
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"\n  Risk/Reward ratio: {rr_ratio:.2f}:1")

        if rr_ratio < 1:
            print(f"  WARNING: Average loss is bigger than average win!")
            print(f"  RECOMMENDATION: Cut losses faster, hold winners longer.")
        else:
            print(f"  GOOD: Winning trades are {rr_ratio:.1f}x larger than losing trades.")

        analysis['risk_reward_ratio'] = float(rr_ratio)

    # === OVERALL RECOMMENDATIONS ===
    print(f"\n{'='*60}")
    print("OVERALL RECOMMENDATIONS")
    print(f"{'='*60}")

    recommendations = []

    # Check win rate
    win_rate = len(winners) / len(trades) * 100
    if win_rate < 50:
        recommendations.append("Improve entry signals - win rate is below 50%")

    # Check duration
    if winner_durations and loser_durations:
        if np.mean(winner_durations) < np.mean(loser_durations):
            recommendations.append("Hold winners longer - currently cutting profits short")
        if np.mean(loser_durations) > 24:  # > 2 hours
            recommendations.append("Cut losers faster - holding losing positions too long")

    # Check efficiency
    if winner_efficiencies and np.mean(winner_efficiencies) < 0.5:
        recommendations.append("Use trailing stops more - not capturing enough of the move")

    # Check exit reasons
    if loser_reasons.get('manual', 0) > loser_reasons.get('stop_loss', 0):
        recommendations.append("Let stop losses trigger - don't close losers manually")

    if winner_reasons.get('manual', 0) > winner_reasons.get('trailing_stop', 0):
        recommendations.append("Let trailing stops run - don't close winners manually")

    if recommendations:
        print("\nKey improvements needed:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\nNo major issues detected! Model is trading well.")

    analysis['recommendations'] = recommendations

    return analysis


def load_feature_names(data_path):
    """Load feature names from CSV header."""
    import pandas as pd

    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'XAUUSD' in f]
    if not csv_files:
        return None

    df = pd.read_csv(os.path.join(data_path, csv_files[0]), nrows=1)
    exclude = ['date', 'time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'return']
    return [c for c in df.columns if c not in exclude]


def run_live_analysis(model_path: str, n_episodes: int = 5):
    """Run the model and analyze trades in real-time."""
    import pandas as pd

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("ERROR: stable-baselines3 not installed!")
        return None

    from env.gold_trading_env import GoldTradingEnv

    # Load data
    data_path = os.path.join(_project_root, 'data')
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'XAUUSD' in f]
    if not csv_files:
        print("ERROR: No XAUUSD data found")
        return None

    df = pd.read_csv(os.path.join(data_path, csv_files[0]))
    feature_cols = [c for c in df.columns if c not in ['date', 'time', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'return']]

    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change().fillna(0)

    X = df[feature_cols].values.astype(np.float32)
    r = df['return'].values.astype(np.float32)
    prices = df['close'].values.astype(np.float32) if 'close' in df.columns else None

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # Use test portion
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:]
    r_test = r[test_start:]
    prices_test = prices[test_start:] if prices is not None else None

    print(f"Running analysis on {len(X_test)} test samples...")

    # Create environment
    env = GoldTradingEnv(
        features=X_test,
        returns=r_test,
        prices=prices_test,
        window=128,
        initial_balance=10000.0,
    )

    # Load model
    print(f"Loading model from: {model_path}")

    # Check for VecNormalize
    vec_normalize_path = model_path.replace('.zip', '_vec_normalize.pkl').replace('_final', '')
    if not os.path.exists(vec_normalize_path):
        vec_normalize_path = os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')

    vec_env = DummyVecEnv([lambda: env])

    if os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    # Run episodes and collect trades
    all_trades = []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # Collect closed trades
            if info and len(info) > 0 and 'closed_trades' in info[0]:
                all_trades.extend(info[0]['closed_trades'])

        print(f"  Episode {ep+1}/{n_episodes}: {len(env.trades_history)} total trades")
        all_trades.extend(env.trades_history)

    print(f"\nCollected {len(all_trades)} trades for analysis")

    # Analyze
    return analyze_trade_patterns(all_trades, feature_cols)


def main():
    parser = argparse.ArgumentParser(description="Analyze trade patterns")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--trades", type=str, help="Path to trades JSON file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--output", type=str, help="Output JSON file for analysis")

    args = parser.parse_args()

    if args.trades:
        # Load and analyze existing trades
        with open(args.trades, 'r') as f:
            trades = json.load(f)

        data_path = os.path.join(_project_root, 'data')
        feature_names = load_feature_names(data_path)
        analysis = analyze_trade_patterns(trades, feature_names)

    elif args.model:
        # Run model and analyze
        analysis = run_live_analysis(args.model, args.episodes)

    else:
        print("Please provide either --model or --trades")
        return

    # Save analysis
    if args.output and analysis:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {args.output}")


if __name__ == "__main__":
    main()
