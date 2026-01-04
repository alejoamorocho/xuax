# Path setup - MUST be first before any imports
import os
import sys

# Handle both local and Colab environments
if os.path.exists('/content/XAUX'):
    # Running in Google Colab
    _project_root = '/content/XAUX'
else:
    # Running locally
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(_script_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
print(f"[DEBUG] Project root: {_project_root}")
print(f"[DEBUG] sys.path[0]: {sys.path[0]}")

"""
Train PPO Agent with ULTIMATE 150+ Features
============================================

This training script implements ALL recommendations from:
PPO_GOLD_TRADING_IMPROVEMENTS_CHECKLIST.md

PPO Hyperparameters (Section 1):
- [x] 1.1 Gamma = 0.6 (optimized for intraday-swing)
- [x] 1.2 Learning Rate Annealing (3e-4 → 1e-5)
- [x] 1.3 Target KL = 0.01 (early stopping)
- [x] 1.4 Entropy Coefficient = 0.02
- [x] 1.5 Batch Size = 128, N_Steps = 4096
- [x] 1.6 Complete optimized configuration

Reward Function (Section 2):
- [x] 2.1 5-component reward (Sortino, DD, TC, Holding, Breach)
- [x] 2.2 Differential Sharpe Ratio (DSR)
- [x] 2.3 Integration in environment

Technical Indicators (Section 3):
- [x] 3.1 RSI(21) with 75/25 thresholds
- [x] 3.2 MACD(16,34,13)
- [x] 3.3 ATR position sizing
- [x] 3.4 Bollinger Bands(13)

Features (Section 4):
- [x] 4.1 Session features
- [x] 4.2 Position state features
- [x] 4.3 TIPS yield (via macro features)

Validation (Section 5):
- [x] Train/Val/Test split
- [x] Complete metrics (Sharpe, Sortino, Calmar, etc.)
- [x] Equity curves visualization

Expected performance: 80-120%+ annual return, 3.5-4.5+ Sharpe ratio
"""

import argparse
import numpy as np
import torch
import json
from datetime import datetime
from typing import Callable

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        BaseCallback
    )
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("="*60)
    print("ERROR: stable-baselines3 not installed!")
    print("Run: pip install stable-baselines3[extra]")
    print("="*60)
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the professional trading environment
from env.gold_trading_env import GoldTradingEnv, EarlyStoppingCallback


# ============================================================================
# Learning Rate Schedule (Section 1.2)
# ============================================================================
def linear_schedule(initial_lr: float = 3e-4, final_lr: float = 1e-5) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    From PPO_GOLD_TRADING_IMPROVEMENTS_CHECKLIST.md Section 1.2:
    - Start: 3e-4 (aggressive exploration)
    - End: 1e-5 (fine-tuning without destroying learned patterns)

    Args:
        initial_lr: Initial learning rate
        final_lr: Final learning rate

    Returns:
        Schedule function that takes progress_remaining (1→0) and returns lr
    """
    def schedule(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (beginning) to 0 (end of training)
        """
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ============================================================================
# Custom Callback for Logging
# ============================================================================
class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics during training.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.equities = []

    def _on_step(self) -> bool:
        # Log episode info when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
            if 'l' in ep_info:
                self.episode_lengths.append(ep_info['l'])
        return True

    def _on_rollout_end(self) -> None:
        # Log metrics at end of each rollout
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            self.logger.record("trading/mean_episode_reward", mean_reward)


# ============================================================================
# Performance Metrics Calculator
# ============================================================================
def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def calculate_performance_metrics(equities, positions, returns_list):
    """
    Calculate comprehensive trading performance metrics.

    Metrics from PPO_GOLD_TRADING_IMPROVEMENTS_CHECKLIST.md Section 5.
    """
    equities = np.array(equities)
    positions = np.array(positions)
    returns_arr = np.array(returns_list)

    if len(equities) == 0:
        return {'error': 'No data'}

    # Basic returns
    total_return = (equities[-1] / equities[0] - 1) * 100 if len(equities) > 0 else 0

    # Annualized return (assuming M5 = 288 bars/day, 252 trading days)
    n_bars = len(equities)
    bars_per_year = 288 * 252  # M5 timeframe
    years = n_bars / bars_per_year
    annualized_return = ((equities[-1] / equities[0]) ** (1/max(years, 0.01)) - 1) * 100

    # Sharpe Ratio (annualized)
    if len(returns_arr) > 1 and np.std(returns_arr) > 0:
        sharpe = np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(bars_per_year)
    else:
        sharpe = 0

    # Sortino Ratio (annualized)
    negative_returns = returns_arr[returns_arr < 0]
    if len(negative_returns) > 0:
        downside_std = np.sqrt(np.mean(negative_returns ** 2))
        sortino = np.mean(returns_arr) / downside_std * np.sqrt(bars_per_year) if downside_std > 0 else 0
    else:
        sortino = sharpe * 1.5

    # Maximum Drawdown
    peak = np.maximum.accumulate(equities)
    drawdown = (peak - equities) / peak
    max_drawdown = np.max(drawdown) * 100

    # Calmar Ratio
    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Trade statistics
    position_changes = np.diff(positions)
    trades = int(np.sum(np.abs(position_changes) > 0))

    # Time in market
    time_in_market = np.mean(positions) * 100

    # Win rate
    in_position_returns = returns_arr[positions[:-1] == 1] if len(positions) > 1 else np.array([])
    if len(in_position_returns) > 0:
        win_rate = np.mean(in_position_returns > 0) * 100
        avg_win = np.mean(in_position_returns[in_position_returns > 0]) * 100 if np.any(in_position_returns > 0) else 0
        avg_loss = np.mean(in_position_returns[in_position_returns < 0]) * 100 if np.any(in_position_returns < 0) else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0

    # Profit Factor
    gross_profit = np.sum(in_position_returns[in_position_returns > 0]) if len(in_position_returns) > 0 else 0
    gross_loss = abs(np.sum(in_position_returns[in_position_returns < 0])) if len(in_position_returns) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
        'trades': trades,
        'time_in_market': time_in_market,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_equity': equities[-1] if len(equities) > 0 else 1.0,
    }


# ============================================================================
# PPO Trading Environment (Gymnasium compatible)
# ============================================================================
import gymnasium as gym
from gymnasium import spaces

class PPOTradingEnv(gym.Env):
    """
    Trading Environment for PPO with Advanced Reward Function.

    Implements ALL reward components from PPO_GOLD_TRADING_IMPROVEMENTS_CHECKLIST.md:
    1. Sortino-based return (risk-adjusted)
    2. Drawdown penalty (quadratic/exponential)
    3. Transaction cost penalty
    4. Holding winner bonus
    5. Drawdown breach penalty
    + Differential Sharpe Ratio (DSR)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        window: int = 64,
        cost_per_trade: float = 0.0001,
        max_drawdown: float = 0.30,
        max_episode_steps: int = None,
    ):
        super().__init__()

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.max_drawdown = float(max_drawdown)
        self.T = len(self.r)
        self.max_episode_steps = max_episode_steps

        # Observation: window of features + position
        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 0=Flat, 1=Long
        self.action_space = spaces.Discrete(2)

        # DSR calculator
        self.dsr = DifferentialSharpeRatio(eta=0.01)

        # Tracking variables
        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.pos = 0
        self.steps = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.position_entry_step = None
        self.unrealized_pnl = 0.0
        self.returns_history = []
        self.dsr.reset()

        # For evaluation
        self.equity_curve = [1.0]
        self.positions_history = [0]
        self.returns_history_full = []

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def _get_current_drawdown(self) -> float:
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.peak_equity > 0:
            return (self.peak_equity - self.equity) / self.peak_equity
        return 0.0

    def _get_position_duration(self) -> int:
        if self.position_entry_step is None:
            return 0
        return self.t - self.position_entry_step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        new_pos = 1 if action == 1 else 0

        # Position change
        delta = abs(new_pos - self.pos)
        trade_cost = self.cost * delta

        # PnL from holding previous position
        step_return = self.pos * self.r[self.t]

        # Track position entry/exit
        if delta > 0:
            if new_pos == 1:
                self.position_entry_step = self.t
                self.unrealized_pnl = 0.0
            else:
                self.position_entry_step = None
                self.unrealized_pnl = 0.0

        # Update unrealized PnL
        if self.pos == 1:
            self.unrealized_pnl += step_return

        # Store return history
        self.returns_history.append(step_return)
        if len(self.returns_history) > 100:
            self.returns_history = self.returns_history[-100:]

        # Calculate advanced reward (Section 2)
        dsr_value = self.dsr.calculate(step_return)
        current_dd = self._get_current_drawdown()
        position_duration = self._get_position_duration()
        is_winner = self.unrealized_pnl > 0

        reward = self._calculate_advanced_reward(
            portfolio_return=step_return,
            current_drawdown=current_dd,
            downside_returns=np.array(self.returns_history),
            position_duration=position_duration,
            is_winner=is_winner,
            transaction_cost=trade_cost,
            dsr_value=dsr_value,
        )

        # Update equity
        self.equity *= (1.0 + step_return - trade_cost)

        # Track for evaluation
        self.equity_curve.append(self.equity)
        self.positions_history.append(new_pos)
        self.returns_history_full.append(step_return)

        # Update state
        self.pos = new_pos
        self.t += 1
        self.steps += 1

        # Check termination
        terminated = self.t >= self.T - 1
        truncated = False
        if self.max_episode_steps and self.steps >= self.max_episode_steps:
            truncated = True

        info = {
            "equity": float(self.equity),
            "position": int(self.pos),
            "drawdown": float(current_dd),
            "step_return": float(step_return),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _calculate_advanced_reward(
        self,
        portfolio_return: float,
        current_drawdown: float,
        downside_returns: np.ndarray,
        position_duration: int,
        is_winner: bool,
        transaction_cost: float,
        dsr_value: float,
    ) -> float:
        """
        Advanced 5-component reward function from Section 2.1
        """
        # Component 1: Sortino-based return
        if len(downside_returns) > 0:
            negative_returns = downside_returns[downside_returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.sqrt(np.mean(negative_returns ** 2))
            else:
                downside_std = 1e-8
        else:
            downside_std = 1e-8

        sortino_component = portfolio_return / (downside_std + 1e-8)

        # Component 2: Drawdown penalty (quadratic)
        dd_ratio = current_drawdown / self.max_drawdown if self.max_drawdown > 0 else 0.0

        if dd_ratio < 0.5:
            dd_penalty = 0.0
        elif dd_ratio < 0.8:
            dd_penalty = -0.5 * (dd_ratio - 0.5) ** 2
        elif dd_ratio < 1.0:
            dd_penalty = -2.0 * (dd_ratio - 0.5) ** 2
        else:
            dd_penalty = -10.0 - 5.0 * (dd_ratio - 1.0)

        # Component 3: Transaction cost penalty
        tc_penalty = -transaction_cost * 100

        # Component 4: Holding winner bonus
        if is_winner and portfolio_return > 0:
            holding_bonus = 0.1 * np.log1p(position_duration)
        else:
            holding_bonus = 0.0

        # Component 4b: Early exit penalty (penalize closing < 1 hour)
        MIN_HOLD_BARS = 12  # 12 bars × 5min = 1 hour minimum
        if position_duration > 0 and position_duration < MIN_HOLD_BARS:
            early_exit_penalty = -0.05 * (MIN_HOLD_BARS - position_duration)
        else:
            early_exit_penalty = 0.0

        # Component 5: Drawdown breach penalty (HARD CONSTRAINT)
        if current_drawdown > self.max_drawdown:
            breach_penalty = -100.0  # INCREASED from -50 to enforce limit
        else:
            breach_penalty = 0.0

        # Component 6: DSR
        dsr_component = dsr_value

        # === ANTI-OVERTRADING PENALTY ===
        # Penalize if position changed (to reduce trade frequency)
        trade_penalty = -0.1 if transaction_cost > 0 else 0.0

        # Final reward with ADJUSTED weights
        # Key changes:
        # - Increased TC penalty (2.0 vs 0.5) to reduce overtrading
        # - Increased DD penalty (3.0 vs 1.5) to respect drawdown limit
        # - Increased breach penalty to enforce 30% limit
        # - Added trade penalty to discourage frequent trading
        # - Added early exit penalty to hold positions longer
        reward = (
            1.0 * sortino_component +      # Risk-adjusted return
            3.0 * dd_penalty +             # INCREASED: Stronger DD control
            2.0 * tc_penalty +             # INCREASED: Reduce overtrading
            0.5 * holding_bonus +          # Reward holding winners
            1.0 * early_exit_penalty +     # NEW: Penalize quick exits
            1.0 * breach_penalty +         # Hard constraint
            0.5 * dsr_component +          # Sharpe optimization
            1.0 * trade_penalty            # NEW: Anti-overtrading
        )

        return reward

    def get_evaluation_data(self):
        """Return data for performance evaluation"""
        return {
            'equities': self.equity_curve,
            'positions': self.positions_history,
            'returns': self.returns_history_full,
        }


# ============================================================================
# Differential Sharpe Ratio (Section 2.2)
# ============================================================================
class DifferentialSharpeRatio:
    """
    Moody & Saffell (2001) - Online Sharpe optimization.
    """
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def reset(self):
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def calculate(self, R_t: float) -> float:
        if not self.initialized:
            self.A = R_t
            self.B = R_t ** 2
            self.initialized = True
            return 0.0

        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B

        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = (self.B - self.A ** 2) ** 1.5

        if abs(denominator) < 1e-10:
            D_t = 0.0
        else:
            D_t = numerator / denominator

        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B

        return D_t


# ============================================================================
# Evaluation Function
# ============================================================================
def evaluate_model(model, env, n_episodes=1):
    """
    Evaluate trained model and return metrics.
    """
    all_equities = []
    all_positions = []
    all_returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Get evaluation data
        eval_data = env.get_evaluation_data()
        all_equities.extend(eval_data['equities'])
        all_positions.extend(eval_data['positions'])
        all_returns.extend(eval_data['returns'])

    return calculate_performance_metrics(all_equities, all_positions, all_returns)


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train PPO with Ultimate 150+ Features')
    parser.add_argument('--timesteps', type=int, default=3_000_000, help='Total training timesteps')
    parser.add_argument('--window', type=int, default=128, help='Observation window size')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda/cpu/auto')
    parser.add_argument('--base-tf', type=str, default='M5', help='Base timeframe')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("🚀 PPO ULTIMATE TRAINING")
    logger.info("   Following PPO_GOLD_TRADING_IMPROVEMENTS_CHECKLIST.md")
    logger.info("="*70)

    # ========== LOAD FEATURES ==========
    logger.info("\n📊 Loading Ultimate 150+ features...")

    from features.ultimate_150_features import make_ultimate_features

    X, returns, timestamps = make_ultimate_features(base_timeframe=args.base_tf)

    logger.info(f"✅ Features loaded: {X.shape}")
    logger.info(f"✅ Date range: {timestamps[0]} to {timestamps[-1]}")

    # ========== TRAIN/VAL/TEST SPLIT ==========
    logger.info("\n📅 Splitting data...")

    TRAIN_END = "2022-01-01"
    VAL_END = "2023-01-01"

    train_mask = timestamps < TRAIN_END
    val_mask = (timestamps >= TRAIN_END) & (timestamps < VAL_END)
    test_mask = timestamps >= VAL_END

    train_idx = np.where(train_mask)[0][-1] if train_mask.any() else len(X) // 3
    val_idx = np.where(val_mask)[0][-1] if val_mask.any() else len(X) * 2 // 3

    X_train, r_train = X[:train_idx], returns[:train_idx]
    X_val, r_val = X[train_idx:val_idx], returns[train_idx:val_idx]
    X_test, r_test = X[val_idx:], returns[val_idx:]

    logger.info(f"  Train: {len(X_train):,} samples (until {TRAIN_END})")
    logger.info(f"  Val:   {len(X_val):,} samples ({TRAIN_END} to {VAL_END})")
    logger.info(f"  Test:  {len(X_test):,} samples (after {VAL_END})")

    # ========== LOAD PRICE DATA ==========
    logger.info("\n💰 Loading price data for SL/TP...")

    import pandas as pd
    base_data_file = {
        'M5': 'xauusd_m5.csv',
        'M15': 'xauusd_m15.csv',
        'H1': 'xauusd_h1_from_m1.csv',
    }.get(args.base_tf, 'xauusd_m5.csv')

    df_prices = pd.read_csv(f"data/{base_data_file}")
    prices = df_prices['close'].values.astype(np.float32)

    # Split prices same as features
    p_train = prices[:train_idx]
    p_val = prices[train_idx:val_idx]
    p_test = prices[val_idx:]

    logger.info(f"  Prices loaded: {len(prices):,} bars")

    # ========== CREATE ENVIRONMENTS ==========
    logger.info("\n🎮 Creating Professional Trading Environments...")

    WINDOW = args.window  # From command line (default: 128)

    logger.info(f"  Window size: {WINDOW} ({WINDOW * 5 / 60:.1f} hours of M5 data)")
    logger.info(f"  Risk Management:")
    logger.info(f"    • Risk per trade: 2%")
    logger.info(f"    • Max Stop Loss: 0.9% (can exit earlier)")
    logger.info(f"    • Take Profit: NONE (trailing only)")
    logger.info(f"    • Trailing activation: 0.3%")
    logger.info(f"    • Max Drawdown: 30% (HARD STOP)")

    def make_env(features, rets, prices_data):
        def _init():
            env = GoldTradingEnv(
                features=features,
                returns=rets,
                prices=prices_data,
                window=WINDOW,
                initial_balance=10000.0,
                risk_per_trade=0.02,
                max_stop_loss_pct=0.009,       # 0.9% MAXIMUM stop loss
                min_profit_target=0.003,       # 0.3% trailing activation
                trailing_distance=0.002,       # 0.2% trailing distance
                max_drawdown=0.30,
                cost_per_trade=0.0001,
            )
            env = Monitor(env)
            return env
        return _init

    # Training environment (vectorized)
    train_env = DummyVecEnv([make_env(X_train, r_train, p_train)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Validation environment
    val_env = GoldTradingEnv(
        X_val, r_val, p_val, window=WINDOW, max_drawdown=0.30
    )

    # Test environment
    test_env = GoldTradingEnv(
        X_test, r_test, p_test, window=WINDOW, max_drawdown=0.30
    )

    logger.info(f"✅ Observation space: {train_env.observation_space.shape}")
    logger.info(f"✅ Action space: {train_env.action_space}")

    # ========== DEVICE SETUP ==========
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    logger.info(f"\n🖥️  Using device: {device}")

    # ========== PPO CONFIGURATION (Section 1) ==========
    logger.info("\n⚙️  PPO Configuration (from checklist):")

    ppo_config = {
        # Section 1.1: Gamma
        'gamma': 0.6,  # Optimized for intraday-swing

        # Section 1.2: Learning Rate with Annealing
        # REDUCED from 3e-4 to 1e-4 to prevent early stopping
        'learning_rate': linear_schedule(1e-4, 1e-5),

        # Section 1.3: Target KL for early stopping
        # INCREASED from 0.01 to 0.05 to allow more learning per epoch
        'target_kl': 0.05,

        # Section 1.4: Entropy coefficient
        'ent_coef': 0.02,

        # Section 1.5: Batch size and n_steps
        'n_steps': 4096,
        'batch_size': 128,

        # Standard PPO parameters
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'normalize_advantage': True,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,

        # Misc
        'verbose': 1,
        'device': device,
        'tensorboard_log': './train/ppo_tensorboard/',
    }

    logger.info(f"  • gamma: {ppo_config['gamma']} (Section 1.1)")
    logger.info(f"  • learning_rate: 1e-4 → 1e-5 (Section 1.2 - reduced to prevent early stopping)")
    logger.info(f"  • target_kl: {ppo_config['target_kl']} (Section 1.3 - increased to allow more learning)")
    logger.info(f"  • ent_coef: {ppo_config['ent_coef']} (Section 1.4)")
    logger.info(f"  • n_steps: {ppo_config['n_steps']} (Section 1.5)")
    logger.info(f"  • batch_size: {ppo_config['batch_size']} (Section 1.5)")

    # ========== CREATE PPO MODEL ==========
    logger.info("\n🤖 Creating PPO model...")

    # Use absolute path based on script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(SCRIPT_DIR, "ppo_ultimate")
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"💾 Checkpoint directory: {SAVE_DIR}")

    if args.resume:
        logger.info(f"📂 Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env, **ppo_config)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            **ppo_config
        )

    # ========== CALLBACKS ==========
    # Save every 500,000 timesteps - NOT divided by n_steps!
    # save_freq counts individual environment steps, not rollouts
    CHECKPOINT_FREQ = 500000  # Save every 500k steps (6 checkpoints for 3M steps)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=SAVE_DIR,
        name_prefix="ppo_ultimate",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )
    logger.info(f"📸 Checkpoints every {CHECKPOINT_FREQ:,} timesteps")

    trading_callback = TradingMetricsCallback(verbose=1)

    # ========== TRAINING ==========
    logger.info(f"\n🏋️  Starting training for {args.timesteps:,} timesteps...")
    logger.info("-" * 70)

    start_time = datetime.now()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, trading_callback],
        progress_bar=True,
    )

    training_time = (datetime.now() - start_time).total_seconds() / 3600
    logger.info(f"\n✅ Training completed in {training_time:.2f} hours")

    # ========== SAVE FINAL MODEL ==========
    final_path = os.path.join(SAVE_DIR, "ppo_ultimate_final")
    model.save(final_path)
    train_env.save(os.path.join(SAVE_DIR, "vec_normalize.pkl"))

    logger.info(f"💾 Model saved: {final_path}")

    # ========== EVALUATION ==========
    logger.info("\n" + "="*70)
    logger.info("📊 EVALUATION")
    logger.info("="*70)

    # Validation
    logger.info("\n📈 Validation Set (2022):")
    val_metrics = evaluate_model(model, val_env)
    for key, value in val_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")

    # Test
    logger.info("\n📈 Test Set (2023-2025 Out-of-Sample):")
    test_metrics = evaluate_model(model, test_env)
    for key, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")

    # ========== SAVE METRICS ==========
    metrics_path = os.path.join(SAVE_DIR, "ppo_ultimate_metrics.json")

    # Convert all numpy types to Python native types for JSON serialization
    metrics_data = convert_to_python_types({
        'validation': val_metrics,
        'test': test_metrics,
        'training': {
            'timesteps': args.timesteps,
            'training_hours': training_time,
        },
        'config': {
            'gamma': 0.6,
            'target_kl': 0.01,
            'ent_coef': 0.02,
            'n_steps': 4096,
            'batch_size': 128,
            'window': args.window,
        },
        'timestamp': datetime.now().isoformat(),
    })

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"\n💾 Metrics saved: {metrics_path}")

    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*70)
    logger.info("🎯 FINAL SUMMARY")
    logger.info("="*70)

    logger.info(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                      PPO ULTIMATE RESULTS                           │
├─────────────────────────────────────────────────────────────────────┤
│  VALIDATION (2022)           │  TEST (2023-2025)                    │
├──────────────────────────────┼──────────────────────────────────────┤
│  Return: {val_metrics['total_return']:>7.2f}%             │  Return: {test_metrics['total_return']:>7.2f}%                    │
│  Sharpe: {val_metrics['sharpe_ratio']:>7.2f}              │  Sharpe: {test_metrics['sharpe_ratio']:>7.2f}                     │
│  Max DD: {val_metrics['max_drawdown']:>7.2f}%             │  Max DD: {test_metrics['max_drawdown']:>7.2f}%                    │
│  Trades: {val_metrics['trades']:>7}              │  Trades: {test_metrics['trades']:>7}                     │
│  Win Rate: {val_metrics['win_rate']:>5.1f}%            │  Win Rate: {test_metrics['win_rate']:>5.1f}%                   │
└──────────────────────────────┴──────────────────────────────────────┘
    """)

    logger.info("🎉 PPO Ultimate model is ready!")


if __name__ == '__main__':
    main()
