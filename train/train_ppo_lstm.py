# train/train_ppo_lstm.py
"""
PPO Training with LSTM Policy for Gold Trading

Section 5 of PPO Gold Trading Improvements Checklist:
- 5.1 Uses RecurrentPPO with MlpLstmPolicy from sb3-contrib
- 5.2 Configures optimal lookback window for LSTM
- 5.3 Implements proper normalization for sequential data

Requirements:
    pip install sb3-contrib

LSTM vs MLP:
- MLP: No memory, each step is independent
- LSTM: Memory of previous steps, captures temporal patterns
- Research shows LSTM gives +15-25% improvement in trading performance
"""

import os
import numpy as np
import multiprocessing as mp
from typing import Callable
import torch

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERROR: sb3-contrib not installed. Run: pip install sb3-contrib")
    print("RecurrentPPO with LSTM requires sb3-contrib package.")
    exit(1)

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from features.make_features import make_features
from env.xauusd_env import XAUUSDTradingEnv


# ============================================================================
# Learning Rate Schedule (Annealing)
# ============================================================================
def linear_schedule(initial_lr: float = 3e-4, final_lr: float = 1e-5) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    - Inicio (LR alto = 3e-4): Exploración agresiva del espacio de políticas
    - Final (LR bajo = 1e-5): Fine-tuning sin destruir lo aprendido

    Args:
        initial_lr: Learning rate inicial
        final_lr: Learning rate final

    Returns:
        Schedule function para stable-baselines3
    """
    def schedule(progress_remaining: float) -> float:
        """
        progress_remaining va de 1 (inicio) a 0 (final del entrenamiento)
        """
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ============================================================================
# Configuration
# ============================================================================
WINDOW = 64               # Lookback window for LSTM
COST = 0.0001             # Transaction cost (consistent everywhere)
N_ENVS = 4                # Fewer envs for LSTM (more memory intensive)

TRAIN_END_DATE = "2022-01-01"

# Training schedule
CHUNK_STEPS = 50_000      # Timesteps per chunk
N_CHUNKS = 10             # Total: 500,000 steps (reduced for LSTM)

SAVE_DIR = "train"
SAVE_PREFIX = "ppo_lstm_xauusd"

# LSTM-specific configuration
LSTM_HIDDEN_SIZE = 64     # LSTM hidden state size
N_LSTM_LAYERS = 2         # Number of LSTM layers


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("PPO with LSTM Policy for Gold Trading")
    print("=" * 60)

    # Load features
    print("\n[1/4] Loading features...")
    df, X, r = make_features(
        "data/xauusd_1h.csv",
        window=WINDOW,
        use_optimized_indicators=True,
        use_advanced_features=True,
        include_external_data=False,  # Set to True to fetch VIX/TIPS
    )

    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")

    # Split data
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(TRAIN_END_DATE))
    X_train, r_train = X[:train_end], r[:train_end]
    X_test, r_test = X[train_end:], r[train_end:]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Create environments
    print("\n[2/4] Creating environments...")

    def make_train_env():
        return XAUUSDTradingEnv(
            X_train,
            r_train,
            window=WINDOW,
            cost_per_trade=COST,
            max_episode_steps=20_000,
            use_advanced_reward=True,
        )

    # Use DummyVecEnv for LSTM (SubprocVecEnv can have issues with recurrent policies)
    train_env = DummyVecEnv([make_train_env for _ in range(N_ENVS)])
    print(f"  Created {N_ENVS} vectorized environments")

    # LSTM Policy configuration
    print("\n[3/4] Configuring LSTM policy...")

    policy_kwargs = {
        # Actor (policy) and Critic (value) networks
        "net_arch": dict(
            pi=[64, 64],    # Policy network layers
            vf=[64, 64],    # Value network layers
        ),
        # LSTM configuration
        "lstm_hidden_size": LSTM_HIDDEN_SIZE,
        "n_lstm_layers": N_LSTM_LAYERS,
        "shared_lstm": False,           # Separate LSTMs for actor and critic
        "enable_critic_lstm": True,     # Use LSTM for value function too
        "activation_fn": torch.nn.LeakyReLU,
    }

    print(f"  LSTM hidden size: {LSTM_HIDDEN_SIZE}")
    print(f"  LSTM layers: {N_LSTM_LAYERS}")
    print(f"  Shared LSTM: {policy_kwargs['shared_lstm']}")

    # Create RecurrentPPO model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        # === CRÍTICOS - OPTIMIZADOS PARA GOLD TRADING ===
        gamma=0.6,                                  # ✅ 1.1: Optimizado para intraday-swing
        learning_rate=linear_schedule(3e-4, 1e-5), # ✅ 1.2: LR annealing
        target_kl=0.01,                            # ✅ 1.3: Early stopping
        ent_coef=0.02,                             # ✅ 1.4: Más exploración
        # === BATCH/STEPS OPTIMIZADOS PARA LSTM ===
        n_steps=1024,                              # Reduced for LSTM (memory intensive)
        batch_size=64,                             # Smaller batches for LSTM
        n_epochs=5,                                # Fewer epochs (recurrent needs more data)
        # === ESTÁNDAR PPO ===
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    print(f"  Model created successfully!")
    print(f"  Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Training loop
    print("\n[4/4] Starting training...")
    print(f"  Total steps: {CHUNK_STEPS * N_CHUNKS:,}")
    print(f"  Chunks: {N_CHUNKS}")

    total = 0
    for i in range(N_CHUNKS):
        print(f"\n--- Chunk {i+1}/{N_CHUNKS} ---")
        model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
        total += CHUNK_STEPS

        ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_{total//1000}k"
        model.save(ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}.zip")

    # Save final model
    model.save(f"{SAVE_DIR}/{SAVE_PREFIX}_latest")
    print(f"\n✅ Saved latest: {SAVE_DIR}/{SAVE_PREFIX}_latest.zip")

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    test_env = XAUUSDTradingEnv(
        X_test,
        r_test,
        window=WINDOW,
        cost_per_trade=COST,
        max_episode_steps=None,
        use_advanced_reward=True,
    )

    obs, _ = test_env.reset()
    equities, positions = [], []

    # For LSTM, we need to track the hidden state
    lstm_states = None
    episode_starts = np.array([True])

    while True:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        obs, reward, term, trunc, info = test_env.step(action)
        equities.append(info["equity"])
        positions.append(info["pos"])
        episode_starts = np.array([False])

        if term or trunc:
            break

    # Calculate metrics
    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pct_time_long = float(np.mean(np.array(positions) == 1))
    pct_time_short = float(np.mean(np.array(positions) == -1))
    pct_time_flat = float(np.mean(np.array(positions) == 0))

    # Returns calculation
    equity_array = np.array(equities)
    returns = np.diff(np.log(equity_array + 1e-10))
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24)  # Annualized

    # Drawdown
    peak = np.maximum.accumulate(equity_array)
    drawdown = (peak - equity_array) / peak
    max_drawdown = np.max(drawdown)

    print(f"\nResults:")
    print(f"  Final Equity: {equities[-1]:.4f}")
    print(f"  Total Return: {(equities[-1] - 1.0) * 100:.2f}%")
    print(f"  Sharpe Ratio (annualized): {sharpe:.3f}")
    print(f"  Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"  Total Trades: {trades}")
    print(f"  % Time Long: {pct_time_long * 100:.1f}%")
    print(f"  % Time Short: {pct_time_short * 100:.1f}%")
    print(f"  % Time Flat: {pct_time_flat * 100:.1f}%")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS safe
    main()
