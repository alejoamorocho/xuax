import os
import numpy as np
import multiprocessing as mp
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from features.make_features import make_features
from env.xauusd_env_aggressive import XAUUSDTradingEnvAggressive


# ============================================================================
# Learning Rate Schedule (Annealing)
# ============================================================================
def linear_schedule(initial_lr: float = 3e-4, final_lr: float = 1e-5) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    - Inicio (LR alto = 3e-4): Exploración agresiva del espacio de políticas
    - Final (LR bajo = 1e-5): Fine-tuning sin destruir lo aprendido
    """
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


WINDOW = 120   # 1 Week of H1 Data (5 days * 24h)
COST = 0.0001    # 1bp
N_ENVS = 16      

TRAIN_END_DATE = "2024-01-01"

# Training schedule - H1 MACRO VERSION
CHUNK_STEPS = 50_000
N_CHUNKS = 10  # 500,000 steps

SAVE_DIR = "train"
SAVE_PREFIX = "ppo_xauusd_macro"


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Use H1 MACRO data (Gold + DXY + SPX + Yields)
    df, X, r = make_features("data/xauusd_1h_macro.csv", window=WINDOW)

    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(TRAIN_END_DATE))
    X_train, r_train = X[:train_end], r[:train_end]
    X_test, r_test = X[train_end:], r[train_end:]

    def make_train_env():
        return XAUUSDTradingEnvAggressive(
            X_train,
            r_train,
            window=WINDOW,
            cost_per_trade=COST,
            turnover_coef=0.0,    # Swing Trading
            leverage=1.0,         
            stop_loss_pct=0.0,    
            max_episode_steps=5_000,
        )

    # Parallel envs
    train_env = SubprocVecEnv([make_train_env for _ in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        # === CRÍTICOS - OPTIMIZADOS PARA GOLD TRADING ===
        gamma=0.6,                                  # ✅ 1.1: Optimizado para intraday-swing (era 0.99)
        learning_rate=linear_schedule(3e-4, 1e-5), # ✅ 1.2: LR annealing (era 3e-4 constante)
        target_kl=0.01,                            # ✅ 1.3: Early stopping si KL divergence > 0.01
        ent_coef=0.02,                             # ✅ 1.4: Más exploración (era 0.01)
        # === BATCH/STEPS OPTIMIZADOS ===
        n_steps=4096,                              # ✅ 1.5: Más contexto temporal (era 2048)
        batch_size=128,                            # ✅ 1.5: Gradientes más estables (era 512)
        n_epochs=10,                               # Epochs por update
        # === ESTÁNDAR PPO ===
        gae_lambda=0.95,                           # GAE para advantage estimation
        clip_range=0.2,                            # Clipping estándar
        normalize_advantage=True,                  # Normalizar advantages
        vf_coef=0.5,                               # Value function coefficient
        max_grad_norm=0.5,                         # Gradient clipping
    )

    print("\n" + "="*60)
    print("🚀 TRAINING H1 MACRO-AWARE AGENT (GOD MODE PHASE 1)")
    print("="*60)
    print("Features:")
    print("  ✅ Gold Price Action")
    print("  ✅ DXY (Dollar Index) - The Currency Filter")
    print("  ✅ SPX (S&P 500) - The Risk Filter")
    print("  ✅ US10Y (Bond Yields) - The Interest Rate Filter")
    print(f"  ✅ Training for {N_CHUNKS * CHUNK_STEPS:,} steps...")
    print("="*60 + "\n")

    total = 0
    for i in range(N_CHUNKS):
        print(f"\n📊 Chunk {i+1}/{N_CHUNKS} ({total:,} steps completed)")
        model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
        total += CHUNK_STEPS

        ckpt_path = f"{SAVE_DIR}/{SAVE_PREFIX}_{total//1000}k"
        model.save(ckpt_path)
        print(f"✅ Saved checkpoint: {ckpt_path}.zip")

    # Save latest
    model.save(f"{SAVE_DIR}/{SAVE_PREFIX}_latest")
    print(f"\n✅ Saved latest: {SAVE_DIR}/{SAVE_PREFIX}_latest.zip\n")

    # Quick test on out-of-sample data
    test_env = XAUUSDTradingEnvAggressive(
        X_test, r_test,
        window=WINDOW,
        cost_per_trade=COST,
        max_episode_steps=None
    )
    obs, _ = test_env.reset()
    equities, positions = [], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action)
        equities.append(info["equity"])
        positions.append(info["pos"])
        if term or trunc:
            break

    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pct_time_long = float(np.mean(np.array(positions) == 1))
    pct_time_short = float(np.mean(np.array(positions) == -1))
    pct_time_flat = float(np.mean(np.array(positions) == 0))

    print("\n" + "="*60)
    print("📊 QUICK TEST RESULTS (Out-of-Sample)")
    print("="*60)
    print(f"Final Equity:     {float(equities[-1]):.4f}x")
    print(f"Total Return:     {(float(equities[-1])-1)*100:.2f}%")
    print(f"Total Trades:     {trades}")
    print(f"% Time Long:      {pct_time_long*100:.1f}%")
    print(f"% Time Short:     {pct_time_short*100:.1f}%")
    print(f"% Time Flat:      {pct_time_flat*100:.1f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
