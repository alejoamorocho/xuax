# env/xauusd_env_aggressive.py
"""
XAUUSD Aggressive Trading Environment with Advanced Reward Function

Supports: Long, Short, and Flat positions (3-action)

Reward Components (from PPO Gold Trading Improvements Checklist):
1. Sortino-based return (risk-adjusted, only penalizes downside volatility)
2. Drawdown penalty (quadratic/exponential near limit)
3. Transaction cost penalty
4. Holding winner bonus (incentivizes swing extension)
5. Drawdown breach penalty (hard constraint)
+ Differential Sharpe Ratio (DSR) for online Sharpe optimization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import shared components from main env
from env.xauusd_env import DifferentialSharpeRatio, calculate_advanced_reward


class XAUUSDTradingEnvAggressive(gym.Env):
    """
    "Smart Aggressive" trading env with ADVANCED REWARD FUNCTION.

    Supports: Long (-1), Flat (0), Short (1) positions

    Philosophy:
    - No "participation trophies" (no flat penalty).
    - No "fake math" (no profit multiplier).
    - Realism: You pay the spread/commissions.
    - Advanced reward function with 5 components + DSR

    This forces the agent to only take trades where the expected return > cost.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,             # (T, F)
        returns: np.ndarray,              # (T,)
        window: int = 128,                # WIDER WINDOW: 2 Hours of vision
        cost_per_trade: float = 0.0002,   # 2bps
        max_drawdown: float = 0.30,       # Maximum allowed drawdown (30%)
        leverage: float = 1.0,
        stop_loss_pct: float = 0.001,
        max_episode_steps: int | None = None,
        # Reward settings
        use_advanced_reward: bool = True,
        w_sortino: float = 1.0,
        w_dd_penalty: float = 1.5,
        w_tc_penalty: float = 0.5,
        w_holding_bonus: float = 0.3,
        w_breach_penalty: float = 1.0,
        w_dsr: float = 0.5,
        # Legacy parameters (for backward compatibility)
        turnover_coef: float = 0.0,
        flat_penalty: float = 0.0,
        hold_bonus: float = 0.0,
        **kwargs
    ):
        super().__init__()

        # Input validation
        assert features.ndim == 2
        assert returns.ndim == 1
        assert len(features) == len(returns)

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)

        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.max_drawdown = float(max_drawdown)
        self.leverage = float(leverage)
        self.stop_loss_pct = float(stop_loss_pct)
        self.use_advanced_reward = use_advanced_reward

        # Reward weights
        self.w_sortino = w_sortino
        self.w_dd_penalty = w_dd_penalty
        self.w_tc_penalty = w_tc_penalty
        self.w_holding_bonus = w_holding_bonus
        self.w_breach_penalty = w_breach_penalty
        self.w_dsr = w_dsr

        # Legacy parameters
        self.turnover_coef = float(turnover_coef)
        self.flat_penalty = float(flat_penalty)
        self.hold_bonus_legacy = float(hold_bonus)

        self.T = len(self.r)
        self.max_episode_steps = max_episode_steps

        # Observation: window features + current position (-1, 0, or 1)
        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 0=Short, 1=Flat, 2=Long
        self.action_space = spaces.Discrete(3)

        # DSR calculator
        self.dsr = DifferentialSharpeRatio(eta=0.01)

        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.pos = 0  # -1 (short), 0 (flat), or 1 (long)
        self.entry_price = 1.0  # Virtual entry price
        self.steps = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.position_entry_step = None
        self.unrealized_pnl = 0.0
        self.returns_history = []

        # Reset DSR
        self.dsr.reset()

    def _get_obs(self):
        # Slice window
        w = self.X[self.t - self.window : self.t]

        # Flatten and append position
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        self.peak_equity = max(self.peak_equity, self.equity)
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
        else:
            drawdown = 0.0
        return drawdown

    def _get_position_duration(self) -> int:
        """How many steps the position has been open"""
        if self.position_entry_step is None:
            return 0
        return self.t - self.position_entry_step

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        # 1. Map action (0,1,2) -> position (-1, 0, 1)
        new_pos = int(action) - 1

        # 2. Calculate costs
        delta = abs(new_pos - self.pos)
        trade_cost = self.cost * delta

        # 3. Calculate PnL
        # r[t] is log return (~pct change).
        # approximate price change = r[t]
        raw_pnl = self.pos * self.r[self.t]
        step_return = raw_pnl * self.leverage

        # 4. Check Stop Loss
        sl_penalty = 0.0
        sl_triggered = False

        if self.pos != 0:
            # If long and price dropped > SL
            if self.pos == 1 and self.r[self.t] < -self.stop_loss_pct:
                sl_penalty = -0.05  # Heavy penalty for hitting SL
                new_pos = 0  # Force Close
                sl_triggered = True

            # If short and price rose > SL
            elif self.pos == -1 and self.r[self.t] > self.stop_loss_pct:
                sl_penalty = -0.05
                new_pos = 0
                sl_triggered = True

        # Track position entry/exit
        if delta > 0:
            if new_pos != 0:
                # Entering new position
                self.position_entry_step = self.t
                self.unrealized_pnl = 0.0
            else:
                # Exiting position
                self.position_entry_step = None
                self.unrealized_pnl = 0.0

        # Update unrealized PnL if in position
        if self.pos != 0:
            self.unrealized_pnl += step_return

        # Store return in history
        self.returns_history.append(step_return)

        # Keep only last 100 returns for downside calculation
        if len(self.returns_history) > 100:
            self.returns_history = self.returns_history[-100:]

        # 5. Calculate reward
        if self.use_advanced_reward:
            # Calculate DSR
            dsr_value = self.dsr.calculate(step_return)

            # Get current drawdown
            current_dd = self._get_current_drawdown()

            # Get position duration
            position_duration = self._get_position_duration()

            # Is current position a winner?
            is_winner = self.unrealized_pnl > 0

            # Calculate advanced reward
            reward, reward_components = calculate_advanced_reward(
                portfolio_return=step_return,
                current_drawdown=current_dd,
                downside_returns=np.array(self.returns_history),
                position_duration=position_duration,
                is_winner=is_winner,
                transaction_cost=trade_cost,
                max_drawdown=self.max_drawdown,
                dsr_value=dsr_value,
                w_sortino=self.w_sortino,
                w_dd_penalty=self.w_dd_penalty,
                w_tc_penalty=self.w_tc_penalty,
                w_holding_bonus=self.w_holding_bonus,
                w_breach_penalty=self.w_breach_penalty,
                w_dsr=self.w_dsr,
            )

            # Add stop loss penalty
            reward += sl_penalty

        else:
            # Legacy reward function (for backward compatibility)
            turnover_pen = self.turnover_coef * delta
            flat_pen = self.flat_penalty if new_pos == 0 else 0.0
            hold_bon = self.hold_bonus_legacy if delta == 0 else 0.0
            reward = step_return - trade_cost - turnover_pen - flat_pen + hold_bon + sl_penalty
            reward_components = {}

        # 6. Update State
        self.equity *= (1.0 + step_return - trade_cost)
        self.pos = new_pos
        self.t += 1
        self.steps += 1

        # Check termination
        terminated = self.t >= self.T
        truncated = sl_triggered  # End episode on stop loss hit

        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        # Get current drawdown for info
        current_dd = self._get_current_drawdown()

        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
            "trade_cost": float(trade_cost),
            "pnl": float(raw_pnl),
            "drawdown": float(current_dd),
            "peak_equity": float(self.peak_equity),
            "step_return": float(step_return),
            "unrealized_pnl": float(self.unrealized_pnl),
            "position_duration": int(self._get_position_duration()),
            "sl_triggered": sl_triggered,
        }

        # Add reward components to info if using advanced reward
        if self.use_advanced_reward and reward_components:
            info["reward_components"] = reward_components

        return self._get_obs(), float(reward), terminated, truncated, info
