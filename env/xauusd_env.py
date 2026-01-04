# env/xauusd_env.py
"""
XAUUSD Trading Environment with Advanced Reward Function

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


# ============================================================================
# Differential Sharpe Ratio (Moody & Saffell, 2001)
# ============================================================================
class DifferentialSharpeRatio:
    """
    Differential Sharpe Ratio for online Sharpe optimization.

    Allows optimizing Sharpe ratio with dense rewards (signal at each step)
    instead of waiting until end of episode.

    Reference: Moody & Saffell (2001) - "Learning to Trade via Direct Reinforcement"
    """

    def __init__(self, eta: float = 0.01):
        """
        Args:
            eta: Decay rate for EMAs (0.01 = ~100 effective periods)
        """
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of returns^2
        self.initialized = False

    def reset(self):
        """Call at the start of each episode"""
        self.A = 0.0
        self.B = 0.0
        self.initialized = False

    def calculate(self, R_t: float) -> float:
        """
        Calculate the Differential Sharpe Ratio for current return.

        Args:
            R_t: Return of current step

        Returns:
            D_t: Differential Sharpe Ratio
        """
        if not self.initialized:
            # First step: initialize with current values
            self.A = R_t
            self.B = R_t ** 2
            self.initialized = True
            return 0.0  # No DSR on first step

        # Calculate deltas
        delta_A = R_t - self.A
        delta_B = R_t ** 2 - self.B

        # Calculate DSR
        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = (self.B - self.A ** 2) ** 1.5

        if abs(denominator) < 1e-10:
            D_t = 0.0
        else:
            D_t = numerator / denominator

        # Update EMAs
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B

        return D_t


# ============================================================================
# Advanced Reward Calculator
# ============================================================================
def calculate_advanced_reward(
    portfolio_return: float,
    current_drawdown: float,
    downside_returns: np.ndarray,
    position_duration: int,
    is_winner: bool,
    transaction_cost: float,
    max_drawdown: float = 0.30,
    dsr_value: float = 0.0,
    # Weights for each component
    w_sortino: float = 1.0,
    w_dd_penalty: float = 1.5,
    w_tc_penalty: float = 0.5,
    w_holding_bonus: float = 0.3,
    w_breach_penalty: float = 1.0,
    w_dsr: float = 0.5,
) -> tuple[float, dict]:
    """
    Advanced reward function with 5 components + DSR.

    Components:
    1. Sortino-based return (risk-adjusted)
    2. Drawdown penalty (quadratic/exponential)
    3. Transaction cost penalty
    4. Holding winner bonus
    5. Drawdown breach penalty (hard constraint)
    + DSR (Differential Sharpe Ratio)

    Returns:
        reward: Total reward
        components: Dict with individual component values
    """

    # =========================================
    # COMPONENT 1: Sortino-based Return
    # =========================================
    # Only penalizes negative volatility (downside)
    if len(downside_returns) > 0:
        negative_returns = downside_returns[downside_returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.sqrt(np.mean(negative_returns ** 2))
        else:
            downside_std = 1e-8
    else:
        downside_std = 1e-8

    sortino_component = portfolio_return / (downside_std + 1e-8)

    # =========================================
    # COMPONENT 2: Drawdown Penalty
    # =========================================
    # Quadratic penalty that increases exponentially near the limit
    dd_ratio = current_drawdown / max_drawdown if max_drawdown > 0 else 0.0

    if dd_ratio < 0.5:
        # Less than 50% of limit used: no penalty
        dd_penalty = 0.0
    elif dd_ratio < 0.8:
        # 50-80% of limit: soft quadratic penalty
        dd_penalty = -0.5 * (dd_ratio - 0.5) ** 2
    elif dd_ratio < 1.0:
        # 80-100% of limit: strong quadratic penalty
        dd_penalty = -2.0 * (dd_ratio - 0.5) ** 2
    else:
        # Exceeded limit: extreme penalty
        dd_penalty = -10.0 - 5.0 * (dd_ratio - 1.0)

    # =========================================
    # COMPONENT 3: Transaction Cost Penalty
    # =========================================
    tc_penalty = -transaction_cost * 100  # Scaled to be significant

    # =========================================
    # COMPONENT 4: Holding Winner Bonus
    # =========================================
    # Incentivizes holding winning positions (for swing extension)
    if is_winner and portfolio_return > 0:
        # Log bonus: grows fast at start, flattens later
        holding_bonus = 0.1 * np.log1p(position_duration)
    else:
        holding_bonus = 0.0

    # =========================================
    # COMPONENT 5: Drawdown Breach Penalty (Hard Constraint)
    # =========================================
    if current_drawdown > max_drawdown:
        breach_penalty = -50.0  # Very strong penalty
    else:
        breach_penalty = 0.0

    # =========================================
    # COMPONENT 6: DSR (Differential Sharpe Ratio)
    # =========================================
    dsr_component = dsr_value

    # =========================================
    # FINAL REWARD
    # =========================================
    reward = (
        w_sortino * sortino_component +
        w_dd_penalty * dd_penalty +
        w_tc_penalty * tc_penalty +
        w_holding_bonus * holding_bonus +
        w_breach_penalty * breach_penalty +
        w_dsr * dsr_component
    )

    components = {
        'sortino': sortino_component,
        'dd_penalty': dd_penalty,
        'tc_penalty': tc_penalty,
        'holding_bonus': holding_bonus,
        'breach_penalty': breach_penalty,
        'dsr': dsr_component,
        'dd_ratio': dd_ratio,
    }

    return reward, components


# ============================================================================
# Main Trading Environment
# ============================================================================
class XAUUSDTradingEnv(gym.Env):
    """
    Discrete long-only trading env with ADVANCED REWARD FUNCTION.

    Actions:
      0 = Flat
      1 = Long

    Position is applied on the NEXT step to avoid look-ahead.

    Reward Components:
      1. Sortino-based return (risk-adjusted)
      2. Drawdown penalty (quadratic near limit)
      3. Transaction cost penalty
      4. Holding winner bonus
      5. Drawdown breach penalty
      + Differential Sharpe Ratio (DSR)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,             # (T, F)
        returns: np.ndarray,              # (T,)
        window: int = 64,
        cost_per_trade: float = 0.0001,   # cost per unit position change
        max_drawdown: float = 0.30,       # Maximum allowed drawdown (30%)
        max_episode_steps: int | None = None,
        # Reward weights
        use_advanced_reward: bool = True, # Use new reward function
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
    ):
        super().__init__()
        assert features.ndim == 2
        assert returns.ndim == 1
        assert len(features) == len(returns)

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)

        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.max_drawdown = float(max_drawdown)
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

        obs_dim = self.window * self.X.shape[1] + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Actions: 0=Flat, 1=Long
        self.action_space = spaces.Discrete(2)

        # DSR calculator
        self.dsr = DifferentialSharpeRatio(eta=0.01)

        self._reset_state()

    def _reset_state(self):
        self.t = self.window
        self.pos = 0  # 0 or 1
        self.steps = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.position_entry_step = None
        self.unrealized_pnl = 0.0
        self.returns_history = []

        # Reset DSR
        self.dsr.reset()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]  # (window, F)
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
        action = int(action)
        new_pos = 1 if action == 1 else 0

        # Position change magnitude (0 or 1)
        delta = abs(new_pos - self.pos)

        # Transaction cost
        trade_cost = self.cost * delta

        # PnL from holding PREVIOUS position over this bar
        step_return = self.pos * self.r[self.t]

        # Track position entry/exit
        if delta > 0:
            if new_pos == 1:
                # Entering new position
                self.position_entry_step = self.t
                self.unrealized_pnl = 0.0
            else:
                # Exiting position
                self.position_entry_step = None
                self.unrealized_pnl = 0.0

        # Update unrealized PnL if in position
        if self.pos == 1:
            self.unrealized_pnl += step_return

        # Store return in history
        self.returns_history.append(step_return)

        # Keep only last 100 returns for downside calculation
        if len(self.returns_history) > 100:
            self.returns_history = self.returns_history[-100:]

        # Calculate reward
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
        else:
            # Legacy reward function (for backward compatibility)
            turnover_penalty = self.turnover_coef * delta
            flat_pen = self.flat_penalty if new_pos == 0 else 0.0
            hold_bonus = self.hold_bonus_legacy if delta == 0 else 0.0
            reward = step_return - trade_cost - turnover_penalty - flat_pen + hold_bonus
            reward_components = {}

        # Track equity
        self.equity *= (1.0 + step_return - trade_cost)

        # Update position after reward (avoid look-ahead)
        self.pos = new_pos

        # Advance time
        self.t += 1
        self.steps += 1

        # Check termination
        terminated = self.t >= self.T
        truncated = False
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            truncated = True

        # Also terminate if max drawdown breached (optional hard stop)
        current_dd = self._get_current_drawdown()
        if current_dd > self.max_drawdown:
            # Don't terminate, but the penalty is severe
            pass

        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
            "trade_cost": float(trade_cost),
            "drawdown": float(current_dd),
            "peak_equity": float(self.peak_equity),
            "step_return": float(step_return),
            "unrealized_pnl": float(self.unrealized_pnl),
            "position_duration": int(self._get_position_duration()),
        }

        # Add reward components to info if using advanced reward
        if self.use_advanced_reward and reward_components:
            info["reward_components"] = reward_components

        return self._get_obs(), float(reward), terminated, truncated, info
