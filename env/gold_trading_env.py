"""
Gold Trading Environment - Professional Risk Management
========================================================

Core Philosophy: "Let winners run, cut losers short"

Key Parameters:
- Risk per trade: 2% of account
- Stop Loss: MAXIMUM 0.9% (can exit earlier if trade looks bad)
- NO fixed Take Profit (let winners run with trailing)
- Trailing activation: 0.3% profit (1:1 RR with min target)
- Max Drawdown: 30% (stops training if reached)

Multiple positions allowed when existing trades are in profit > 0.3%

Rewards:
- Maximum bonus for closing at peak profit
- Bonus for cutting losers quickly
- Penalty for closing winners too early
- Capital preservation is paramount
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GoldTradingEnv(gym.Env):
    """
    Professional Gold Trading Environment v2.0

    Philosophy: Let winners run, cut losers short.

    NEW in v2.0:
    - Hold Bonus: Reward for holding winning positions longer
    - Trade Frequency Penalty: Penalize over-trading
    - Quality over Quantity: Better reward shaping

    Actions:
        0 = Stay flat / Close position
        1 = Open/Hold long position

    Risk Management:
        - Stop Loss: Maximum 0.9% (can exit earlier)
        - No Take Profit (trailing only)
        - Trailing starts at 0.3% profit
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        prices: np.ndarray = None,
        window: int = 128,
        initial_balance: float = 10000.0,
        # Risk Management
        risk_per_trade: float = 0.02,       # 2% risk per trade
        max_stop_loss_pct: float = 0.009,   # 0.9% MAXIMUM stop loss
        min_profit_target: float = 0.003,   # 0.3% minimum target (trailing activation)
        trailing_distance: float = 0.002,   # 0.2% trailing distance
        max_drawdown: float = 0.30,         # 30% max drawdown - HARD STOP
        cost_per_trade: float = 0.0001,     # Transaction cost
        # Multiple positions
        allow_multiple_positions: bool = True,
        max_positions: int = 3,
        # NEW: Anti-overtrading parameters
        max_trades_per_day: int = 8,        # Max trades per day (288 bars @ M5)
        trade_cooldown: int = 12,           # Min bars between trades (1 hour @ M5)
    ):
        super().__init__()

        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.prices = prices if prices is not None else self._reconstruct_prices(returns)

        self.window = int(window)
        self.initial_balance = float(initial_balance)
        self.T = len(self.r)

        # Risk Management Parameters
        self.risk_per_trade = risk_per_trade
        self.max_stop_loss_pct = max_stop_loss_pct  # MAXIMUM SL (can exit earlier)
        self.min_profit_target = min_profit_target  # Trailing activation point
        self.trailing_distance = trailing_distance
        self.max_drawdown = max_drawdown
        self.cost_per_trade = cost_per_trade

        # Multiple positions
        self.allow_multiple_positions = allow_multiple_positions
        self.max_positions = max_positions

        # NEW: Anti-overtrading parameters
        self.max_trades_per_day = max_trades_per_day
        self.trade_cooldown = trade_cooldown
        self.bars_per_day = 288  # M5 timeframe

        # Observation space: window features + position state (8 features)
        obs_dim = self.window * self.X.shape[1] + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 0=Flat/Close, 1=Long/Hold
        self.action_space = spaces.Discrete(2)

        self._reset_state()

    def _reconstruct_prices(self, returns):
        """Reconstruct prices from returns if not provided."""
        prices = np.zeros(len(returns) + 1)
        prices[0] = 1800.0  # Starting gold price estimate
        for i, r in enumerate(returns):
            prices[i + 1] = prices[i] * (1 + r)
        return prices[1:].astype(np.float32)

    def _reset_state(self):
        """Reset all state variables."""
        self.t = self.window
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance

        # Active positions (list of dicts)
        self.positions = []

        # Statistics
        self.trades_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        # For evaluation
        self.equity_curve = [self.initial_balance]
        self.positions_count_history = [0]
        self.returns_history = []

        # Episode flags
        self.hard_stop_triggered = False

        # NEW: Trade frequency tracking
        self.recent_trade_steps = []  # Steps when trades were opened
        self.last_trade_step = -999   # Last trade step (for cooldown)
        self.daily_trades = 0         # Trades today
        self.current_day = 0          # Track current day

        # NEW: Hold tracking for winners
        self.consecutive_hold_steps = 0  # Steps holding same position
        self.last_action = None          # Previous action

    def _get_current_price(self):
        """Get current price safely."""
        if self.t < len(self.prices):
            return self.prices[self.t]
        return self.prices[-1]

    def _get_obs(self):
        """Get observation with position state."""
        # Market features
        w = self.X[self.t - self.window : self.t]

        # Position state features
        num_positions = len(self.positions)
        total_unrealized_pnl = sum(p['unrealized_pnl_pct'] for p in self.positions) if self.positions else 0
        avg_unrealized_pnl = total_unrealized_pnl / max(1, num_positions)
        best_position_pnl = max((p['unrealized_pnl_pct'] for p in self.positions), default=0)
        worst_position_pnl = min((p['unrealized_pnl_pct'] for p in self.positions), default=0)
        any_trailing_active = float(any(p.get('trailing_active', False) for p in self.positions))
        drawdown = self._get_current_drawdown()
        capital_ratio = self.balance / self.initial_balance

        position_state = np.array([
            float(num_positions) / self.max_positions,  # Normalized position count
            avg_unrealized_pnl * 10,                    # Scaled avg PnL
            best_position_pnl * 10,                     # Best position
            worst_position_pnl * 10,                    # Worst position
            any_trailing_active,                        # Trailing active flag
            drawdown,                                   # Current drawdown
            capital_ratio,                              # Capital preservation
            float(num_positions > 0),                   # Has position flag
        ], dtype=np.float32)

        obs = np.concatenate([w.reshape(-1), position_state])
        return obs.astype(np.float32)

    def _get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        current_equity = self._get_total_equity()
        self.peak_balance = max(self.peak_balance, current_equity)
        if self.peak_balance > 0:
            return (self.peak_balance - current_equity) / self.peak_balance
        return 0.0

    def _get_total_equity(self) -> float:
        """Get total equity including unrealized PnL."""
        unrealized = sum(p.get('unrealized_pnl', 0) for p in self.positions)
        return self.balance + unrealized

    def _can_open_new_position(self) -> bool:
        """Check if we can open a new position."""
        if not self.allow_multiple_positions:
            return len(self.positions) == 0

        # Can open if under max positions
        if len(self.positions) >= self.max_positions:
            return False

        # If we have existing positions, they must be in profit > min target
        if self.positions:
            all_profitable = all(
                p['unrealized_pnl_pct'] >= self.min_profit_target
                for p in self.positions
            )
            if not all_profitable:
                return False

        return True

    def _open_position(self):
        """Open a new long position."""
        current_price = self._get_current_price()

        # Calculate position size based on risk (2% of current balance)
        risk_amount = self.balance * self.risk_per_trade
        sl_distance = current_price * self.max_stop_loss_pct
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0

        position = {
            'entry_price': current_price,
            'entry_step': self.t,
            'size': position_size,
            'stop_loss_price': current_price * (1 - self.max_stop_loss_pct),
            'trailing_active': False,
            'trailing_stop_price': None,
            'max_price': current_price,
            'max_unrealized_pnl_pct': 0.0,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0,
        }

        self.positions.append(position)
        self.total_trades += 1

        # Transaction cost
        cost = self.balance * self.cost_per_trade
        self.balance -= cost

    def _update_positions(self):
        """Update all positions with current price."""
        current_price = self._get_current_price()

        for pos in self.positions:
            # Update unrealized PnL
            pos['unrealized_pnl_pct'] = (current_price - pos['entry_price']) / pos['entry_price']
            pos['unrealized_pnl'] = pos['size'] * (current_price - pos['entry_price'])

            # Track maximum price and PnL
            if current_price > pos['max_price']:
                pos['max_price'] = current_price
            if pos['unrealized_pnl_pct'] > pos['max_unrealized_pnl_pct']:
                pos['max_unrealized_pnl_pct'] = pos['unrealized_pnl_pct']

            # Activate trailing after reaching min profit target (0.3%)
            if not pos['trailing_active'] and pos['unrealized_pnl_pct'] >= self.min_profit_target:
                pos['trailing_active'] = True
                # Move stop to breakeven + small profit (0.1%)
                pos['trailing_stop_price'] = pos['entry_price'] * 1.001

            # Update trailing stop (only moves up, never down)
            if pos['trailing_active']:
                new_trailing = current_price * (1 - self.trailing_distance)
                if pos['trailing_stop_price'] is None or new_trailing > pos['trailing_stop_price']:
                    pos['trailing_stop_price'] = new_trailing

    def _check_auto_exits(self) -> list:
        """Check and execute automatic exits (SL, Trailing)."""
        current_price = self._get_current_price()
        closed_trades = []

        positions_to_close = []
        for i, pos in enumerate(self.positions):
            exit_reason = None

            # Check MAX stop loss (1%)
            if current_price <= pos['stop_loss_price']:
                exit_reason = "stop_loss"

            # Check trailing stop
            elif pos['trailing_active'] and pos['trailing_stop_price']:
                if current_price <= pos['trailing_stop_price']:
                    exit_reason = "trailing_stop"

            if exit_reason:
                positions_to_close.append((i, exit_reason))

        # Close positions in reverse order to maintain indices
        for i, reason in reversed(positions_to_close):
            trade_info = self._close_position(i, reason)
            closed_trades.append(trade_info)

        return closed_trades

    def _close_position(self, position_idx: int, reason: str = "manual") -> dict:
        """Close a specific position."""
        if position_idx >= len(self.positions):
            return {'pnl': 0, 'reason': 'invalid_index'}

        pos = self.positions[position_idx]
        current_price = self._get_current_price()

        # Calculate final PnL
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        pnl = pos['size'] * (current_price - pos['entry_price'])

        # Calculate efficiency: how close did we close to the maximum?
        if pos['max_unrealized_pnl_pct'] > 0:
            efficiency = pnl_pct / pos['max_unrealized_pnl_pct'] if pos['max_unrealized_pnl_pct'] > 0 else 0
        else:
            efficiency = 1.0 if pnl_pct >= 0 else 0.0

        # Update balance
        self.balance += pnl

        # Transaction cost
        cost = self.balance * self.cost_per_trade
        self.balance -= cost

        # Track statistics
        if pnl > 0:
            self.winning_trades += 1
        self.total_pnl += pnl

        # Record trade
        trade_info = {
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'max_pnl_pct': pos['max_unrealized_pnl_pct'],
            'efficiency': efficiency,
            'duration': self.t - pos['entry_step'],
            'reason': reason,
            'trailing_was_active': pos['trailing_active'],
        }
        self.trades_history.append(trade_info)

        # Remove position
        self.positions.pop(position_idx)

        return trade_info

    def _close_all_positions(self, reason: str = "manual") -> list:
        """Close all open positions."""
        closed_trades = []
        while self.positions:
            trade_info = self._close_position(0, reason)
            closed_trades.append(trade_info)
        return closed_trades

    def _calculate_reward(self, closed_trades: list, action: int, opened_trade: bool = False) -> float:
        """
        Calculate reward based on professional trading principles v2.0

        NEW Features:
        1. HOLD BONUS: Reward for holding winning positions longer
        2. TRADE FREQUENCY PENALTY: Penalize over-trading
        3. QUALITY OVER QUANTITY: Better reward shaping

        Key principles:
        1. Let winners run (penalize early close of winners)
        2. Cut losers short (reward quick exit of losers)
        3. Maximize profit capture (reward closing near peak)
        4. Preserve capital
        5. Trade less, but better
        """
        reward = 0.0
        current_drawdown = self._get_current_drawdown()

        # === 1. CAPITAL PRESERVATION (Most Important) ===
        capital_ratio = self.balance / self.initial_balance
        if capital_ratio > 1.0:
            reward += 0.05 * (capital_ratio - 1.0)  # Bonus for profit
        else:
            reward += 0.2 * (capital_ratio - 1.0)   # Stronger penalty for loss

        # === 2. DRAWDOWN PENALTY (Progressive) ===
        if current_drawdown > 0.10:
            reward -= 0.2 * (current_drawdown - 0.10)
        if current_drawdown > 0.20:
            reward -= 0.4 * (current_drawdown - 0.20)
        if current_drawdown > 0.25:
            reward -= 0.8 * (current_drawdown - 0.25)

        # === 3. NEW: TRADE FREQUENCY PENALTY ===
        # Penalize opening too many trades
        if opened_trade:
            # Base cost for every trade (forces selectivity)
            reward -= 0.05

            # Extra penalty if trading too frequently (cooldown)
            bars_since_last = self.t - self.last_trade_step
            if bars_since_last < self.trade_cooldown:
                # Strong penalty for trading within cooldown
                cooldown_penalty = 0.1 * (1 - bars_since_last / self.trade_cooldown)
                reward -= cooldown_penalty

            # Penalty for exceeding daily trade limit
            if self.daily_trades > self.max_trades_per_day:
                reward -= 0.15 * (self.daily_trades - self.max_trades_per_day)

        # === 4. NEW: HOLD BONUS FOR WINNERS ===
        # Reward for holding winning positions - THIS IS KEY!
        for pos in self.positions:
            unrealized_pct = pos['unrealized_pnl_pct']
            hold_duration = self.t - pos['entry_step']

            if unrealized_pct > 0:
                # === WINNING POSITION - REWARD FOR HOLDING ===

                # Time-scaled bonus: longer holds get more reward
                # This encourages "let winners run"
                time_factor = min(hold_duration / 24, 2.0)  # Cap at 2x after 2 hours

                # Profit-scaled bonus: more profit = more bonus
                profit_factor = min(unrealized_pct / self.min_profit_target, 3.0)  # Cap at 3x

                # Combined hold bonus (significant!)
                hold_bonus = 0.02 * time_factor * profit_factor
                reward += hold_bonus

                # Extra bonus when trailing is active (position is secured)
                if pos['trailing_active']:
                    reward += 0.01

            elif unrealized_pct < 0:
                # === LOSING POSITION - NO BONUS, SMALL PENALTY ===
                # We want model to cut losers, not hold them

                # Progressive penalty for holding losers too long
                if hold_duration > 12:  # > 1 hour
                    reward -= 0.01 * (hold_duration / 24)  # Small but cumulative

                # Warning for positions approaching stop loss
                if unrealized_pct < -self.max_stop_loss_pct * 0.7:
                    reward -= 0.02  # Position getting dangerous

        # === 5. CLOSED TRADES REWARDS (Quality Focus) ===
        for trade in closed_trades:
            pnl_pct = trade['pnl_pct']
            efficiency = trade['efficiency']
            duration = trade['duration']
            reason = trade['reason']
            was_winner = pnl_pct > 0

            # Base PnL reward (scaled down from v1)
            reward += pnl_pct * 30  # Reduced from 50 to focus on quality

            if was_winner:
                # === WINNING TRADE ===

                # DURATION BONUS: Reward for holding winners longer
                # This is key for "let winners run"
                duration_bonus = min(duration / 48, 1.0) * 0.3  # Up to +0.3 for 4hr+ hold
                reward += duration_bonus

                # EFFICIENCY BONUS: Closing near the maximum
                if efficiency >= 0.8:
                    reward += 0.8
                elif efficiency >= 0.6:
                    reward += 0.4
                elif efficiency >= 0.4:
                    reward += 0.2

                # TRAILING STOP BONUS: Let winners run
                if reason == "trailing_stop":
                    reward += 0.5  # Used trailing properly

                # PENALTY for closing winners too early (not via trailing)
                if reason == "manual" and pnl_pct < self.min_profit_target * 2:
                    reward -= 0.4  # Increased penalty

                # BIG BONUS for excellent trades (high profit + long duration)
                if pnl_pct >= 0.005 and duration >= 24:  # 0.5%+ held 2+ hours
                    reward += 0.5  # Quality trade bonus

            else:
                # === LOSING TRADE ===

                # BONUS for cutting losses quickly
                if duration < 12:  # < 1 hour in M5
                    reward += 0.15  # Good! Cut loss fast
                elif duration < 24:  # < 2 hours
                    reward += 0.05
                else:
                    # Penalty for holding losers too long
                    reward -= 0.1 * min(duration / 48, 1.0)

                # BONUS for using stop loss (capital protection)
                if reason == "stop_loss":
                    reward += 0.1  # Respected risk management

                # BONUS for small losses (controlled risk)
                if abs(pnl_pct) < self.max_stop_loss_pct * 0.5:
                    reward += 0.1  # Cut loss before max SL

        # === 6. NEW: ACTION CONSISTENCY BONUS ===
        # Reward for not flip-flopping (staying with decision)
        if self.last_action is not None and action == self.last_action:
            self.consecutive_hold_steps += 1
            if self.consecutive_hold_steps > 6:  # 30 min of consistency
                reward += 0.005  # Small but cumulative
        else:
            self.consecutive_hold_steps = 0

        # === 7. POSITION MANAGEMENT ===
        # Reward for having profitable positions (not just count)
        total_unrealized = sum(p['unrealized_pnl_pct'] for p in self.positions)
        if total_unrealized > self.min_profit_target:
            reward += 0.05

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        closed_trades = []
        opened_trade = False  # NEW: Track if we opened a trade this step

        # NEW: Track daily trades (reset at day boundary)
        current_day = self.t // self.bars_per_day
        if current_day != self.current_day:
            self.current_day = current_day
            self.daily_trades = 0

        # Update all positions with current price
        self._update_positions()

        # Check automatic exits (SL, Trailing)
        auto_closed = self._check_auto_exits()
        closed_trades.extend(auto_closed)

        # Process action
        if action == 0:
            # Close all positions
            if self.positions:
                manual_closed = self._close_all_positions("manual")
                closed_trades.extend(manual_closed)

        elif action == 1:
            # Open new position if allowed
            if self._can_open_new_position():
                self._open_position()
                opened_trade = True  # NEW: Mark that we opened a trade
                self.last_trade_step = self.t
                self.daily_trades += 1
                self.recent_trade_steps.append(self.t)

        # NEW: Clean up old trade steps (keep last day only)
        cutoff = self.t - self.bars_per_day
        self.recent_trade_steps = [s for s in self.recent_trade_steps if s > cutoff]

        # Calculate reward (NEW: pass opened_trade flag)
        reward = self._calculate_reward(closed_trades, action, opened_trade)

        # NEW: Track last action for consistency bonus
        self.last_action = action

        # Track equity
        total_equity = self._get_total_equity()
        self.equity_curve.append(total_equity)
        self.positions_count_history.append(len(self.positions))

        # Track returns
        step_return = self.r[self.t] if self.t < len(self.r) else 0
        self.returns_history.append(step_return)

        # Advance time
        self.t += 1

        # Check termination
        terminated = self.t >= self.T - 1
        truncated = False

        # === HARD STOP: 30% Drawdown ===
        current_dd = self._get_current_drawdown()
        if current_dd >= self.max_drawdown:
            self.hard_stop_triggered = True
            terminated = True
            reward -= 20.0  # Large penalty

        info = {
            "balance": float(self.balance),
            "equity": float(total_equity),
            "num_positions": len(self.positions),
            "drawdown": float(current_dd),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": self.winning_trades / max(1, self.total_trades),
            "hard_stop": self.hard_stop_triggered,
            # NEW: Trade frequency info
            "daily_trades": self.daily_trades,
            "bars_since_last_trade": self.t - self.last_trade_step,
        }

        if closed_trades:
            info["closed_trades"] = closed_trades

        return self._get_obs(), float(reward), terminated, truncated, info

    def get_evaluation_data(self):
        """Return data for performance evaluation."""
        return {
            'equities': self.equity_curve,
            'positions': self.positions_count_history,
            'returns': self.returns_history,
            'trades': self.trades_history,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'hard_stop_triggered': self.hard_stop_triggered,
        }


class GoldTradingEnvDreamer:
    """
    Wrapper for GoldTradingEnv compatible with DreamerV3's interface.
    """

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        prices: np.ndarray = None,
        window: int = 128,
        initial_balance: float = 10000.0,
        max_drawdown: float = 0.30,
        **kwargs
    ):
        self.env = GoldTradingEnv(
            features=features,
            returns=returns,
            prices=prices,
            window=window,
            initial_balance=initial_balance,
            max_drawdown=max_drawdown,
            **kwargs
        )
        self.hard_stop_triggered = False

    def reset(self):
        """Reset and return observation (not tuple)."""
        obs, _ = self.env.reset()
        self.hard_stop_triggered = False
        return obs

    def step(self, action_onehot):
        """Step with one-hot encoded action."""
        action = int(np.argmax(action_onehot))
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('hard_stop', False):
            self.hard_stop_triggered = True
        done = terminated or truncated
        return obs, reward, done, info

    @property
    def observation_space(self):
        return self.env.observation_space.shape[0]

    @property
    def action_space(self):
        return self.env.action_space.n

    def get_evaluation_data(self):
        return self.env.get_evaluation_data()


class EarlyStoppingCallback:
    """Callback to stop training if drawdown exceeds limit."""

    def __init__(self, max_drawdown: float = 0.30, check_freq: int = 1000):
        self.max_drawdown = max_drawdown
        self.check_freq = check_freq
        self.n_calls = 0
        self.should_stop = False

    def __call__(self, locals_dict, globals_dict):
        self.n_calls += 1
        if self.n_calls % self.check_freq == 0:
            infos = locals_dict.get('infos', [])
            for info in infos:
                if info.get('hard_stop', False):
                    print(f"\n⛔ TRAINING STOPPED: Max drawdown {self.max_drawdown*100}% reached!")
                    self.should_stop = True
                    return False
        return True
