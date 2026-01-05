# PPO Gold Trading Research - January 2025

## Key Findings for XAUUSD M5 Intraday-Swing Trading

Based on extensive research of 2024-2025 papers and implementations.

---

## 1. PPO Hyperparameters (CRITICAL CHANGES NEEDED)

### Current vs Recommended

| Parameter | Current | Recommended | Why |
|-----------|---------|-------------|-----|
| **gamma** | 0.6 | **0.98-0.99** | Higher = longer-term thinking, fewer trades |
| **learning_rate** | 5e-5 | 3e-4 -> 1e-5 | Linear decay schedule |
| **n_steps** | 4096 | 4096 | OK |
| **batch_size** | 128 | **256-512** | Larger = more stable |
| **ent_coef** | 0.02 | **0.01** | Less random exploration |
| **gae_lambda** | 0.95 | 0.95-0.97 | OK |
| **clip_range** | 0.2 | 0.2 | OK |

### Gamma is CRITICAL

- **Current gamma=0.6**: Agent only cares about next ~2.5 steps (1/(1-0.6)=2.5)
- **Recommended gamma=0.98**: Agent considers next ~50 steps (1/(1-0.98)=50)
- This single change can reduce over-trading dramatically

---

## 2. Reward Function Design

### Multi-Objective Reward (Not Single PnL!)

```
Total Reward =
    1.0 * return_reward +
    2.0 * differential_sharpe_ratio +   # Risk-adjusted
    5.0 * transaction_cost_penalty +     # Anti-overtrading
    3.0 * drawdown_penalty +              # Risk control
    1.0 * holding_incentive +             # Longer holds
    0.5 * inaction_penalty               # Don't be flat forever
```

### Differential Sharpe Ratio (DSR)

```python
class DSR:
    def __init__(self, eta=0.01):
        self.A = 0  # Running mean
        self.B = 0  # Running variance

    def update(self, r):
        self.A += eta * (r - self.A)
        self.B += eta * (r**2 - self.B)

        if self.B - self.A**2 > 0:
            dsr = (self.B * (r - self.A) - 0.5 * self.A * (r**2 - self.B)) / (self.B - self.A**2)**1.5
        else:
            dsr = 0
        return dsr
```

### Transaction Cost Penalty (Key for over-trading)

```python
# Per trade penalty
SPREAD_COST = 0.0003  # 3 pips
COMMISSION = 0.00007  # $7 per $100k

if position_changed:
    reward -= 5.0 * (SPREAD_COST + COMMISSION)
```

### Drawdown Penalty (Key for 30% max DD issue)

```python
# Activate at 15% drawdown
if drawdown > 0.15:
    penalty = -3.0 * ((drawdown - 0.15) / 0.15) ** 2
```

### Holding Incentive (Key for short duration)

```python
# Only for WINNING positions
if profitable and duration >= 6:  # 30 min for M5
    bonus = 0.05 * min(duration / 6, 3)  # Cap at 3x
```

---

## 3. Gold (XAUUSD) Specific

### Best Trading Windows

| Session | GMT | Recommendation |
|---------|-----|----------------|
| **London-NY Overlap** | 13:00-17:00 | **PRIMARY (highest liquidity)** |
| London | 08:00-16:00 | Good |
| NY afternoon | 17:00-21:00 | OK |
| Asian | 23:00-07:00 | **AVOID (low liquidity)** |

### Key Correlations

| Feature | Correlation | Usage |
|---------|-------------|-------|
| **DXY** | -0.45 (inverse) | Primary indicator |
| US 10Y Yields | Negative | Rate expectations |
| VIX | Positive when >20 | Safe-haven demand |
| Real Yields | Strong negative | Key driver |

---

## 4. Recommended Configuration

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,  # With linear decay to 1e-5
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.98,          # CRITICAL: Changed from 0.6!
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,       # Reduced from 0.02
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    verbose=1
)
```

---

## 5. Expected Results After Changes

| Metric | Before | Expected After |
|--------|--------|----------------|
| Trades | 3000+ | 500-1000 |
| Win Rate | 50% | 55-60% |
| Avg Duration | <30 min | 1-3 hours |
| Max Drawdown | 30% | <15% |
| Sharpe | <1 | >2 |

---

## Sources

- Stable Baselines3 PPO Documentation
- FinRL FAQ and Implementations
- "Risk-Adjusted DRL Multi-reward Approach" (2024)
- "Embedded Drawdown Constraint Reward" (2024)
- "Differential Sharpe Ratio" Implementation
- XAUUSD Trading Hours Research
- Gold-DXY Correlation Studies
