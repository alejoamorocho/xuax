# XAUX - Gold Trading AI

> *"Cut your losses short, let your winners run."* — Every successful trader who didn't blow up their account.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced Deep Reinforcement Learning system for XAUUSD (Gold) trading. Built with love, caffeine, and an unhealthy obsession with Sharpe ratios.

---

## What is this?

XAUX is a professional-grade trading AI that uses state-of-the-art reinforcement learning algorithms to trade gold. Unlike your cousin's "guaranteed profit" forex bot, this one actually understands risk management.

**Two powerful algorithms, one goal: consistent profits.**

| Algorithm | Type | Superpower |
|-----------|------|------------|
| **PPO** | Model-Free | Fast, stable, battle-tested |
| **DreamerV3** | Model-Based | Learns from imagination, literally |

---

## The Philosophy

```
           LOSERS                    WINNERS
    ┌─────────────────┐      ┌─────────────────────────────┐
    │  Cut fast!      │      │  Let them run...            │
    │  Max 0.9%       │      │  No limit! Trailing only    │
    └─────────────────┘      └─────────────────────────────┘
           ↓                              ↓
       Small loss                    Big profit
       Move on                       Compound gains
```

**Translation:** Lose small, win big, repeat until rich (or at least until you can afford that fancy coffee).

---

## Features at a Glance

### 152 Features Per Decision

The AI doesn't just look at price charts. It analyzes:

| Category | Features | What it sees |
|----------|----------|--------------|
| **Multi-Timeframe** | 96 | M5, M15, H1, H4, D1, W1 simultaneously |
| **Cross-Timeframe** | 12 | Divergences between timeframes |
| **Macro** | 24 | DXY, VIX, SPX, US10Y, Oil, BTC, EUR, Silver |
| **Calendar** | 8 | NFP, FOMC, CPI, GDP impact |
| **Microstructure** | 12 | Spread, volume, volatility regimes |

Each decision uses **128 bars x 152 features = 19,456 data points**.

The AI literally sees the matrix.

---

## Professional Risk Management

This is where the magic happens. No amateur hour here.

```python
Risk per trade:     2% of capital      # Enough to matter, not enough to hurt
Max Stop Loss:      0.9%               # Can exit earlier if trade looks ugly
Take Profit:        None               # No ceiling on winners
Trailing Start:     0.3%               # Lock in profits after 0.3% gain
Max Drawdown:       30%                # Hard stop - model is useless beyond this
Multiple Positions: Yes (max 3)        # Only when existing trades are green
```

### The Risk Math

| Scenario | Outcome |
|----------|---------|
| 10 trades, 6 losses at 0.9% | -5.4% |
| 10 trades, 4 wins at 3%+ avg | +12%+ |
| **Net result** | +6.6%+ |

Win rate doesn't need to be 90%. It needs to be *asymmetric*.

---

## Reward System (How the AI Learns)

The AI gets rewarded for being disciplined and punished for being emotional:

| Action | Reward | Why |
|--------|--------|-----|
| Cut loss quickly (< 1 hour) | **+0.2** | Smart traders don't hope |
| Use stop loss properly | **+0.1** | Risk management = survival |
| Close via trailing stop | **+0.5** | Let winners run! |
| Close near maximum profit (80%+) | **+1.0** | Peak efficiency |
| Close winner too early | **-0.3** | Paper hands penalty |
| Hit 30% drawdown | **-20.0** | Game over |

**Translation:** The AI gets a cookie for being patient with winners and gets slapped for being emotional.

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/alejoamorocho/xuax.git
cd xuax
pip install -r requirements.txt
```

### 2. Get the Data

```bash
# Fetch macro data (VIX, DXY, Oil, etc.)
python scripts/fetch_all_data.py

# Generate economic calendar
python scripts/generate_economic_calendar.py

# Export XAUUSD M5 data from MetaTrader 5 to data/XAUUSD_M5.csv
```

### 3. Train on Google Colab (Recommended)

Upload to Google Drive and open the notebooks:

| Notebook | Algorithm | Time (A100) | Recommended Steps |
|----------|-----------|-------------|-------------------|
| `colab_train_ppo_ultimate.ipynb` | PPO | ~9 hours | 3,000,000 |
| `colab_train_ultimate_150.ipynb` | DreamerV3 | ~5 hours | 1,000,000 |

### Training Configuration

```python
TIMESTEPS = 3_000_000    # Total training steps (be patient!)
WINDOW = 128             # 10.6 hours of market context
DEVICE = 'cuda'          # GPU acceleration (A100 > V100 > T4)
```

---

## Project Structure

```
XAUX/
├── data/                    # Market data (OHLCV + Macro)
│   ├── XAUUSD_M5.csv       # 5-minute gold data
│   ├── dxy_daily.csv       # Dollar Index
│   ├── vix_daily.csv       # Fear Index
│   └── economic_events.json # NFP, FOMC, CPI dates
│
├── features/                # 152 feature engineering modules
│   ├── ultimate_150_features.py   # Main feature pipeline
│   ├── macro_features.py          # DXY, VIX, yields correlation
│   ├── calendar_features.py       # Economic event impact
│   └── microstructure_features.py # Market structure analysis
│
├── env/                     # Trading environments
│   └── gold_trading_env.py        # Professional trading logic
│
├── models/                  # Neural network architectures
│   └── dreamer_agent.py           # DreamerV3 world model
│
├── train/                   # Training scripts
│   ├── train_ppo_ultimate.py      # PPO training
│   └── train_ultimate_150.py      # DreamerV3 training
│
├── eval/                    # Evaluation & backtesting
│   ├── baselines.py               # Compare vs buy-and-hold
│   └── crisis_validation.py       # Test on market crashes
│
└── colab_*.ipynb           # Google Colab notebooks
```

---

## PPO Hyperparameters (Battle-Tested)

Based on extensive research for intraday-swing gold trading:

```python
gamma = 0.6              # Short-term focus (not HODLing here)
learning_rate = 3e-4 → 1e-5  # Starts aggressive, becomes careful
target_kl = 0.01         # "Whoa, slow down" threshold
ent_coef = 0.02          # Exploration bonus (try new things!)
n_steps = 4096           # Experience buffer
batch_size = 128         # Mini-batch size
```

Why gamma = 0.6? Because we're trading intraday-swing, not investing for retirement.

---

## Training Splits

| Set | Period | Purpose | Rule |
|-----|--------|---------|------|
| **Train** | 2015 - 2021 | Learn patterns | AI sees this |
| **Validation** | 2022 | Tune hyperparameters | AI validates on this |
| **Test** | 2023 - 2025 | Final evaluation | **AI NEVER sees this until the end** |

No peeking. No data leakage. No BS.

---

## Expected Results

After proper training (3M+ steps):

| Metric | Target | Reality Check |
|--------|--------|---------------|
| Sharpe Ratio | > 1.5 | > 1.0 is already good |
| Sortino Ratio | > 2.0 | Focuses on downside risk |
| Max Drawdown | < 25% | Sleep at night levels |
| Win Rate | > 55% | Doesn't need to be 80% |
| Profit Factor | > 1.5 | Gross profit / Gross loss |

*Results may vary. Markets are not your friend.*

---

## Why Two Algorithms?

### PPO (Proximal Policy Optimization)
```
Pros:  Fast, stable, well-documented, less magic
Cons:  Needs more data, no imagination
Best:  Quick iterations, proven results
```

### DreamerV3 (World Model)
```
Pros:  Sample efficient, "dreams" about markets, better planning
Cons:  More complex, slower per step, harder to debug
Best:  Maximum performance when you're patient
```

**Recommendation:** Train both. Compare on test set. Deploy the winner. May the best algorithm win.

---

## Live Trading

Ready to go live? Take a deep breath first.

```bash
# MetaTrader 5 (local)
python live_trade_mt5.py

# MetaAPI (cloud, no VPS needed)
python live_trade_metaapi.py
```

### The Golden Rules

1. **Paper trade first** — Minimum 2 weeks, no exceptions
2. **Start small** — Smallest position size possible
3. **Monitor daily** — At least for the first month
4. **Have a kill switch** — Maximum daily loss = trading stops
5. **Don't override the AI** — You built it for a reason

---

## Disclaimer

**IMPORTANT — Read this before you lose your life savings**

This software is for **educational purposes only**.

- Trading involves **substantial risk of loss**
- Past performance does **NOT** guarantee future results
- You could lose **more than your initial investment**
- Automated trading can fail due to bugs, connectivity, or "the market being the market"

**The authors are NOT responsible for any financial losses.**

*Translation: Don't blame us if you YOLO your rent money and lose it all.*

---

## Contributing

Found a bug? Have an idea? PRs welcome!

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

MIT License — Do whatever you want, just don't sue us.

---

## Acknowledgments

- **Stable-Baselines3** team for PPO implementation
- **DreamerV3** paper authors for the brilliant world model
- **Coffee** for making late-night debugging sessions possible
- **Whoever invented trailing stops** — You're a legend

---

## Final Words

> *"The market can stay irrational longer than you can stay solvent."* — John Maynard Keynes

But with proper risk management, you might just survive long enough to profit.

**Happy trading!**

*Remember: The best trade is sometimes no trade at all.*

---

*Built with frustration and eventual triumph by traders who've been there.*
