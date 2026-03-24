<p align="center">
  <h1 align="center">PhantomBot</h1>
  <p align="center">
    <strong>ML-Enhanced Cryptocurrency Trading Bot</strong>
  </p>
  <p align="center">
    Signal-scored entries &bull; ATR-based risk management &bull; LLM trade analysis &bull; Auto-retraining ML filter
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/freqtrade-2024.x-green?logo=bitcoin&logoColor=white" alt="Freqtrade">
  <img src="https://img.shields.io/badge/ML-LightGBM-orange" alt="ML">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/mode-paper%20%7C%20live-yellow" alt="Mode">
</p>

---

## Overview

PhantomBot is a quantitative cryptocurrency trading bot built on [Freqtrade](https://www.freqtrade.io/) with a custom multi-signal scoring strategy, machine learning trade filtering, and LLM-powered trade analysis. Designed for 1-minute scalping on Binance spot markets.

### Key Features

- **Signal Scoring Engine** — 6-factor scoring system (EMA trend, RSI zone, MACD momentum, volume spike, Bollinger position, macro trend) with configurable thresholds
- **Expected Value Gating** — Only enters trades with positive mathematical expectation after fees and slippage
- **ATR-Based Risk Management** — Dynamic stop-loss and take-profit levels based on entry-time ATR (prevents stop widening during adverse moves)
- **ML Signal Filter** — GradientBoosting classifier trained on real trade outcomes to filter low-quality signals (auto-retrains every 6h when sufficient new data exists)
- **Model Versioning** — Archives previous models, tracks AUC history, only deploys new model if it meets quality threshold
- **LLM Trade Supervisor** — Analyzes trade history using free LLM models (via OpenRouter) and provides optimization recommendations
- **Circuit Breaker** — Automatically halts trading if drawdown exceeds 20%
- **Dynamic Position Sizing** — Adjusts max open trades based on wallet balance
- **Professional Dashboard** — Real-time web UI with dark/light themes, live P&L charts, trade history, ML status, and bot controls

## Architecture

```
freqtrade-bot/
├── .env.example                  # Environment variable template
├── .gitignore                    # Excludes secrets, DBs, models, logs
├── user_data/
│   ├── config.example.json       # Config template (copy to config.json)
│   ├── strategies/
│   │   ├── PhantomStrategy.py    # Main strategy — signal scoring + EV gating
│   │   └── PhantomAdaptive.py    # ML-enhanced layer (extends PhantomStrategy)
│   └── logs/                     # Bot logs (gitignored)
├── ml/
│   ├── trainer.py                # ML model training pipeline
│   ├── llm_supervisor.py         # LLM-powered trade analysis
│   └── models/                   # Trained models + metadata (gitignored)
└── scripts/
    ├── start_bot.sh              # Start trading bot
    ├── stop_bot.sh               # Stop trading bot
    ├── smart_retrain.sh          # Conditional ML retraining (cron)
    ├── retrain.sh                # Force ML retraining
    ├── run_supervisor.sh         # Run LLM analysis
    └── auto_apply_hyperopt.sh    # Auto-apply hyperopt results
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Freqtrade](https://www.freqtrade.io/en/stable/installation/) installed
- Binance account (for live/paper trading)
- Linux server recommended (tested on Kali Linux)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Endercraft2017/Trading-bot.git
cd Trading-bot/freqtrade-bot

# 2. Install Freqtrade (if not already installed)
# See: https://www.freqtrade.io/en/stable/installation/

# 3. Install ML dependencies
pip install scikit-learn joblib lightgbm openai

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Configure Freqtrade
cp user_data/config.example.json user_data/config.json
# Edit config.json with your exchange keys and preferences

# 6. Start in paper trading mode (default)
bash scripts/start_bot.sh
```

### Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `OPENROUTER_KEY` | OpenRouter API key for LLM analysis (free models) | Optional |
| `EXCHANGE_KEY` | Binance API key | For live trading |
| `EXCHANGE_SECRET` | Binance API secret | For live trading |
| `FT_API_USERNAME` | Freqtrade API username | For dashboard |
| `FT_API_PASSWORD` | Freqtrade API password | For dashboard |
| `FT_JWT_SECRET` | JWT secret for API auth | For dashboard |

> **Important:** Never commit your `.env` or `config.json` files. Use the provided `.example` templates.

## Strategy Details

### PhantomStrategy

The core strategy uses a **6-point signal scoring system**:

| Score | Condition | Description |
|-------|-----------|-------------|
| +1 | EMA Fast > EMA Slow | Short-term trend is bullish |
| +1 | Price > EMA Trend | Macro trend aligned |
| +1 | RSI in buy zone | Not oversold or overbought |
| +1 | MACD histogram positive & rising | Momentum building |
| +1 | Volume > threshold | Volume spike confirmation |
| +1 | BB% < 0.75 | Room to move up within Bollinger Bands |

Trades require a minimum score of **4/6** plus positive expected value to enter.

### Risk Management

- **Hard stop-loss:** -3% (catastrophic protection)
- **Trailing stop:** Activates at +1.5% profit, trails at 0.5%
- **ATR Take-Profit:** Exits at 3x entry ATR for ~2:1 reward-to-risk
- **Circuit breaker:** Halts all trading at 20% drawdown
- **Min stake:** 10 USDT per trade (Binance minimum)

### ML Signal Filter (PhantomAdaptive)

The adaptive layer trains a GradientBoosting classifier on 13 features extracted from real trade outcomes:

```
RSI, RSI lag, MACD histogram, MACD slope, EMA spread,
ATR normalized, Volume ratio, Bollinger %B,
Hour of day, Day of week, 1/3/6-candle returns
```

- Auto-retrains every 6 hours (only if 10+ new real-feature trades)
- Archives previous models before deploying
- Only deploys if new AUC >= 0.55 and not worse than previous by >0.01
- Tracks full AUC history for monitoring model quality

## Backtesting

```bash
# Create backtest config (uses StaticPairList)
cp user_data/config.example.json user_data/config_backtest.json
# Edit to use StaticPairList (see config.example.json comments)

# Run backtest
freqtrade backtesting \
  --config user_data/config_backtest.json \
  --strategy PhantomStrategy \
  --timerange YYYYMMDD-YYYYMMDD

# Run hyperopt optimization
freqtrade hyperopt \
  --config user_data/config_backtest.json \
  --strategy PhantomStrategy \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces buy \
  --epochs 200 \
  -j 1
```

## Dashboard

The trading dashboard provides real-time monitoring with:

- Live P&L tracking and cumulative profit charts
- Open/closed trade tables with signal details
- Market watchlist with live Binance prices
- ML model status, AUC history, and retrain controls
- LLM analysis summaries and recommendations
- Bot start/stop/restart controls
- Dark and light theme support

## Security

- All secrets stored in environment variables (`.env` file, gitignored)
- Config files with credentials excluded from version control
- API server binds to `127.0.0.1` by default (not exposed to network)
- Input validation on all trade parameters
- Circuit breaker prevents runaway losses
- Model versioning prevents deploying degraded ML models
- Shell scripts use `set -euo pipefail` for safe execution

## Disclaimer

> **This software is provided for educational and research purposes only.**
>
> Trading cryptocurrency involves substantial risk of loss and is not suitable for every investor. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this software.
>
> Always start with paper trading mode (`dry_run: true`) and thoroughly test any strategy before risking real capital. Never trade with money you cannot afford to lose.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Freqtrade](https://www.freqtrade.io/) — Open-source crypto trading bot framework
- [scikit-learn](https://scikit-learn.org/) — ML model training
- [OpenRouter](https://openrouter.ai/) — Free LLM access for trade analysis
- [Chart.js](https://www.chartjs.org/) & [Lightweight Charts](https://tradingview.github.io/lightweight-charts/) — Dashboard charting
