# AlgoStock

<p align="center">
  <b>Korean Stock Market Data Collection, Screening & Backtesting System</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/market-KOSPI%20%7C%20KOSDAQ%20%7C%20KODEX-orange.svg" alt="Markets">
</p>

---

A comprehensive system for automated collection, screening, and backtesting of Korean stock market data from KRX (Korea Exchange).

## Features

| Feature | Description |
|---------|-------------|
| **Data Collection** | Automated ETL pipeline with KRX API integration |
| **Stock Screening** | Find top performers by price increase, trading value, or combined metrics |
| **Backtesting** | Test screening strategies with customizable parameters |
| **Historical Data** | 15+ years of data (2011-present) for KOSPI, KOSDAQ, KODEX |

## Quick Start

```bash
# Clone
git clone https://github.com/yourusername/algostock.git
cd algostock

# Install
pip install -r requirements.txt

# Configure
cp config.example.json config.json
# Edit config.json with your KRX API key

# Check status
python3 algostock_cli.py etl status
```

## Getting KRX API Key

1. Visit [KRX Open API](https://data.krx.co.kr/)
2. Create an account and apply for API access
3. Copy your API key to `config.json`

## Usage

### ETL (Extract-Transform-Load)

```bash
# Database status
python3 algostock_cli.py etl status

# Daily update
python3 algostock_cli.py etl update

# Auto catch-up (fills any gaps)
python3 algostock_cli.py etl update --catchup

# Historical backfill
python3 algostock_cli.py etl backfill -s 20200101 -e 20201231

# Verify data completeness
python3 algostock_cli.py etl verify

# Auto-fix missing data
python3 algostock_cli.py etl verify --fix
```

### Stock Screening

```bash
# Top 5% by price increase
python3 algostock_cli.py analyze price -s 20240101 -e 20241231 -p 5

# Top 5% by trading value
python3 algostock_cli.py analyze value -s 20240101 -e 20241231 -p 5

# Combined: top 5% in BOTH price AND value
python3 algostock_cli.py analyze combined -s 20240101 -e 20241231 -pp 5 -vp 5

# Filter by market (kospi, kosdaq, kodex)
python3 algostock_cli.py analyze combined -s 20240101 -e 20241231 -m kospi,kosdaq
```

### Backtesting

```bash
# Basic: yearly screening, 1-year hold
python3 algostock_cli.py backtest -s 20200101 -e 20241231

# Custom strategy
python3 algostock_cli.py backtest -s 20200101 -e 20241231 \
    --n-stocks 10 \
    --holding-months 6 \
    --rebalance-months 6 \
    --weighting mcap \
    --sort-by combined
```

#### Backtest Options

| Option | Values | Description |
|--------|--------|-------------|
| `--n-stocks` | 1-100 | Number of stocks per period |
| `--holding-months` | 1-60 | How long to hold positions |
| `--rebalance-months` | 1-60 | How often to rebalance |
| `--weighting` | `equal`, `value`, `inverse_value`, `mcap`, `inverse_mcap` | Portfolio weighting method |
| `--sort-by` | `price`, `value`, `combined` | Stock selection criteria |

## Project Structure

```
algostock/
├── algostock_cli.py       # Unified CLI interface
├── krx_api.py             # KRX API client
├── clean_etl.py           # ETL pipeline
├── config.py              # Configuration loader
├── config.example.json    # Example config template
├── analyzer/
│   ├── screeners.py       # Stock screening engine
│   └── backtester.py      # Backtesting engine
└── requirements.txt
```

## Database

SQLite database with ~8M+ records:

| Table | Description |
|-------|-------------|
| `stocks` | Stock metadata (code, name, market) |
| `daily_prices` | Daily OHLCV data with optimized indexes |

## Configuration

Key settings in `config.json`:

```json
{
  "api": {
    "auth_key": "YOUR_KRX_API_KEY",
    "request_delay": 1.0,
    "max_concurrent_requests": 3
  },
  "backfill": {
    "start_year": 2011
  }
}
```

## Requirements

- Python 3.8+
- pandas
- requests
- openpyxl (Excel export)

## Sample Output

```
=== Backtest Results ===
Period  Hold Start   Hold End   Return    Cumulative
──────────────────────────────────────────────────────
1       2020-01-01   2020-12-31  +45.2%    +45.2%
2       2021-01-01   2021-12-31  +12.8%    +63.8%
3       2022-01-01   2022-12-31  -18.5%    +33.5%

Best:  삼성전자 (+89.2%)
Worst: 카카오 (-45.1%)
```

## License

MIT

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. This is not financial advice.
