# AlgoStock Analyzer Module

Fast and efficient stock screening and analysis capabilities for KRX market data.

## 🚀 Features

- **Fast Screening**: Optimized with pandas and numpy for performance
- **Multiple Metrics**: Price increase, trading volume, combined screening
- **Flexible Filtering**: Date ranges, market selection, percentile thresholds
- **Export Capabilities**: CSV and Excel export with summary statistics
- **CLI Interface**: Command-line tools for easy analysis

## 📊 Screening Capabilities

### 1. Price Increase Screening
Find stocks with the highest percentage gains over a date range.

```python
from analyzer.screeners import StockScreener

screener = StockScreener("krx_stock_data.db")
results = screener.top_price_increase("20250101", "20251231", ["kospi", "kosdaq"], percentile=1)
```

### 2. Volume Screening
Find stocks with the highest trading volume over a date range.

```python
results = screener.top_volume("20250101", "20251231", ["kospi", "kosdaq"], percentile=1)
```

### 3. Combined Screening
Find stocks that are in the top X% for both price increase AND trading volume.

```python
results = screener.combined_screen("20250101", "20251231", ["kospi", "kosdaq"], 
                                  price_percentile=5, volume_percentile=5)
```

## 🖥️ Command Line Interface

### Quick Screening for Current Year
```bash
python3 cli/analyzer_cli.py quick --percentile 5
```

### Price Increase Screening
```bash
python3 cli/analyzer_cli.py price-increase --start-date 20250101 --end-date 20251231 --percentile 1
```

### Volume Screening
```bash
python3 cli/analyzer_cli.py volume --start-date 20250101 --end-date 20251231 --percentile 1
```

### Combined Screening
```bash
python3 cli/analyzer_cli.py combined --start-date 20250101 --end-date 20251231 --price-percentile 1 --volume-percentile 1
```

## 📈 Performance

The analyzer is optimized for speed:
- **10-30 seconds** for complex screenings (vs minutes with SQL)
- **Memory efficient** pandas operations
- **Parallel processing** ready for future enhancements
- **Database optimized** queries

## 🎯 Example Results

### Top Performers in 2025
```
세종텔레콤 (kosdaq): +2377% (400 → 9910)
이스트아시아홀딩스 (kosdaq): +1753% (60 → 1112)
원익홀딩스 (kosdaq): +1633% (2810 → 48700)
미래산업 (kospi): +1618% (802 → 13780)
```

### High Volume Stocks in 2025
```
KBI동양철관 (kospi): 6.1B shares, 9.4T won
삼성전자 (kospi): 4.7B shares, 341.9T won
이스트아시아홀딩스 (kosdaq): 3.9B shares, 341B won
```

## 📁 Module Structure

```
analyzer/
├── __init__.py          # Module exports
├── screeners.py         # Core screening functionality
├── technical_analysis.py # Technical indicators (future)
└── reports.py          # Report generation (future)
```

## 🔧 Dependencies

- pandas
- numpy
- sqlite3 (built-in)
- openpyxl (for Excel export)

## 🚀 Usage Examples

### Basic Screening
```python
from analyzer.screeners import StockScreener

# Initialize screener
screener = StockScreener("krx_stock_data.db")

# Screen for top 1% by price increase in 2025
results = screener.top_price_increase("20250101", "20251231", ["kospi", "kosdaq"], 1)

# Export results
filename = screener.export_results(results, "price_increase_2025")
print(f"Results saved to {filename}")
```

### Combined Analysis
```python
# Find stocks in top 5% for both price and volume
combined_results = screener.combined_screen(
    "20250101", "20251231", 
    ["kospi", "kosdaq"], 
    price_percentile=5, 
    volume_percentile=5
)

print(f"Found {len(combined_results)} elite performers")
```

## 📊 Output Format

Results are returned as pandas DataFrames with columns:
- `stock_code`: Stock identifier
- `name`: Company name
- `market_type`: kospi/kosdaq/kodex
- `start_price`: Starting price
- `end_price`: Ending price
- `price_change_pct`: Percentage change
- `total_volume`: Total trading volume
- `total_value_billion`: Total trading value (in billions)

## 🎯 Future Enhancements

- Technical indicators (moving averages, RSI, MACD)
- Fundamental analysis integration
- Custom screening criteria
- Real-time screening
- Web dashboard interface
- Machine learning predictions

## 📝 Notes

- Markets: kospi, kosdaq, kodex
- Date format: YYYYMMDD
- Percentile: 1-100 (1 = top 1%)
- Minimum trading days: 100 (configurable)
- Export formats: Excel with multiple sheets