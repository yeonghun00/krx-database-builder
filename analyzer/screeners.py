"""
Stock Screening Module

Provides fast and efficient stock screening capabilities for KRX market data.
Optimized with SQL-level aggregation to minimize memory usage.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging


class StockScreener:
    """Fast stock screening for KRX market data."""

    DEFAULT_MIN_TRADING_DAYS = 100

    def __init__(self, db_path: str = "krx_stock_data.db"):
        """
        Initialize stock screener.

        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _build_market_filter(self, markets: List[str]) -> tuple:
        """Build SQL market filter with placeholders."""
        placeholders = ','.join(['?' for _ in markets])
        return placeholders, markets

    def top_price_increase(
        self,
        start_date: str,
        end_date: str,
        markets: List[str] = None,
        percentile: int = 1,
        min_trading_days: int = None
    ) -> pd.DataFrame:
        """
        Screen for top X% by price increase over date range.

        Uses optimized single-pass query with conditional aggregation.

        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            markets (List[str]): List of markets to include (default: kospi, kosdaq)
            percentile (int): Top percentile to return (1 = top 1%)
            min_trading_days (int): Minimum trading days required (default: 100)

        Returns:
            pd.DataFrame: Top performing stocks
        """
        markets = markets or ['kospi', 'kosdaq']
        min_trading_days = min_trading_days or self.DEFAULT_MIN_TRADING_DAYS

        self.logger.info(
            f"Screening for top {percentile}% by price increase "
            f"from {start_date} to {end_date}"
        )

        market_placeholders, market_params = self._build_market_filter(markets)

        # Optimized single-pass query using MIN/MAX with CASE for first/last prices
        # This avoids expensive JOINs by computing everything in one aggregation
        query = f"""
        SELECT
            dp.stock_code,
            s.current_name as name,
            MAX(dp.market_type) as market_type,
            MIN(dp.date) as first_date,
            MAX(dp.date) as last_date,
            MAX(CASE WHEN dp.date = sub.first_date THEN dp.closing_price END) as start_price,
            MAX(CASE WHEN dp.date = sub.last_date THEN dp.closing_price END) as end_price,
            COUNT(*) as trading_days
        FROM daily_prices dp
        INNER JOIN (
            SELECT stock_code, MIN(date) as first_date, MAX(date) as last_date
            FROM daily_prices
            WHERE date >= ? AND date <= ?
              AND market_type IN ({market_placeholders})
            GROUP BY stock_code
        ) sub ON dp.stock_code = sub.stock_code
        INNER JOIN stocks s ON dp.stock_code = s.stock_code
        WHERE dp.date >= ? AND dp.date <= ?
          AND dp.market_type IN ({market_placeholders})
        GROUP BY dp.stock_code, s.current_name
        HAVING COUNT(*) >= ?
           AND start_price > 0 AND end_price > 0
        ORDER BY (end_price - start_price) * 100.0 / start_price DESC
        """

        # Parameters: subquery (date, date, markets), main query (date, date, markets), having
        params = ([start_date, end_date] + market_params +
                  [start_date, end_date] + market_params + [min_trading_days])

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            self.logger.warning("No data found for screening criteria")
            return pd.DataFrame()

        # Calculate price change percentage
        df['price_change_pct'] = ((df['end_price'] - df['start_price']) * 100.0 / df['start_price']).round(2)

        self.logger.info(f"Found {len(df)} valid stocks")

        # Get top percentile
        top_count = max(1, int(len(df) * percentile / 100))
        result_df = df.head(top_count).copy()

        # Format results
        result_df['start_price'] = result_df['start_price'].astype(int)
        result_df['end_price'] = result_df['end_price'].astype(int)
        result_df = result_df.set_index('stock_code')
        result_df = result_df.drop(columns=['first_date', 'last_date', 'trading_days'], errors='ignore')

        self.logger.info(f"Returning top {len(result_df)} stocks ({percentile}%)")
        return result_df

    def top_value(
        self,
        start_date: str,
        end_date: str,
        markets: List[str] = None,
        percentile: int = 1,
        min_trading_days: int = None
    ) -> pd.DataFrame:
        """
        Screen for top X% by total trading value over date range.

        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            markets (List[str]): List of markets to include (default: kospi, kosdaq)
            percentile (int): Top percentile to return (1 = top 1%)
            min_trading_days (int): Minimum trading days required (default: 100)

        Returns:
            pd.DataFrame: Top value stocks
        """
        markets = markets or ['kospi', 'kosdaq']
        min_trading_days = min_trading_days or self.DEFAULT_MIN_TRADING_DAYS

        self.logger.info(
            f"Screening for top {percentile}% by trading value "
            f"from {start_date} to {end_date}"
        )

        market_placeholders, market_params = self._build_market_filter(markets)

        query = f"""
        SELECT
            dp.stock_code,
            s.current_name as name,
            dp.market_type,
            SUM(dp.volume) AS total_volume,
            SUM(dp.value) AS total_value,
            ROUND(SUM(dp.value) / 1000000000.0, 2) AS total_value_billion,
            COUNT(*) AS trading_days
        FROM daily_prices dp
        INNER JOIN stocks s ON dp.stock_code = s.stock_code
        WHERE dp.date >= ? AND dp.date <= ?
          AND dp.market_type IN ({market_placeholders})
        GROUP BY dp.stock_code, s.current_name, dp.market_type
        HAVING COUNT(*) >= ?
        ORDER BY total_value DESC
        """

        params = [start_date, end_date] + market_params + [min_trading_days]

        conn = self._get_connection()
        df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            self.logger.warning("No data found for screening criteria")
            return pd.DataFrame()

        # Get top percentile
        top_count = max(1, int(len(df) * percentile / 100))
        result_df = df.head(top_count).copy()

        # Format results
        result_df['total_volume'] = result_df['total_volume'].astype(int)
        result_df = result_df.set_index('stock_code')

        self.logger.info(f"Returning top {len(result_df)} stocks ({percentile}%)")
        return result_df

    def combined_screen(
        self,
        start_date: str,
        end_date: str,
        markets: List[str] = None,
        price_percentile: int = 5,
        value_percentile: int = 5,
        min_trading_days: int = None
    ) -> pd.DataFrame:
        """
        Screen for stocks in top X% by BOTH price increase AND trading value.

        Uses optimized two-step query for better index utilization.

        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            markets (List[str]): List of markets to include (default: kospi, kosdaq)
            price_percentile (int): Top percentile for price increase
            value_percentile (int): Top percentile for trading value
            min_trading_days (int): Minimum trading days required (default: 100)

        Returns:
            pd.DataFrame: Stocks meeting both criteria
        """
        markets = markets or ['kospi', 'kosdaq']
        min_trading_days = min_trading_days or self.DEFAULT_MIN_TRADING_DAYS

        self.logger.info(
            f"Combined screening: top {price_percentile}% price AND "
            f"top {value_percentile}% value from {start_date} to {end_date}"
        )

        market_placeholders, market_params = self._build_market_filter(markets)
        conn = self._get_connection()

        # Step 1: Get aggregated stats (volume, value, date range) - uses date index efficiently
        agg_query = f"""
        SELECT
            stock_code,
            MIN(date) as first_date,
            MAX(date) as last_date,
            SUM(volume) as total_volume,
            SUM(value) as total_value,
            COUNT(*) as trading_days
        FROM daily_prices
        WHERE date BETWEEN ? AND ?
          AND market_type IN ({market_placeholders})
        GROUP BY stock_code
        HAVING COUNT(*) >= ?
        """
        agg_params = [start_date, end_date] + market_params + [min_trading_days]
        agg_df = pd.read_sql_query(agg_query, conn, params=agg_params)

        if len(agg_df) == 0:
            self.logger.warning("No data found for combined screening")
            return pd.DataFrame()

        # Step 2: Get first and last prices using primary key lookups
        # Build list of (stock_code, date) pairs for efficient batch lookup
        price_lookups = []
        for _, row in agg_df.iterrows():
            price_lookups.append((row['stock_code'], row['first_date']))
            price_lookups.append((row['stock_code'], row['last_date']))

        # Batch query for prices - uses primary key (stock_code, date)
        price_placeholders = ','.join(['(?,?)' for _ in price_lookups])
        price_params = [item for pair in price_lookups for item in pair]

        price_query = f"""
        SELECT stock_code, date, closing_price, market_type
        FROM daily_prices
        WHERE (stock_code, date) IN ({price_placeholders})
        """
        price_df = pd.read_sql_query(price_query, conn, params=price_params)

        # Create lookup dict for prices
        price_lookup = {(r['stock_code'], r['date']): (r['closing_price'], r['market_type'])
                        for _, r in price_df.iterrows()}

        # Step 3: Combine data
        results = []
        for _, row in agg_df.iterrows():
            stock_code = row['stock_code']
            first_price_data = price_lookup.get((stock_code, row['first_date']))
            last_price_data = price_lookup.get((stock_code, row['last_date']))

            if first_price_data and last_price_data:
                start_price, _ = first_price_data
                end_price, market_type = last_price_data

                if start_price and end_price and start_price > 0:
                    results.append({
                        'stock_code': stock_code,
                        'market_type': market_type,
                        'start_price': int(start_price),
                        'end_price': int(end_price),
                        'total_volume': int(row['total_volume']),
                        'total_value': row['total_value'],
                    })

        if not results:
            self.logger.warning("No valid price data found")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Calculate derived columns
        df['price_change_pct'] = ((df['end_price'] - df['start_price']) * 100.0 / df['start_price']).round(2)
        df['total_value_billion'] = (df['total_value'] / 1_000_000_000).round(2)

        # Get stock names
        stock_codes = df['stock_code'].tolist()
        name_placeholders = ','.join(['?' for _ in stock_codes])
        name_query = f"SELECT stock_code, current_name FROM stocks WHERE stock_code IN ({name_placeholders})"
        name_df = pd.read_sql_query(name_query, conn, params=stock_codes)
        name_lookup = dict(zip(name_df['stock_code'], name_df['current_name']))
        df['name'] = df['stock_code'].map(name_lookup)

        total_stocks = len(df)
        self.logger.info(f"Found {total_stocks} valid stocks for combined analysis")

        # Calculate thresholds using quantiles
        price_threshold = df['price_change_pct'].quantile(1 - price_percentile / 100)
        value_threshold = df['total_value'].quantile(1 - value_percentile / 100)

        # Filter stocks meeting both criteria
        result_df = df[
            (df['price_change_pct'] >= price_threshold) &
            (df['total_value'] >= value_threshold)
        ].copy()

        if len(result_df) == 0:
            self.logger.warning("No stocks found meeting both criteria")
            return pd.DataFrame()

        # Format results
        result_df = result_df.sort_values('price_change_pct', ascending=False)
        result_df = result_df.set_index('stock_code')
        result_df = result_df[['name', 'market_type', 'start_price', 'end_price',
                               'price_change_pct', 'total_volume', 'total_value_billion']]

        self.logger.info(f"Found {len(result_df)} stocks meeting both criteria")
        return result_df

    def export_results(
        self,
        df: pd.DataFrame,
        filename_prefix: str = "screening_results"
    ) -> str:
        """
        Export screening results to Excel.

        Args:
            df (pd.DataFrame): Results DataFrame
            filename_prefix (str): Prefix for output files

        Returns:
            str: Path to exported file
        """
        if len(df) == 0:
            self.logger.warning("No results to export")
            return ""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.xlsx"

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=True)

            # Build summary based on available columns
            summary_data = {'Metric': ['Total Stocks'], 'Value': [len(df)]}

            if 'price_change_pct' in df.columns:
                summary_data['Metric'].extend([
                    'Average Price Change',
                    'Max Price Change',
                    'Min Price Change'
                ])
                summary_data['Value'].extend([
                    f"{df['price_change_pct'].mean():.2f}%",
                    f"{df['price_change_pct'].max():.2f}%",
                    f"{df['price_change_pct'].min():.2f}%"
                ])

            if 'total_value_billion' in df.columns:
                summary_data['Metric'].extend([
                    'Total Trading Value',
                    'Average Trading Value',
                    'Max Trading Value'
                ])
                summary_data['Value'].extend([
                    f"{df['total_value_billion'].sum():.2f}B",
                    f"{df['total_value_billion'].mean():.2f}B",
                    f"{df['total_value_billion'].max():.2f}B"
                ])

            summary = pd.DataFrame(summary_data)
            summary.to_excel(writer, sheet_name='Summary', index=False)

        self.logger.info(f"Results exported to {filename}")
        return filename

    def screen_and_export(
        self,
        start_date: str,
        end_date: str,
        markets: List[str] = None,
        price_percentile: int = 1,
        value_percentile: int = 1,
        export: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Complete screening workflow with optional export.

        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            markets (List[str]): List of markets to include
            price_percentile (int): Top percentile for price increase
            value_percentile (int): Top percentile for trading value
            export (bool): Whether to export results

        Returns:
            Dict[str, pd.DataFrame]: All screening results
        """
        markets = markets or ['kospi', 'kosdaq']

        results = {
            'price_increase': self.top_price_increase(
                start_date, end_date, markets, price_percentile
            ),
            'value': self.top_value(
                start_date, end_date, markets, value_percentile
            ),
            'combined': self.combined_screen(
                start_date, end_date, markets, price_percentile, value_percentile
            )
        }

        if export and len(results['combined']) > 0:
            results['export_file'] = self.export_results(
                results['combined'], "combined_screening"
            )

        return results
