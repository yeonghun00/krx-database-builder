"""
Backtester Module

Backtests stock screening strategies with configurable parameters.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple
import logging
import os


class Backtester:
    """Backtest screening strategies over historical data."""

    WEIGHTING_METHODS = ['equal', 'value', 'inverse_value', 'mcap', 'inverse_mcap']
    SORT_METHODS = ['price', 'value', 'combined']

    def __init__(self, db_path: str = "krx_stock_data.db"):
        """
        Initialize backtester.

        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        n_stocks: int = 5,
        holding_months: int = 12,
        rebalance_months: int = 12,
        weighting: str = 'equal',
        sort_by: str = 'price',
        price_percentile: int = 5,
        value_percentile: int = 5,
        markets: List[str] = None
    ) -> Dict:
        """
        Run backtest on combined screening strategy.

        Args:
            start_date (str): Backtest start date (YYYYMMDD) - first holding period starts here
            end_date (str): Backtest end date (YYYYMMDD)
            n_stocks (int): Number of top stocks to select each period
            holding_months (int): Months to hold portfolio (also used as screening lookback)
            rebalance_months (int): Months between rebalancing
            weighting (str): 'equal', 'value', 'inverse_value', 'mcap', 'inverse_mcap'
            sort_by (str): How to rank top N: 'price' (return), 'value' (trading value), 'combined' (score)
            price_percentile (int): Top percentile for price screening
            value_percentile (int): Top percentile for value screening
            markets (List[str]): Markets to include

        Returns:
            Dict with 'periods', 'summary', 'cumulative' DataFrames
        """
        markets = markets or ['kospi', 'kosdaq']

        if weighting not in self.WEIGHTING_METHODS:
            raise ValueError(f"Invalid weighting: {weighting}. Use one of {self.WEIGHTING_METHODS}")

        if sort_by not in self.SORT_METHODS:
            raise ValueError(f"Invalid sort_by: {sort_by}. Use one of {self.SORT_METHODS}")

        self.logger.info(
            f"Running backtest: {start_date} to {end_date}, "
            f"n={n_stocks}, hold={holding_months}m, rebalance={rebalance_months}m, "
            f"weighting={weighting}, sort_by={sort_by}"
        )

        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')

        # Generate rebalancing dates
        periods = []
        current_dt = start_dt

        while current_dt < end_dt:
            # Screening period: [current - holding_months, current)
            screen_end = current_dt - relativedelta(days=1)
            screen_start = current_dt - relativedelta(months=holding_months)

            # Holding period: [current, current + holding_months)
            hold_start = current_dt
            hold_end = current_dt + relativedelta(months=holding_months) - relativedelta(days=1)

            # Don't go beyond end_date
            if hold_start >= end_dt:
                break

            periods.append({
                'screen_start': screen_start.strftime('%Y%m%d'),
                'screen_end': screen_end.strftime('%Y%m%d'),
                'hold_start': hold_start.strftime('%Y%m%d'),
                'hold_end': min(hold_end, end_dt).strftime('%Y%m%d'),
            })

            current_dt += relativedelta(months=rebalance_months)

        if not periods:
            self.logger.warning("No valid periods found for backtest")
            return {'periods': pd.DataFrame(), 'summary': pd.DataFrame(), 'cumulative': pd.DataFrame()}

        self.logger.info(f"Found {len(periods)} rebalancing periods")

        # Run backtest for each period
        all_results = []
        period_returns = []

        for i, period in enumerate(periods):
            self.logger.info(f"Period {i+1}/{len(periods)}: screen {period['screen_start']}-{period['screen_end']}, hold {period['hold_start']}-{period['hold_end']}")

            result = self._backtest_period(
                period['screen_start'],
                period['screen_end'],
                period['hold_start'],
                period['hold_end'],
                n_stocks,
                weighting,
                sort_by,
                price_percentile,
                value_percentile,
                markets
            )

            if result is not None and len(result) > 0:
                result['period'] = i + 1
                result['screen_start'] = period['screen_start']
                result['screen_end'] = period['screen_end']
                result['hold_start'] = period['hold_start']
                result['hold_end'] = period['hold_end']
                all_results.append(result)

                # Calculate portfolio return for this period
                portfolio_return = (result['weight'] * result['return_pct']).sum()
                period_returns.append({
                    'period': i + 1,
                    'hold_start': period['hold_start'],
                    'hold_end': period['hold_end'],
                    'portfolio_return_pct': round(portfolio_return, 2),
                    'n_stocks': len(result),
                    'best_stock': result.loc[result['return_pct'].idxmax(), 'name'] if len(result) > 0 else None,
                    'best_return_pct': result['return_pct'].max() if len(result) > 0 else None,
                    'worst_stock': result.loc[result['return_pct'].idxmin(), 'name'] if len(result) > 0 else None,
                    'worst_return_pct': result['return_pct'].min() if len(result) > 0 else None,
                })

        if not all_results:
            self.logger.warning("No results from backtest")
            return {'periods': pd.DataFrame(), 'summary': pd.DataFrame(), 'cumulative': pd.DataFrame()}

        # Combine all period results
        periods_df = pd.concat(all_results, ignore_index=True)

        # Summary by period
        summary_df = pd.DataFrame(period_returns)

        # Calculate cumulative returns
        summary_df['cumulative_return_pct'] = (
            (1 + summary_df['portfolio_return_pct'] / 100).cumprod() - 1
        ) * 100
        summary_df['cumulative_return_pct'] = summary_df['cumulative_return_pct'].round(2)

        # Overall statistics
        total_return = summary_df['cumulative_return_pct'].iloc[-1] if len(summary_df) > 0 else 0
        avg_return = summary_df['portfolio_return_pct'].mean() if len(summary_df) > 0 else 0
        win_rate = (summary_df['portfolio_return_pct'] > 0).mean() * 100 if len(summary_df) > 0 else 0

        self.logger.info(f"Backtest complete: Total return={total_return:.2f}%, Avg={avg_return:.2f}%, Win rate={win_rate:.1f}%")

        return {
            'periods': periods_df,
            'summary': summary_df,
            'total_return_pct': round(total_return, 2),
            'avg_period_return_pct': round(avg_return, 2),
            'win_rate_pct': round(win_rate, 1),
            'n_periods': len(summary_df),
        }

    def _backtest_period(
        self,
        screen_start: str,
        screen_end: str,
        hold_start: str,
        hold_end: str,
        n_stocks: int,
        weighting: str,
        sort_by: str,
        price_percentile: int,
        value_percentile: int,
        markets: List[str]
    ) -> Optional[pd.DataFrame]:
        """Run backtest for a single period."""
        conn = self._get_connection()
        market_placeholders = ','.join(['?' for _ in markets])

        # Step 1: Run combined screening to get top stocks
        screen_query = f"""
        SELECT
            dp.stock_code,
            s.current_name as name,
            MAX(dp.market_type) as market_type,
            MAX(CASE WHEN dp.date = sub.first_date THEN dp.closing_price END) as screen_start_price,
            MAX(CASE WHEN dp.date = sub.last_date THEN dp.closing_price END) as screen_end_price,
            SUM(dp.volume) as total_volume,
            SUM(dp.value) as total_value,
            MAX(CASE WHEN dp.date = sub.last_date THEN dp.market_cap END) as market_cap,
            COUNT(*) as trading_days
        FROM daily_prices dp
        INNER JOIN (
            SELECT stock_code, MIN(date) as first_date, MAX(date) as last_date
            FROM daily_prices
            WHERE date BETWEEN ? AND ?
              AND market_type IN ({market_placeholders})
            GROUP BY stock_code
        ) sub ON dp.stock_code = sub.stock_code
        INNER JOIN stocks s ON dp.stock_code = s.stock_code
        WHERE dp.date BETWEEN ? AND ?
          AND dp.market_type IN ({market_placeholders})
        GROUP BY dp.stock_code, s.current_name
        HAVING COUNT(*) >= 50
           AND screen_start_price > 0 AND screen_end_price > 0
        """

        params = ([screen_start, screen_end] + markets +
                  [screen_start, screen_end] + markets)

        screen_df = pd.read_sql_query(screen_query, conn, params=params)

        if len(screen_df) == 0:
            self.logger.warning(f"No stocks found for screening period {screen_start}-{screen_end}")
            return None

        # Calculate screening metrics
        screen_df['price_change_pct'] = (
            (screen_df['screen_end_price'] - screen_df['screen_start_price']) * 100.0 /
            screen_df['screen_start_price']
        ).round(2)

        # Apply percentile filters
        price_threshold = screen_df['price_change_pct'].quantile(1 - price_percentile / 100)
        value_threshold = screen_df['total_value'].quantile(1 - value_percentile / 100)

        filtered_df = screen_df[
            (screen_df['price_change_pct'] >= price_threshold) &
            (screen_df['total_value'] >= value_threshold)
        ].copy()

        if len(filtered_df) == 0:
            self.logger.warning(f"No stocks passed filters for period {screen_start}-{screen_end}")
            return None

        # Sort and take top n based on sort_by parameter
        if sort_by == 'price':
            filtered_df = filtered_df.sort_values('price_change_pct', ascending=False)
        elif sort_by == 'value':
            filtered_df = filtered_df.sort_values('total_value', ascending=False)
        elif sort_by == 'combined':
            # Combined score: normalize both metrics and average them
            filtered_df['price_rank'] = filtered_df['price_change_pct'].rank(pct=True)
            filtered_df['value_rank'] = filtered_df['total_value'].rank(pct=True)
            filtered_df['combined_score'] = (filtered_df['price_rank'] + filtered_df['value_rank']) / 2
            filtered_df = filtered_df.sort_values('combined_score', ascending=False)

        top_stocks = filtered_df.head(n_stocks).copy()

        # Step 2: Get holding period prices
        stock_codes = top_stocks['stock_code'].tolist()
        stock_placeholders = ','.join(['?' for _ in stock_codes])

        # Get prices closest to hold_start and hold_end
        hold_query = f"""
        SELECT
            stock_code,
            MIN(date) as actual_start_date,
            MAX(date) as actual_end_date,
            MAX(CASE WHEN date = (
                SELECT MIN(date) FROM daily_prices
                WHERE stock_code = dp.stock_code AND date >= ?
            ) THEN closing_price END) as hold_start_price,
            MAX(CASE WHEN date = (
                SELECT MAX(date) FROM daily_prices
                WHERE stock_code = dp.stock_code AND date <= ?
            ) THEN closing_price END) as hold_end_price
        FROM daily_prices dp
        WHERE stock_code IN ({stock_placeholders})
          AND date BETWEEN ? AND ?
        GROUP BY stock_code
        """

        hold_params = [hold_start, hold_end] + stock_codes + [hold_start, hold_end]
        hold_df = pd.read_sql_query(hold_query, conn, params=hold_params)

        if len(hold_df) == 0:
            self.logger.warning(f"No holding period data for {hold_start}-{hold_end}")
            return None

        # Merge screening and holding data
        result_df = top_stocks.merge(hold_df, on='stock_code', how='inner')

        if len(result_df) == 0:
            return None

        # Calculate holding period returns
        result_df['return_pct'] = (
            (result_df['hold_end_price'] - result_df['hold_start_price']) * 100.0 /
            result_df['hold_start_price']
        ).round(2)

        # Calculate weights based on weighting method
        result_df['weight'] = self._calculate_weights(result_df, weighting)

        # Select and order columns
        result_df = result_df[[
            'stock_code', 'name', 'market_type',
            'price_change_pct',  # screening period performance
            'total_value', 'market_cap',
            'hold_start_price', 'hold_end_price', 'return_pct',
            'weight'
        ]].copy()

        result_df['total_value_billion'] = (result_df['total_value'] / 1e9).round(2)
        result_df['market_cap_billion'] = (result_df['market_cap'] / 1e8).round(2) if result_df['market_cap'].notna().any() else 0

        return result_df

    def _calculate_weights(self, df: pd.DataFrame, weighting: str) -> pd.Series:
        """Calculate portfolio weights based on weighting method."""
        n = len(df)

        if weighting == 'equal':
            return pd.Series([1.0 / n] * n, index=df.index)

        elif weighting == 'value':
            # Weight by trading value (larger value = higher weight)
            total = df['total_value'].sum()
            return df['total_value'] / total

        elif weighting == 'inverse_value':
            # Inverse value weighting (smaller value = higher weight)
            inv_values = 1.0 / df['total_value']
            return inv_values / inv_values.sum()

        elif weighting == 'mcap':
            # Weight by market cap
            if df['market_cap'].isna().all() or df['market_cap'].sum() == 0:
                return pd.Series([1.0 / n] * n, index=df.index)
            total = df['market_cap'].sum()
            return df['market_cap'] / total

        elif weighting == 'inverse_mcap':
            # Inverse market cap weighting (smaller cap = higher weight)
            if df['market_cap'].isna().all() or df['market_cap'].sum() == 0:
                return pd.Series([1.0 / n] * n, index=df.index)
            inv_mcap = 1.0 / df['market_cap'].replace(0, np.nan)
            return inv_mcap / inv_mcap.sum()

        else:
            return pd.Series([1.0 / n] * n, index=df.index)

    def export_results(
        self,
        results: Dict,
        output_prefix: str = "backtest"
    ) -> Dict[str, str]:
        """
        Export backtest results to CSV files.

        Args:
            results: Output from run_backtest()
            output_prefix: Prefix for output files

        Returns:
            Dict with paths to created files
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}

        # Export detailed periods
        if len(results.get('periods', [])) > 0:
            periods_file = f"{output_prefix}_details_{timestamp}.csv"
            results['periods'].to_csv(periods_file, index=False)
            files['details'] = periods_file
            self.logger.info(f"Exported details to {periods_file}")

        # Export summary
        if len(results.get('summary', [])) > 0:
            summary_file = f"{output_prefix}_summary_{timestamp}.csv"
            results['summary'].to_csv(summary_file, index=False)
            files['summary'] = summary_file
            self.logger.info(f"Exported summary to {summary_file}")

        return files

    def print_report(self, results: Dict):
        """Print a formatted backtest report to console."""
        print("\n" + "=" * 70)
        print("BACKTEST REPORT")
        print("=" * 70)

        print(f"\nTotal Periods: {results.get('n_periods', 0)}")
        print(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
        print(f"Avg Period Return: {results.get('avg_period_return_pct', 0):.2f}%")
        print(f"Win Rate: {results.get('win_rate_pct', 0):.1f}%")

        summary = results.get('summary')
        if summary is not None and len(summary) > 0:
            print("\n" + "-" * 70)
            print("PERIOD SUMMARY")
            print("-" * 70)
            print(f"{'Period':<8} {'Hold Start':<12} {'Hold End':<12} {'Return %':<12} {'Cumulative %':<12}")
            print("-" * 70)

            for _, row in summary.iterrows():
                print(f"{row['period']:<8} {row['hold_start']:<12} {row['hold_end']:<12} "
                      f"{row['portfolio_return_pct']:>10.2f}% {row['cumulative_return_pct']:>11.2f}%")

        periods = results.get('periods')
        if periods is not None and len(periods) > 0:
            print("\n" + "-" * 70)
            print("STOCK SELECTIONS BY PERIOD")
            print("-" * 70)

            for period_num in periods['period'].unique():
                period_data = periods[periods['period'] == period_num]
                hold_start = period_data['hold_start'].iloc[0]
                hold_end = period_data['hold_end'].iloc[0]

                print(f"\nPeriod {period_num}: {hold_start} → {hold_end}")
                print(f"{'Stock':<12} {'Name':<20} {'Weight':<8} {'Return %':<10}")

                for _, row in period_data.iterrows():
                    name = row['name'][:18] if len(row['name']) > 18 else row['name']
                    print(f"{row['stock_code']:<12} {name:<20} {row['weight']*100:>6.1f}% {row['return_pct']:>9.2f}%")

        print("\n" + "=" * 70)
