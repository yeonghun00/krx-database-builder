"""
Financial Feature Engineering Module

Generates quant features from financial statements with proper null handling:
1. Forward Fill (available_date 기준)
2. Sector Median Imputation
3. Cross-sectional Rank with null handling
4. Missing Indicator features

Usage:
    from features.financial_features import FinancialFeatureGenerator

    generator = FinancialFeatureGenerator('krx_stock_data.db')
    features_df = generator.generate_features(start_date='20240101', end_date='20260128')
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialFeatureGenerator:
    """Generates financial features with professional null handling."""

    # Key financial items mapping
    ITEM_CODES = {
        # Balance Sheet
        'assets': 'ifrs-full_Assets',
        'equity': 'ifrs-full_Equity',
        'liabilities': 'ifrs-full_Liabilities',
        'current_assets': 'ifrs-full_CurrentAssets',
        'current_liabilities': 'ifrs-full_CurrentLiabilities',

        # Income Statement
        'revenue': 'ifrs-full_Revenue',
        'gross_profit': 'ifrs-full_GrossProfit',
        'net_income': 'ifrs-full_ProfitLoss',
        'operating_income': 'dart_OperatingIncomeLoss',
    }

    def __init__(self, db_path: str = 'krx_stock_data.db'):
        """
        Initialize feature generator.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.conn = None

    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # =========================================================================
    # Step 1: Load Raw Data
    # =========================================================================

    def load_daily_prices(self, start_date: str, end_date: str,
                          min_market_cap: float = 500_000_000_000) -> pd.DataFrame:
        """
        Load daily price data with market cap filter.

        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            min_market_cap: Minimum market cap filter (default 5000억)
        """
        query = """
        SELECT
            dp.stock_code,
            dp.date,
            dp.closing_price,
            dp.market_cap,
            dp.volume,
            dp.market_type,
            s.current_name as company_name,
            s.current_sector_type as sector
        FROM daily_prices dp
        JOIN stocks s ON s.stock_code = dp.stock_code
        WHERE dp.date >= ? AND dp.date <= ?
          AND dp.market_cap >= ?
        ORDER BY dp.stock_code, dp.date
        """

        df = pd.read_sql_query(query, self.conn, params=[start_date, end_date, min_market_cap])
        logger.info(f"Loaded {len(df):,} daily price records for {df['stock_code'].nunique()} stocks")
        return df

    def load_financial_data(self, consolidation_type: str = '연결') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load financial statement data (BS/CF and PL).

        Args:
            consolidation_type: '연결' (consolidated) or '별도' (separate)

        Returns:
            (bs_cf_df, pl_df) tuple
        """
        # Load Balance Sheet / Cash Flow items
        bs_cf_query = """
        SELECT
            fp.stock_code,
            fp.fiscal_date,
            fp.available_date,
            fp.industry_name as sector,
            bs.item_code_normalized as item_code,
            bs.amount_current as amount
        FROM financial_periods fp
        JOIN financial_items_bs_cf bs ON bs.period_id = fp.id
        WHERE fp.consolidation_type = ?
          AND bs.item_code_normalized IN (?, ?, ?, ?, ?)
        ORDER BY fp.stock_code, fp.fiscal_date
        """

        bs_cf_df = pd.read_sql_query(
            bs_cf_query, self.conn,
            params=[
                consolidation_type,
                self.ITEM_CODES['assets'],
                self.ITEM_CODES['equity'],
                self.ITEM_CODES['liabilities'],
                self.ITEM_CODES['current_assets'],
                self.ITEM_CODES['current_liabilities'],
            ]
        )

        # Load Income Statement items
        pl_query = """
        SELECT
            fp.stock_code,
            fp.fiscal_date,
            fp.available_date,
            fp.industry_name as sector,
            pl.item_code_normalized as item_code,
            pl.amount_current_ytd as amount_ytd,
            pl.amount_current_qtr as amount_qtr,
            pl.amount_prev_year as amount_prev_year
        FROM financial_periods fp
        JOIN financial_items_pl pl ON pl.period_id = fp.id
        WHERE fp.consolidation_type = ?
          AND pl.item_code_normalized IN (?, ?, ?, ?)
        ORDER BY fp.stock_code, fp.fiscal_date
        """

        pl_df = pd.read_sql_query(
            pl_query, self.conn,
            params=[
                consolidation_type,
                self.ITEM_CODES['revenue'],
                self.ITEM_CODES['gross_profit'],
                self.ITEM_CODES['net_income'],
                self.ITEM_CODES['operating_income'],
            ]
        )

        logger.info(f"Loaded BS/CF: {len(bs_cf_df):,} records, PL: {len(pl_df):,} records")
        return bs_cf_df, pl_df

    # =========================================================================
    # Step 2: Pivot and Prepare Financial Data
    # =========================================================================

    def pivot_bs_data(self, bs_cf_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot BS data from long to wide format."""
        # Pivot to wide format
        pivot = bs_cf_df.pivot_table(
            index=['stock_code', 'fiscal_date', 'available_date', 'sector'],
            columns='item_code',
            values='amount',
            aggfunc='first'
        ).reset_index()

        # Rename columns
        rename_map = {v: k for k, v in self.ITEM_CODES.items() if v in pivot.columns}
        pivot = pivot.rename(columns=rename_map)

        return pivot

    def pivot_pl_data(self, pl_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot PL data and calculate YoY by comparing same quarter last year."""
        # Pivot YTD values
        pivot_ytd = pl_df.pivot_table(
            index=['stock_code', 'fiscal_date', 'available_date', 'sector'],
            columns='item_code',
            values='amount_ytd',
            aggfunc='first'
        ).reset_index()

        # Rename columns
        rename_map = {v: k for k, v in self.ITEM_CODES.items() if v in pivot_ytd.columns}
        pivot_ytd = pivot_ytd.rename(columns=rename_map)

        # Sort by stock and fiscal_date
        pivot_ytd = pivot_ytd.sort_values(['stock_code', 'fiscal_date'])

        # Calculate prev year same quarter values by shifting 4 quarters
        # fiscal_date format: YYYY-MM-DD, same quarter = 1 year ago
        pivot_ytd['fiscal_date_dt'] = pd.to_datetime(pivot_ytd['fiscal_date'])
        pivot_ytd['year'] = pivot_ytd['fiscal_date_dt'].dt.year
        pivot_ytd['quarter'] = pivot_ytd['fiscal_date_dt'].dt.quarter

        # Create prev year same quarter lookup
        for col in ['revenue', 'gross_profit', 'net_income', 'operating_income']:
            if col in pivot_ytd.columns:
                # Merge with self to get previous year same quarter
                prev_year_col = f'{col}_prev_year'

                # Create lookup key: stock_code + quarter
                pivot_ytd['lookup_key'] = pivot_ytd['stock_code'] + '_' + pivot_ytd['quarter'].astype(str)
                pivot_ytd['prev_year'] = pivot_ytd['year'] - 1
                pivot_ytd['prev_lookup_key'] = pivot_ytd['stock_code'] + '_' + pivot_ytd['quarter'].astype(str)

                # Create a mapping of (stock, year, quarter) -> value
                lookup = pivot_ytd.set_index(['stock_code', 'year', 'quarter'])[col].to_dict()

                # Look up previous year same quarter value
                pivot_ytd[prev_year_col] = pivot_ytd.apply(
                    lambda row: lookup.get((row['stock_code'], row['prev_year'], row['quarter'])),
                    axis=1
                )

        # Clean up temporary columns
        pivot_ytd = pivot_ytd.drop(columns=['fiscal_date_dt', 'year', 'quarter',
                                             'lookup_key', 'prev_year', 'prev_lookup_key'],
                                    errors='ignore')

        return pivot_ytd

    # =========================================================================
    # Step 3: Calculate Raw Ratios (Before Null Handling)
    # =========================================================================

    def calculate_raw_ratios(self, bs_df: pd.DataFrame, pl_df: pd.DataFrame,
                              price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate raw financial ratios.

        Note: Use available_date for joining to prevent look-ahead bias.
        """
        # Merge BS and PL on stock_code and fiscal_date
        fin_df = pd.merge(
            bs_df, pl_df,
            on=['stock_code', 'fiscal_date', 'available_date', 'sector'],
            how='outer'
        )

        # Sort for forward fill
        fin_df = fin_df.sort_values(['stock_code', 'available_date'])

        # ===== Valuation Ratios (need market_cap from price data) =====
        # These will be calculated after merge with daily prices

        # ===== Quality Ratios =====
        # ROE = Net Income / Equity
        fin_df['roe'] = fin_df['net_income'] / fin_df['equity']

        # GP/A = Gross Profit / Assets
        fin_df['gpa'] = fin_df['gross_profit'] / fin_df['assets']

        # Operating Margin = Operating Income / Revenue
        fin_df['operating_margin'] = fin_df['operating_income'] / fin_df['revenue']

        # ===== Safety Ratios =====
        # Debt Ratio = Liabilities / Equity
        fin_df['debt_ratio'] = fin_df['liabilities'] / fin_df['equity']

        # Current Ratio = Current Assets / Current Liabilities
        fin_df['current_ratio'] = fin_df['current_assets'] / fin_df['current_liabilities']

        # ===== Growth Ratios =====
        # Revenue YoY Growth
        fin_df['revenue_yoy'] = (fin_df['revenue'] / fin_df['revenue_prev_year']) - 1

        # Operating Income YoY Growth
        fin_df['operating_income_yoy'] = (
            fin_df['operating_income'] / fin_df['operating_income_prev_year']
        ) - 1

        # Net Income YoY Growth
        fin_df['net_income_yoy'] = (fin_df['net_income'] / fin_df['net_income_prev_year']) - 1

        # Handle infinite values from division
        ratio_cols = ['roe', 'gpa', 'operating_margin', 'debt_ratio', 'current_ratio',
                      'revenue_yoy', 'operating_income_yoy', 'net_income_yoy']
        for col in ratio_cols:
            if col in fin_df.columns:
                fin_df[col] = fin_df[col].replace([np.inf, -np.inf], np.nan)

        logger.info(f"Calculated raw ratios for {fin_df['stock_code'].nunique()} stocks")
        return fin_df

    # =========================================================================
    # Step 4: Forward Fill (available_date 기준)
    # =========================================================================

    def forward_fill_to_daily(self, price_df: pd.DataFrame,
                               fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill financial data to daily prices.
        Uses available_date to prevent look-ahead bias.

        This is the critical step - merge_asof joins on available_date,
        so we only use financial data that was publicly available.
        """
        # Ensure date columns are properly formatted
        price_df = price_df.copy()
        fin_df = fin_df.copy()

        price_df['date'] = pd.to_datetime(price_df['date'], format='%Y%m%d')
        fin_df['available_date'] = pd.to_datetime(fin_df['available_date'], format='%Y%m%d')

        # Columns to bring from financial data
        fin_cols = [
            'stock_code', 'available_date', 'sector',
            'assets', 'equity', 'liabilities', 'current_assets', 'current_liabilities',
            'revenue', 'gross_profit', 'net_income', 'operating_income',
            'roe', 'gpa', 'operating_margin', 'debt_ratio', 'current_ratio',
            'revenue_yoy', 'operating_income_yoy', 'net_income_yoy'
        ]
        fin_cols = [c for c in fin_cols if c in fin_df.columns]
        fin_subset = fin_df[fin_cols].drop_duplicates()

        # Process each stock separately for merge_asof (to avoid sorting issues)
        result_dfs = []
        stock_codes = price_df['stock_code'].unique()

        for stock_code in stock_codes:
            price_stock = price_df[price_df['stock_code'] == stock_code].sort_values('date')
            fin_stock = fin_subset[fin_subset['stock_code'] == stock_code].sort_values('available_date')

            if len(fin_stock) == 0:
                # No financial data for this stock - keep price data with nulls
                result_dfs.append(price_stock)
                continue

            # Merge asof - joins each daily price row to the most recent
            # financial data where available_date <= price date
            merged_stock = pd.merge_asof(
                price_stock,
                fin_stock,
                left_on='date',
                right_on='available_date',
                by='stock_code',
                direction='backward'  # Only use past data
            )
            result_dfs.append(merged_stock)

        merged = pd.concat(result_dfs, ignore_index=True)

        # Calculate valuation ratios (need market_cap from prices)
        # P/E = Market Cap / Net Income (TTM)
        merged['pe'] = merged['market_cap'] / merged['net_income']

        # P/S = Market Cap / Revenue (TTM)
        merged['ps'] = merged['market_cap'] / merged['revenue']

        # P/B = Market Cap / Equity
        merged['pb'] = merged['market_cap'] / merged['equity']

        # Handle infinite values
        for col in ['pe', 'ps', 'pb']:
            merged[col] = merged[col].replace([np.inf, -np.inf], np.nan)
            # P/E for loss-making companies should be null (not negative)
            if col == 'pe':
                merged.loc[merged['net_income'] <= 0, 'pe'] = np.nan

        logger.info(f"Forward filled to {len(merged):,} daily records")
        return merged

    # =========================================================================
    # Step 5: Sector Median Imputation
    # =========================================================================

    def impute_sector_median(self, df: pd.DataFrame,
                              feature_cols: List[str]) -> pd.DataFrame:
        """
        Fill null values with sector median (within same date).

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to impute
        """
        df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                continue

            # Calculate sector median for each date
            sector_medians = df.groupby(['date', 'sector_x'])[col].transform('median')

            # Fill nulls with sector median
            null_mask = df[col].isna()
            df.loc[null_mask, col] = sector_medians[null_mask]

            # If still null (sector has no data), use market median
            market_medians = df.groupby('date')[col].transform('median')
            still_null = df[col].isna()
            df.loc[still_null, col] = market_medians[still_null]

            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.warning(f"{col}: {null_count} nulls remain after imputation")

        return df

    # =========================================================================
    # Step 6: Create Missing Indicators
    # =========================================================================

    def create_missing_indicators(self, df: pd.DataFrame,
                                   feature_cols: List[str]) -> pd.DataFrame:
        """
        Create binary indicators for missing values.
        These help the model learn that "missing" itself is informative.
        """
        df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                continue

            indicator_col = f'is_null_{col}'
            df[indicator_col] = df[col].isna().astype(int)

        return df

    # =========================================================================
    # Step 7: Cross-sectional Rank with Null Handling
    # =========================================================================

    def calculate_ranks(self, df: pd.DataFrame, feature_cols: List[str],
                        null_strategy: str = 'median') -> pd.DataFrame:
        """
        Calculate cross-sectional ranks with proper null handling.

        Args:
            df: DataFrame with features
            feature_cols: List of feature columns to rank
            null_strategy: 'median' (assign middle rank), 'bottom' (assign worst rank)
        """
        df = df.copy()

        for col in feature_cols:
            if col not in df.columns:
                continue

            rank_col = f'{col}_rank'

            # Calculate percentile rank (0-1) within each date
            # na_option='keep' excludes nulls from ranking
            df[rank_col] = df.groupby('date')[col].rank(pct=True, na_option='keep')

            # Handle nulls based on strategy
            null_mask = df[rank_col].isna()

            if null_strategy == 'median':
                # Assign middle rank (0.5) to nulls
                df.loc[null_mask, rank_col] = 0.5
            elif null_strategy == 'bottom':
                # Assign worst rank (0 or 1 depending on feature direction)
                df.loc[null_mask, rank_col] = 0.0

        return df

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def generate_features(self, start_date: str = '20240101',
                          end_date: str = '20260128',
                          min_market_cap: float = 500_000_000_000,
                          include_ranks: bool = True,
                          include_missing_indicators: bool = True) -> pd.DataFrame:
        """
        Generate complete feature set.

        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            min_market_cap: Minimum market cap filter (default 5000억)
            include_ranks: Whether to include cross-sectional ranks
            include_missing_indicators: Whether to include is_null_* features

        Returns:
            DataFrame with all features
        """
        if self.conn is None:
            self.connect()

        logger.info("=" * 60)
        logger.info("Starting Financial Feature Generation")
        logger.info("=" * 60)

        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data...")
        price_df = self.load_daily_prices(start_date, end_date, min_market_cap)
        bs_cf_df, pl_df = self.load_financial_data()

        # Step 2: Pivot financial data
        logger.info("Step 2: Pivoting financial data...")
        bs_pivot = self.pivot_bs_data(bs_cf_df)
        pl_pivot = self.pivot_pl_data(pl_df)

        # Step 3: Calculate raw ratios
        logger.info("Step 3: Calculating raw ratios...")
        fin_df = self.calculate_raw_ratios(bs_pivot, pl_pivot, price_df)

        # Step 4: Forward fill to daily
        logger.info("Step 4: Forward filling to daily prices...")
        merged_df = self.forward_fill_to_daily(price_df, fin_df)

        # Define feature columns
        ratio_features = [
            'pe', 'ps', 'pb',  # Valuation
            'roe', 'gpa', 'operating_margin',  # Quality
            'debt_ratio', 'current_ratio',  # Safety
            'revenue_yoy', 'operating_income_yoy', 'net_income_yoy',  # Growth
        ]

        # Step 5: Create missing indicators (before imputation)
        if include_missing_indicators:
            logger.info("Step 5: Creating missing indicators...")
            merged_df = self.create_missing_indicators(merged_df, ratio_features)

        # Step 6: Sector median imputation
        logger.info("Step 6: Imputing with sector medians...")
        merged_df = self.impute_sector_median(merged_df, ratio_features)

        # Step 7: Calculate cross-sectional ranks
        if include_ranks:
            logger.info("Step 7: Calculating cross-sectional ranks...")
            merged_df = self.calculate_ranks(merged_df, ratio_features)

        # Final cleanup
        merged_df = merged_df.sort_values(['stock_code', 'date'])

        # Clean up duplicate sector columns from merge
        if 'sector_x' in merged_df.columns and 'sector_y' in merged_df.columns:
            # Use sector_x (from price data) as primary, fill with sector_y if null
            merged_df['sector'] = merged_df['sector_x'].fillna(merged_df['sector_y'])
            merged_df = merged_df.drop(columns=['sector_x', 'sector_y'])
        elif 'sector_x' in merged_df.columns:
            merged_df = merged_df.rename(columns={'sector_x': 'sector'})
        elif 'sector_y' in merged_df.columns:
            merged_df = merged_df.rename(columns={'sector_y': 'sector'})

        # Summary statistics
        logger.info("=" * 60)
        logger.info("Feature Generation Complete!")
        logger.info(f"  Total records: {len(merged_df):,}")
        logger.info(f"  Unique stocks: {merged_df['stock_code'].nunique()}")
        logger.info(f"  Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
        logger.info(f"  Features: {len([c for c in merged_df.columns if c not in ['stock_code', 'date']])}")
        logger.info("=" * 60)

        return merged_df

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for all features."""
        feature_cols = [c for c in df.columns if c not in
                        ['stock_code', 'date', 'company_name', 'sector', 'sector_x',
                         'sector_y', 'available_date', 'market_type']]

        summary = df[feature_cols].describe().T
        summary['null_pct'] = (df[feature_cols].isna().sum() / len(df) * 100).round(2)
        return summary


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate financial features')
    parser.add_argument('--db-path', default='krx_stock_data.db', help='Database path')
    parser.add_argument('--start-date', default='20240101', help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', default='20260128', help='End date (YYYYMMDD)')
    parser.add_argument('--min-market-cap', type=float, default=500_000_000_000,
                        help='Minimum market cap (default: 5000억)')
    parser.add_argument('--output', default='features.parquet', help='Output file path')

    args = parser.parse_args()

    with FinancialFeatureGenerator(args.db_path) as generator:
        features_df = generator.generate_features(
            start_date=args.start_date,
            end_date=args.end_date,
            min_market_cap=args.min_market_cap
        )

        # Save to file (parquet or csv based on extension)
        output_path = args.output
        if output_path.endswith('.parquet'):
            try:
                features_df.to_parquet(output_path, index=False)
                print(f"Features saved to {output_path}")
            except ImportError:
                # Fallback to CSV if pyarrow not installed
                csv_path = output_path.replace('.parquet', '.csv')
                features_df.to_csv(csv_path, index=False)
                print(f"pyarrow not installed. Features saved to {csv_path}")
        elif output_path.endswith('.csv'):
            features_df.to_csv(output_path, index=False)
            print(f"Features saved to {output_path}")
        else:
            # Default to CSV
            features_df.to_csv(output_path + '.csv', index=False)
            print(f"Features saved to {output_path}.csv")

        # Print summary
        print("\nFeature Summary:")
        print(generator.get_feature_summary(features_df))
