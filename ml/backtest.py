"""
ML Backtesting Module - Walk-Forward Validation and Performance Analysis.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .features import FeatureEngineer
from .model import MLRanker, walk_forward_split


class MLBacktester:
    """Walk-forward backtester for ML ranking model."""

    BENCHMARK_CODE = '069500'  # KODEX 200 (KOSPI 200 ETF)

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.feature_engineer = FeatureEngineer(db_path)
        self.results = []
        self.models = {}
        self.benchmark_data = None

    def _load_benchmark_data(self, start_date: str, end_date: str, horizon: int = 21):
        """Load KODEX 200 benchmark data with forward returns."""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT date, closing_price
        FROM daily_prices
        WHERE stock_code = ?
          AND date >= ? AND date <= ?
        ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=[self.BENCHMARK_CODE, start_date, end_date])
        conn.close()

        if len(df) == 0:
            self.logger.warning(f"No benchmark data found for {self.BENCHMARK_CODE}")
            return {}

        # Calculate forward returns
        df['forward_return'] = df['closing_price'].pct_change(horizon).shift(-horizon)

        # Create lookup dict: date -> forward_return
        benchmark_dict = dict(zip(df['date'], df['forward_return']))

        self.logger.info(f"Loaded {len(df)} benchmark (KODEX 200) data points")
        return benchmark_dict

    def _validate_market_data(self, df: pd.DataFrame, start_date: str, end_date: str) -> bool:
        """Validate that market data is complete and consistent."""
        if len(df) == 0:
            self.logger.error("No market data available")
            return False

        # Check for missing dates
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = pd.to_datetime(df['date'].unique(), format='%Y%m%d')
        missing_dates = all_dates[~all_dates.isin(trading_dates)]

        if len(missing_dates) > 0:
            self.logger.warning(f"Missing market data for {len(missing_dates)} dates")

        # Check for missing values in key columns
        key_cols = ['closing_price', 'volume', 'market_cap']
        missing_values = df[key_cols].isna().sum().sum()

        if missing_values > 0:
            self.logger.warning(f"Found {missing_values} missing values in key columns")

        return True

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        n_stocks: int = 20,
        holding_days: int = 21,
        train_years: int = 5,
        target_horizon: int = 21,
        weighting: str = 'equal',
        markets: List[str] = None,
        model_type: str = 'regressor',
        min_market_cap: int = 500000000000,
        min_value: int = 10000000000,
        time_decay: float = 0.5
    ) -> Dict:
        """
        Run walk-forward ML backtest.

        Args:
            start_date: Backtest start date (YYYYMMDD)
            end_date: Backtest end date (YYYYMMDD)
            n_stocks: Number of stocks to hold
            holding_days: Days to hold before rebalancing
            train_years: Years of data for training
            target_horizon: Forward return horizon for target
            weighting: Portfolio weighting ('equal', 'score', 'mcap')
            markets: Markets to include
            model_type: 'ranker' or 'regressor'
            min_market_cap: Minimum market cap in KRW (default: 5000억)
            min_value: Minimum daily trading value in KRW (default: 100억)
            time_decay: Weight decay for recent data (0=none, 1=strong). Default 0.5

        Returns:
            Dictionary with backtest results
        """
        self.time_decay = time_decay
        self.logger.info(f"Starting ML backtest: {start_date} to {end_date}")
        self.logger.info(f"Config: n_stocks={n_stocks}, holding={holding_days}d, "
                        f"train_years={train_years}, model={model_type}, "
                        f"min_mcap={min_market_cap/100000000:.0f}억, time_decay={time_decay}")

        # Prepare data
        self.logger.info("Preparing ML data...")
        df = self.feature_engineer.prepare_ml_data(
            start_date, end_date, target_horizon,
            min_market_cap=min_market_cap
        )
        # Filter by markets if specified
        if markets:
            df = df[df["market_type"].isin(markets)]
            return {}

        feature_cols = FeatureEngineer.FEATURE_COLUMNS
        target_col = f'target_rank_{target_horizon}d'

        # Load benchmark (KODEX 200) data
        buffer_start = str(int(start_date[:4]) - 1) + start_date[4:]
        self.benchmark_data = self._load_benchmark_data(buffer_start, end_date, target_horizon)

        # Get unique rebalancing dates
        all_dates = sorted(df['date'].unique())
        rebalance_dates = all_dates[::holding_days]  # Every N days

        self.logger.info(f"Total dates: {len(all_dates)}, Rebalance dates: {len(rebalance_dates)}")

        # Walk-forward backtest
        results = []
        cumulative_return = 1.0

        # Get train/test splits by year
        splits = walk_forward_split(df, train_years)

        for train_df, test_df, fold_info in splits:
            self.logger.info(f"Training on {fold_info['train_period']}, "
                           f"testing on {fold_info['test_year']}")

            # Train model with time decay weighting
            model = MLRanker(feature_cols, target_col, model_type, time_decay=self.time_decay)

            # Use last year of training as validation
            train_years_list = sorted(train_df['date'].str[:4].unique())
            if len(train_years_list) > 1:
                val_year = train_years_list[-1]
                actual_train = train_df[train_df['date'].str[:4] != val_year]
                val_df = train_df[train_df['date'].str[:4] == val_year]
                model.train(actual_train, val_df)
            else:
                model.train(train_df)

            self.models[fold_info['test_year']] = model

            # Get rebalance dates for this test year
            test_rebalance_dates = [d for d in rebalance_dates
                                   if d[:4] == str(fold_info['test_year'])]

            if not test_rebalance_dates:
                continue

            # Simulate trading for each rebalance period
            for i, rebal_date in enumerate(test_rebalance_dates):
                # Get data for this date
                date_df = test_df[test_df['date'] == rebal_date].copy()

                if len(date_df) < n_stocks:
                    continue

                # Get top N stocks by model score
                date_df['ml_score'] = model.predict(date_df)
                top_stocks = date_df.nlargest(n_stocks, 'ml_score').copy()

                # Calculate weights
                if weighting == 'equal':
                    top_stocks['weight'] = 1.0 / n_stocks
                elif weighting == 'score':
                    scores = top_stocks['ml_score'] - top_stocks['ml_score'].min() + 1
                    top_stocks['weight'] = scores / scores.sum()
                elif weighting == 'mcap':
                    top_stocks['weight'] = top_stocks['market_cap'] / top_stocks['market_cap'].sum()

                # Get next rebalance date (or end of year)
                if i + 1 < len(test_rebalance_dates):
                    next_date = test_rebalance_dates[i + 1]
                else:
                    # Find last date in test year
                    year_dates = [d for d in all_dates if d[:4] == str(fold_info['test_year'])]
                    next_date = year_dates[-1] if year_dates else rebal_date

                # Calculate actual returns
                forward_col = f'forward_return_{target_horizon}d'
                if forward_col in top_stocks.columns:
                    # Use actual forward returns from data
                    stock_returns = top_stocks[forward_col].fillna(0)
                    portfolio_return = (stock_returns * top_stocks['weight']).sum()
                else:
                    # Calculate from price data
                    portfolio_return = self._calculate_period_return(
                        top_stocks, rebal_date, next_date
                    )

                cumulative_return *= (1 + portfolio_return)

                # Get benchmark return (universe average)
                benchmark_return = date_df[forward_col].mean() if forward_col in date_df.columns else 0

                results.append({
                    'rebalance_date': rebal_date,
                    'year': fold_info['test_year'],
                    'n_stocks': len(top_stocks),
                    'portfolio_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'alpha': portfolio_return - benchmark_return,
                    'cumulative_return': cumulative_return - 1,
                    'top_stocks': top_stocks['stock_code'].tolist()[:5],
                    'top_names': top_stocks['name'].tolist()[:5] if 'name' in top_stocks.columns else []
                })

        self.results = results
        return self._summarize_results(results)

    def _calculate_period_return(self, stocks_df: pd.DataFrame,
                                  start_date: str, end_date: str) -> float:
        """Calculate return for a holding period from database."""
        # This is a fallback - we typically use pre-computed forward returns
        return 0.0

    def _summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize backtest results."""
        if not results:
            return {}

        results_df = pd.DataFrame(results)

        # Annual summary
        annual = results_df.groupby('year').agg({
            'portfolio_return': 'sum',
            'benchmark_return': 'sum',
            'alpha': 'sum'
        }).reset_index()

        # Overall metrics
        returns = results_df['portfolio_return']
        total_return = results_df['cumulative_return'].iloc[-1]

        # Approximate annual metrics (assuming monthly rebalancing)
        n_years = len(annual)
        annual_return = (1 + total_return) ** (1 / max(n_years, 1)) - 1
        annual_vol = returns.std() * np.sqrt(12)  # Assuming monthly
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).mean()
        alpha_win_rate = (results_df['alpha'] > 0).mean()

        summary = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'alpha_win_rate': alpha_win_rate,
            'total_periods': len(results),
            'annual_results': annual.to_dict('records'),
            'all_results': results
        }

        return summary

    def print_results(self, summary: Dict):
        """Print formatted backtest results."""
        if not summary:
            print("No results to display")
            return

        print("\n" + "=" * 70)
        print("ML FACTOR MODEL BACKTEST RESULTS")
        print("=" * 70)

        # Overall metrics
        print(f"\n{'Metric':<25} {'Value':>15}")
        print("-" * 40)
        print(f"{'Total Return':<25} {summary['total_return']:>14.1%}")
        print(f"{'Annual Return':<25} {summary['annual_return']:>14.1%}")
        print(f"{'Annual Volatility':<25} {summary['annual_volatility']:>14.1%}")
        print(f"{'Sharpe Ratio':<25} {summary['sharpe_ratio']:>15.2f}")
        print(f"{'Max Drawdown':<25} {summary['max_drawdown']:>14.1%}")
        print(f"{'Win Rate':<25} {summary['win_rate']:>14.1%}")
        print(f"{'Alpha Win Rate':<25} {summary['alpha_win_rate']:>14.1%}")

        # Annual breakdown
        print("\n" + "-" * 70)
        print("ANNUAL BREAKDOWN")
        print("-" * 70)
        print(f"{'Year':<8} {'Portfolio':>12} {'Universe':>12} {'Alpha':>12}")
        print("-" * 45)

        for row in summary['annual_results']:
            print(f"{row['year']:<8} {row['portfolio_return']:>11.1%} "
                  f"{row['benchmark_return']:>11.1%} {row['alpha']:>11.1%}")

        # Feature importance (if model available)
        if self.models:
            last_model = list(self.models.values())[-1]
            importance = last_model.feature_importance()

            print("\n" + "-" * 70)
            print("TOP 10 FEATURE IMPORTANCE")
            print("-" * 70)
            for _, row in importance.head(10).iterrows():
                bar = "█" * int(row['importance'] / importance['importance'].max() * 20)
                print(f"{row['feature']:<25} {bar}")

    def get_current_picks(self, date: str = None, n_stocks: int = 20) -> pd.DataFrame:
        """
        Get current stock picks using most recent model.

        Args:
            date: Date for prediction (default: most recent in data)
            n_stocks: Number of stocks to return

        Returns:
            DataFrame with stock picks
        """
        if not self.models:
            raise ValueError("No trained models. Run backtest first.")

        # Use most recent model
        latest_year = max(self.models.keys())
        model = self.models[latest_year]

        # Get current data
        if date is None:
            # Get most recent date from database
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM daily_prices")
            date = cursor.fetchone()[0]
            conn.close()

        # Prepare features for this date
        df = self.feature_engineer.prepare_ml_data(date, date)

        if len(df) == 0:
            self.logger.warning(f"No data for date {date}")
            return pd.DataFrame()

        # Get predictions
        df['ml_score'] = model.predict(df)
        df['ml_rank'] = df['ml_score'].rank(ascending=False).astype(int)

        top_stocks = df.nsmallest(n_stocks, 'ml_rank')

        return top_stocks[['stock_code', 'name', 'market_type', 'closing_price',
                          'market_cap', 'ml_score', 'ml_rank',
                          'mom_12m', 'turnover_ratio', 'volatility_20d']]

    def export_results(self, filepath: str = None) -> str:
        """Export results to CSV."""
        if not self.results:
            self.logger.warning("No results to export")
            return ""

        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"ml_backtest_{timestamp}.csv"

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filepath, index=False)

        self.logger.info(f"Results exported to {filepath}")
        return filepath

    def plot_results(self, summary: Dict, save_path: str = None):
        """Generate performance visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            self.logger.warning("matplotlib not installed. Run: pip install matplotlib")
            return

        if not self.results:
            self.logger.warning("No results to plot")
            return

        results_df = pd.DataFrame(self.results)
        results_df['date'] = pd.to_datetime(results_df['rebalance_date'], format='%Y%m%d')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('ML Factor Model Backtest Results', fontsize=14, fontweight='bold')

        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        cum_portfolio = (1 + results_df['portfolio_return']).cumprod()
        cum_benchmark = (1 + results_df['benchmark_return']).cumprod()

        ax1.plot(results_df['date'], cum_portfolio, label='ML Model', linewidth=2, color='#2E86AB')
        ax1.plot(results_df['date'], cum_benchmark, label='Universe Avg',
                 linewidth=2, color='#A23B72', linestyle='--')
        ax1.fill_between(results_df['date'], cum_portfolio, cum_benchmark,
                        alpha=0.3, color='#2E86AB', where=cum_portfolio >= cum_benchmark)
        ax1.fill_between(results_df['date'], cum_portfolio, cum_benchmark,
                        alpha=0.3, color='#A23B72', where=cum_portfolio < cum_benchmark)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Growth of $1')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 2. Annual Returns Comparison
        ax2 = axes[0, 1]
        annual_df = pd.DataFrame(summary['annual_results'])
        x = np.arange(len(annual_df))
        width = 0.35

        bars1 = ax2.bar(x - width/2, annual_df['portfolio_return'] * 100, width,
                       label='ML Model', color='#2E86AB')
        bars2 = ax2.bar(x + width/2, annual_df['benchmark_return'] * 100, width,
                       label='Universe', color='#A23B72')

        ax2.set_title('Annual Returns Comparison')
        ax2.set_ylabel('Return (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(annual_df['year'].astype(str))
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

        # 3. Drawdown
        ax3 = axes[1, 0]
        cum_returns = (1 + results_df['portfolio_return']).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max * 100

        ax3.fill_between(results_df['date'], drawdown, 0, alpha=0.5, color='#E74C3C')
        ax3.plot(results_df['date'], drawdown, color='#C0392B', linewidth=1)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # 4. Alpha per Period
        ax4 = axes[1, 1]
        colors = ['#2ECC71' if x > 0 else '#E74C3C' for x in results_df['alpha']]
        ax4.bar(range(len(results_df)), results_df['alpha'] * 100, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_title('Alpha per Rebalancing Period')
        ax4.set_ylabel('Alpha (%)')
        ax4.set_xlabel('Period')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add summary stats text box
        stats_text = (
            f"Total Return: {summary['total_return']:.1%}\n"
            f"Annual Return: {summary['annual_return']:.1%}\n"
            f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {summary['max_drawdown']:.1%}\n"
            f"Win Rate: {summary['win_rate']:.1%}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Chart saved to {save_path}")
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"ml_backtest_chart_{timestamp}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Chart saved to {save_path}")

        plt.show()
        return save_path
