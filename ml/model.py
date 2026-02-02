"""
ML Model Module - LightGBM Ranker for Stock Selection.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MLRanker:
    """LightGBM-based stock ranking model."""

    DEFAULT_PARAMS = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 20, 50],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_estimators': 500,
        'seed': 42
    }

    # Alternative: Regression-based ranking
    REGRESSION_PARAMS = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 500,
        'seed': 42
    }

    def __init__(self, feature_cols: List[str], target_col: str = 'target_rank_21d',
                 model_type: str = 'ranker', time_decay: float = 0.5):
        """
        Initialize ML Ranker.

        Args:
            feature_cols: List of feature column names
            target_col: Target column name
            model_type: 'ranker' (lambdarank) or 'regressor'
            time_decay: Time decay factor (0=no decay, 1=strong decay). Default 0.5
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model_type = model_type
        self.time_decay = time_decay
        self.model = None
        self.logger = logging.getLogger(__name__)

    def _calculate_time_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate time-based sample weights.
        Recent data gets higher weight.

        Args:
            df: DataFrame with 'date' column

        Returns:
            Array of weights (0 to 1, recent = higher)
        """
        if self.time_decay == 0:
            return None  # No weighting

        dates = pd.to_datetime(df['date'], format='%Y%m%d')
        min_date = dates.min()
        max_date = dates.max()
        date_range = (max_date - min_date).days

        if date_range == 0:
            return None

        # Normalize dates to 0-1 range (0=oldest, 1=newest)
        date_position = (dates - min_date).dt.days / date_range

        # Exponential decay: weight = exp(decay * position) normalized
        # Higher decay = more emphasis on recent data
        raw_weights = np.exp(self.time_decay * 2 * date_position)

        # Normalize to mean=1 (so total weight is same as unweighted)
        weights = raw_weights / raw_weights.mean()

        return weights.values

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
              params: Dict = None) -> 'MLRanker':
        """
        Train the model.

        Args:
            train_df: Training data with features and target
            val_df: Validation data (optional)
            params: Model parameters (optional)
        """
        if self.model_type == 'ranker':
            return self._train_ranker(train_df, val_df, params)
        else:
            return self._train_regressor(train_df, val_df, params)

    def _train_ranker(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                      params: Dict = None) -> 'MLRanker':
        """Train LambdaRank model."""
        params = params or self.DEFAULT_PARAMS.copy()

        X_train = train_df[self.feature_cols].values
        y_train = train_df[self.target_col].values

        # Calculate group sizes (samples per date)
        train_groups = train_df.groupby('date').size().values

        train_data = lgb.Dataset(
            X_train, label=y_train, group=train_groups,
            feature_name=self.feature_cols
        )

        callbacks = [lgb.log_evaluation(period=100)]

        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df[self.target_col].values
            val_groups = val_df.groupby('date').size().values

            val_data = lgb.Dataset(
                X_val, label=y_val, group=val_groups,
                reference=train_data
            )

            callbacks.append(lgb.early_stopping(50))

            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get('n_estimators', 500),
                valid_sets=[val_data],
                callbacks=callbacks
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get('n_estimators', 500),
                callbacks=callbacks
            )

        self.logger.info(f"Trained ranker with {len(train_df):,} samples")
        return self

    def _train_regressor(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                         params: Dict = None) -> 'MLRanker':
        """Train regression model with time-decay weighting."""
        params = params or self.REGRESSION_PARAMS.copy()

        X_train = train_df[self.feature_cols].values
        y_train = train_df[self.target_col].values

        # Calculate time-based sample weights (recent = higher)
        weights = self._calculate_time_weights(train_df)

        train_data = lgb.Dataset(
            X_train, label=y_train,
            weight=weights,  # Time decay weights
            feature_name=self.feature_cols
        )

        callbacks = [lgb.log_evaluation(period=100)]

        if val_df is not None:
            X_val = val_df[self.feature_cols].values
            y_val = val_df[self.target_col].values

            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks.append(lgb.early_stopping(50))

            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get('n_estimators', 500),
                valid_sets=[val_data],
                callbacks=callbacks
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get('n_estimators', 500),
                callbacks=callbacks
            )

        decay_info = f", time_decay={self.time_decay}" if self.time_decay > 0 else ""
        self.logger.info(f"Trained regressor with {len(train_df):,} samples{decay_info}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions (scores) for ranking."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X = df[self.feature_cols].values
        scores = self.model.predict(X)
        return scores

    def rank_stocks(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Rank stocks and return top N.

        Args:
            df: DataFrame with features for a single date
            top_n: Number of top stocks to return

        Returns:
            DataFrame with top stocks and their scores
        """
        df = df.copy()
        df['ml_score'] = self.predict(df)
        df['ml_rank'] = df['ml_score'].rank(ascending=False)

        top_stocks = df.nsmallest(top_n, 'ml_rank')

        return top_stocks[['stock_code', 'name', 'closing_price', 'market_cap',
                           'ml_score', 'ml_rank'] + self.feature_cols[:5]]

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        return importance

    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'model_type': self.model_type,
            'saved_at': datetime.now().isoformat()
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'MLRanker':
        """Load model from file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        ranker = cls(
            feature_cols=model_data['feature_cols'],
            target_col=model_data['target_col'],
            model_type=model_data['model_type']
        )
        ranker.model = model_data['model']

        return ranker


def walk_forward_split(df: pd.DataFrame, train_years: int = 5) -> List[Tuple]:
    """
    Generate walk-forward train/test splits.

    Args:
        df: DataFrame with 'date' column (YYYYMMDD format)
        train_years: Number of years for training

    Yields:
        (train_df, test_df, fold_info)
    """
    df = df.copy()
    df['year'] = df['date'].str[:4].astype(int)
    years = sorted(df['year'].unique())

    splits = []

    for i in range(train_years, len(years)):
        test_year = years[i]
        train_start_year = years[max(0, i - train_years)]
        train_end_year = years[i - 1]

        train_mask = (df['year'] >= train_start_year) & (df['year'] <= train_end_year)
        test_mask = df['year'] == test_year

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        fold_info = {
            'train_period': f"{train_start_year}-{train_end_year}",
            'test_year': test_year,
            'train_samples': len(train_df),
            'test_samples': len(test_df)
        }

        splits.append((train_df, test_df, fold_info))

    return splits
