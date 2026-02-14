"""LightGBM ranker — the default model."""

from __future__ import annotations

from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseRanker


class LGBMRanker(BaseRanker):
    """LightGBM ranking model (extracted from original MLRanker)."""

    BEST_PARAMS = {
        "objective": "huber",
        "metric": "huber",
        "huber_delta": 1.0,
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 80,
        "verbose": -1,
        "n_estimators": 800,
        "n_jobs": -1,
        "seed": 42,
    }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LGBMRanker":
        params = params or self.BEST_PARAMS.copy()

        X_train = train_df[self.feature_cols].to_numpy()
        y_train = train_df[self.target_col].to_numpy()
        time_weight = self._calculate_time_weights(train_df)
        if sample_weight is not None and time_weight is not None:
            weight = sample_weight * time_weight
            weight = weight / weight.mean()
        elif sample_weight is not None:
            weight = sample_weight
        else:
            weight = time_weight

        train_data = lgb.Dataset(X_train, label=y_train, weight=weight, feature_name=self.feature_cols)
        callbacks = [lgb.log_evaluation(period=100)]

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].to_numpy()
            y_val = val_df[self.target_col].to_numpy()
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            callbacks.append(lgb.early_stopping(self.patience))
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get("n_estimators", 800),
                valid_sets=[val_data],
                callbacks=callbacks,
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=params.get("n_estimators", 800),
                callbacks=callbacks,
            )

        self.logger.info("Trained LGBMRanker with %s samples", len(train_df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(df[self.feature_cols].to_numpy())

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        return (
            pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.model.feature_importance(importance_type="gain"),
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
