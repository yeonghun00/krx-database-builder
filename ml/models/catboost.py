"""CatBoost ranker."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseRanker


class CatBoostRanker(BaseRanker):
    """CatBoost ranking model."""

    BEST_PARAMS = {
        "loss_function": "Huber:delta=1.0",
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "colsample_bylevel": 0.75,
        "min_data_in_leaf": 80,
        "iterations": 800,
        "verbose": 100,
        "random_seed": 42,
    }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CatBoostRanker":
        from catboost import CatBoostRegressor, Pool

        params = params or self.BEST_PARAMS.copy()
        iterations = params.pop("iterations", params.pop("n_estimators", 800))

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

        train_pool = Pool(X_train, label=y_train, weight=weight, feature_names=self.feature_cols)

        model = CatBoostRegressor(iterations=iterations, **params)

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].to_numpy()
            y_val = val_df[self.target_col].to_numpy()
            val_pool = Pool(X_val, label=y_val, feature_names=self.feature_cols)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=self.patience)
        else:
            model.fit(train_pool)

        self.model = model
        self.logger.info("Trained CatBoostRanker with %s samples", len(train_df))
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
                "importance": self.model.get_feature_importance(),
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
