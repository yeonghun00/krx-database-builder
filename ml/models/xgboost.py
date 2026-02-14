"""XGBoost ranker."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseRanker


class XGBRanker(BaseRanker):
    """XGBoost ranking model."""

    BEST_PARAMS = {
        "objective": "reg:pseudohubererror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "min_child_weight": 80,
        "n_estimators": 800,
        "verbosity": 0,
        "seed": 42,
    }

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        params: Optional[Dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBRanker":
        import xgboost as xgb

        params = params or self.BEST_PARAMS.copy()
        n_estimators = params.pop("n_estimators", 800)

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

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=weight, feature_names=self.feature_cols)
        evals = [(dtrain, "train")]

        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].to_numpy()
            y_val = val_df[self.target_col].to_numpy()
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)
            evals.append((dval, "val"))
            self.model = xgb.train(
                params, dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                early_stopping_rounds=self.patience,
                verbose_eval=100,
            )
        else:
            self.model = xgb.train(
                params, dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                verbose_eval=100,
            )

        self.logger.info("Trained XGBRanker with %s samples", len(train_df))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        dmat = xgb.DMatrix(df[self.feature_cols].to_numpy(), feature_names=self.feature_cols)
        return self.model.predict(dmat)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained.")
        scores = self.model.get_score(importance_type="gain")
        return (
            pd.DataFrame([
                {"feature": k, "importance": v} for k, v in scores.items()
            ])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
