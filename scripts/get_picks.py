#!/usr/bin/env python3
"""Generate latest long/avoid picks with the unified model.

Fixes vs previous version:
  1. Prediction uses today's data (not ~21 days stale)
  2. Uses same target as backtest (risk-adjusted rank)
  3. Loads backtest model by default; retraining uses proper val split + early stopping
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import FeatureEngineer
from ml.model import MLRanker
from ml.models import get_model_class


def _select_target_col(df: pd.DataFrame, horizon: int) -> str:
    """Match the backtest's target: cross-sectional z-score of residual return."""
    fwd_col = f"forward_return_{horizon}d"
    residual_col = f"target_residual_{horizon}d"
    zscore_col = f"target_residual_zscore_{horizon}d"
    base_col = residual_col if residual_col in df.columns else fwd_col
    if base_col in df.columns and zscore_col not in df.columns:
        grp = df.groupby("date")[base_col]
        df[zscore_col] = (df[base_col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        df[zscore_col] = df[zscore_col].fillna(0.0)
    return zscore_col


def _retrain_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    model_path: str,
    time_decay: float,
    patience: int,
    learning_rate: float,
    n_estimators: int,
    model_name: str = "lgbm",
) -> MLRanker:
    """Retrain with proper val split + early stopping (matches backtest logic)."""
    train_years = sorted(df["date"].str[:4].unique())
    val_year = train_years[-1]
    sub_train = df[df["date"].str[:4] != val_year].copy()
    val_df = df[df["date"].str[:4] == val_year].copy()
    if sub_train.empty:
        sub_train, val_df = df.copy(), None

    print(f"[Retrain] train years={train_years[:-1]}, val year={val_year}")
    print(f"[Retrain] train rows={len(sub_train):,}, val rows={len(val_df) if val_df is not None else 0:,}")

    ModelClass = get_model_class(model_name)
    model = ModelClass(
        feature_cols=feature_cols,
        target_col=target_col,
        time_decay=time_decay,
        patience=patience,
    )
    params = model.BEST_PARAMS.copy()
    params["learning_rate"] = learning_rate
    params["n_estimators"] = n_estimators
    params["n_jobs"] = max(1, (os.cpu_count() or 4) // 2)

    model.train(sub_train, val_df, params=params)
    model.save(model_path)
    print(f"[Retrain] saved model to {model_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Get latest picks from unified model")
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "xgboost", "catboost"],
                        help="Model type to use (default: lgbm)")
    parser.add_argument("--db", default="krx_stock_data.db")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD (default: today)")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--bottom", type=int, default=10)
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000)
    parser.add_argument("--model-path", default="models/lgbm_unified.pkl",
                        help="Path to pre-trained model from backtest")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain instead of loading saved model")
    parser.add_argument("--train-start", default="20120101",
                        help="Training data start date when --retrain is used")
    parser.add_argument("--time-decay", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    from datetime import datetime
    end_date = args.end or datetime.now().strftime("%Y%m%d")

    fe = FeatureEngineer(args.db)

    # --- Load or retrain model ---
    model_file = Path(args.model_path)
    if not args.retrain and model_file.exists():
        from ml.models.base import BaseRanker
        print(f"Loading pre-trained model from {model_file}")
        model = BaseRanker.load(str(model_file))
        feature_cols = model.feature_cols
        print(f"Model target: {model.target_col}, features: {len(feature_cols)}")
    elif args.retrain:
        print("Retraining model...")
        train_df = fe.prepare_ml_data(
            start_date=args.train_start,
            end_date=end_date,
            target_horizon=args.horizon,
            min_market_cap=args.min_market_cap,
            use_cache=not args.no_cache,
            n_workers=args.workers,
        )
        if train_df.empty:
            print("No training data available.")
            return
        feature_cols = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in train_df.columns]
        target_col = _select_target_col(train_df, args.horizon)
        print(f"Target: {target_col}")
        model = _retrain_model(
            df=train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=args.horizon,
            model_path=str(model_file),
            time_decay=args.time_decay,
            patience=args.patience,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            model_name=args.model,
        )
    else:
        print(f"No model found at {model_file}. Run backtest first or use --retrain.")
        return

    # --- Build prediction features for the latest date (no forward return needed) ---
    pred_df = fe.prepare_prediction_data(
        end_date=end_date,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
    )
    if pred_df.empty:
        print("No prediction data available.")
        return

    # Check feature alignment
    missing_features = [c for c in feature_cols if c not in pred_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} model features missing from prediction data: {missing_features[:5]}")
        feature_cols = [c for c in feature_cols if c in pred_df.columns]

    pred_df["score"] = model.predict(pred_df)
    pred_df["rank"] = pred_df["score"].rank(ascending=False, method="first").astype(int)

    latest_date = pred_df["date"].max()

    view_cols = [
        "rank",
        "stock_code",
        "name",
        "sector",
        "closing_price",
        "market_cap",
        "score",
        "roe",
        "gpa",
        "mom_21d",
        "sector_relative_momentum_21d",
        "sector_relative_momentum_20d",
        "sector_breadth_21d",
        "liquidity_decay_score",
        "low_price_trap",
        "sector_dispersion_21d",
    ]
    view_cols = [c for c in view_cols if c in pred_df.columns]

    longs = pred_df.nsmallest(args.top, "rank")[view_cols]
    avoids = pred_df.nlargest(args.bottom, "rank")[view_cols]

    print(f"\nBase date: {latest_date}")
    print(f"Universe: {len(pred_df)}")

    print("\n=== Top Picks ===")
    print(longs.to_string(index=False))

    print("\n=== Avoid Picks ===")
    print(avoids.to_string(index=False))

    out_file = Path(f"picks_unified_{latest_date}.csv")
    pred_df[view_cols].sort_values("rank").to_csv(out_file, index=False)
    print(f"\nSaved ranking to {out_file}")


if __name__ == "__main__":
    main()
