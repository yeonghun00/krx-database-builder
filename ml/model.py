"""Backward-compatible re-exports from ml.models.

Existing callers (run_backtest.py, get_picks.py) import from here:
    from ml.model import MLRanker, walk_forward_split
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from ml.models import LGBMRanker, get_model_class  # noqa: F401

# MLRanker is now LGBMRanker — same class, same API.
MLRanker = LGBMRanker


def walk_forward_split(df: pd.DataFrame, train_years: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """Create yearly walk-forward splits with a rolling train window."""
    local = df.copy()
    local["year"] = local["date"].str[:4].astype(int)
    years = sorted(local["year"].unique())
    splits: List[Tuple[pd.DataFrame, pd.DataFrame, Dict]] = []

    for idx in range(train_years, len(years)):
        test_year = years[idx]
        train_start = years[idx - train_years]
        train_end = years[idx - 1]

        train_df = local[(local["year"] >= train_start) & (local["year"] <= train_end)].copy()
        test_df = local[local["year"] == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        info = {
            "train_period": f"{train_start}-{train_end}",
            "test_year": test_year,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }
        splits.append((train_df, test_df, info))

    return splits
