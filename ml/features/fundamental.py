"""Fundamental ratio features (ROE, GPA)."""

from __future__ import annotations

import pandas as pd

from .registry import FeatureGroup, register


@register
class FundamentalFeatures(FeatureGroup):
    name = "fundamental"
    columns = ["roe", "gpa", "sector_zscore_roe", "sector_zscore_gpa"]
    dependencies = []  # Merged externally via pipeline before compute

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # ROE/GPA already merged by pipeline via merge_asof.
        # This group computes sector z-scores for fundamentals.
        for col in ["roe", "gpa"]:
            if col not in df.columns:
                df[col] = 0.0
            output_col = f"sector_zscore_{col}"
            value = pd.to_numeric(df[col], errors="coerce")
            sector_grp = df.groupby(["date", "sector"])
            market_grp = df.groupby("date")

            group_n = sector_grp[col].transform("count")
            group_mean = sector_grp[col].transform("mean")
            group_std = sector_grp[col].transform("std")
            market_mean = market_grp[col].transform("mean")
            market_std = market_grp[col].transform("std")

            use_market = (group_n < 3) | group_std.isna() | (group_std <= 1e-12)
            mu = group_mean.where(~use_market, market_mean)
            sigma = group_std.where(~use_market, market_std)
            import numpy as np
            df[output_col] = ((value - mu) / sigma.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        return df
