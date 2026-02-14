"""Sector-neutral z-score features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class SectorNeutralFeatures(FeatureGroup):
    name = "sector_neutral"
    columns = [
        "sector_zscore_mom_21d",
        "sector_zscore_turnover_21d",
        "sector_zscore_volatility_21d",
        "sector_zscore_drawdown_252d",
    ]
    dependencies = ["mom_21d", "turnover_21d", "volatility_21d", "drawdown_252d"]

    # (value_col, output_col)
    PAIRS = [
        ("mom_21d", "sector_zscore_mom_21d"),
        ("turnover_21d", "sector_zscore_turnover_21d"),
        ("volatility_21d", "sector_zscore_volatility_21d"),
        ("drawdown_252d", "sector_zscore_drawdown_252d"),
    ]

    def compute(self, df: pd.DataFrame, min_group_size: int = 3) -> pd.DataFrame:
        sector_grp = df.groupby(["date", "sector"])
        market_grp = df.groupby("date")

        for value_col, output_col in self.PAIRS:
            value = pd.to_numeric(df[value_col], errors="coerce")
            group_n = sector_grp[value_col].transform("count")
            group_mean = sector_grp[value_col].transform("mean")
            group_std = sector_grp[value_col].transform("std")

            market_mean = market_grp[value_col].transform("mean")
            market_std = market_grp[value_col].transform("std")

            use_market = (group_n < min_group_size) | group_std.isna() | (group_std <= 1e-12)
            mu = group_mean.where(~use_market, market_mean)
            sigma = group_std.where(~use_market, market_std)
            df[output_col] = ((value - mu) / sigma.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
        return df
