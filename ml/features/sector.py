"""Sector momentum, breadth, and relative momentum features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import FeatureGroup, register


@register
class SectorFeatures(FeatureGroup):
    name = "sector"
    columns = [
        "sector_momentum_21d",
        "sector_momentum_63d",
        "sector_relative_momentum_20d",
        "sector_relative_momentum_21d",
        "sector_relative_momentum_63d",
        "sector_breadth_21d",
        "sector_constituent_share",
    ]
    dependencies = ["mom_21d", "mom_63d", "constituent_index_count"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sector index returns are merged externally by pipeline.
        # Fill NaN for sector momentum columns if they were merged.
        for col in [
            "sector_momentum_21d",
            "sector_momentum_63d",
            "sector_relative_momentum_20d",
            "sector_relative_momentum_21d",
            "sector_relative_momentum_63d",
        ]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        # Compute cross-sectional sector-relative momentum as fallback
        sec21_xs = df.groupby(["date", "sector"])["mom_21d"].transform("mean")
        sec63_xs = df.groupby(["date", "sector"])["mom_63d"].transform("mean")
        stock_vs_21 = df["mom_21d"] - sec21_xs
        stock_vs_63 = df["mom_63d"] - sec63_xs
        df["sector_relative_momentum_21d"] = df["sector_relative_momentum_21d"].fillna(stock_vs_21)
        df["sector_relative_momentum_63d"] = df["sector_relative_momentum_63d"].fillna(stock_vs_63)

        # Breadth and constituent share
        has_constituent = df["constituent_index_count"] > 0
        if not has_constituent.any():
            df["sector_breadth_21d"] = 0.0
            df["sector_constituent_share"] = 0.0
            return df

        constituent = df.loc[has_constituent]
        breadth = (
            constituent.assign(pos=(constituent["mom_21d"] > 0).astype(float))
            .groupby(["date", "sector"], as_index=False)
            .agg(
                sector_breadth_21d=("pos", "mean"),
                sector_constituent_share=("stock_code", "count"),
            )
        )
        daily_max = breadth.groupby("date")["sector_constituent_share"].transform("max").replace(0, np.nan)
        breadth["sector_constituent_share"] = breadth["sector_constituent_share"] / daily_max

        df = df.merge(breadth, on=["date", "sector"], how="left")
        df["sector_breadth_21d"] = df["sector_breadth_21d"].fillna(0.0)
        df["sector_constituent_share"] = df["sector_constituent_share"].fillna(0.0)
        return df
