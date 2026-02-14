"""Market regime and index membership features."""

from __future__ import annotations

import pandas as pd

from .registry import FeatureGroup, register


@register
class MarketFeatures(FeatureGroup):
    name = "market"
    columns = ["market_regime_120d", "constituent_index_count"]
    dependencies = []  # Merged externally by pipeline

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # These columns are merged from external data sources in the pipeline.
        # This compute is a no-op; they're already present.
        if "market_regime_120d" not in df.columns:
            df["market_regime_120d"] = 0.0
        if "constituent_index_count" not in df.columns:
            df["constituent_index_count"] = 0.0
        return df
