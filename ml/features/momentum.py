"""Momentum features."""

from __future__ import annotations

import pandas as pd

from .registry import FeatureGroup, register


@register
class MomentumFeatures(FeatureGroup):
    name = "momentum"
    columns = ["mom_5d", "mom_21d", "mom_63d", "mom_126d"]
    dependencies = ["closing_price"]
    phase = 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("stock_code")
        df["ret_1d"] = g["closing_price"].pct_change()
        df["mom_5d"] = g["closing_price"].pct_change(5)
        df["mom_21d"] = g["closing_price"].pct_change(21)
        df["mom_63d"] = g["closing_price"].pct_change(63)
        df["mom_126d"] = g["closing_price"].pct_change(126)
        return df
