"""Feature engineering pipeline — data loading, merging, and orchestration.

This module contains all the database-level logic (loading prices, financials,
index membership, etc.) and orchestrates feature groups via the registry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from .registry import get_all_feature_columns, get_all_groups, resolve_order


class _FeatureColumnsDescriptor:
    """Descriptor that works on both class and instance access."""

    def __get__(self, obj, objtype=None):
        return get_all_feature_columns()


class FeatureEngineer:
    """Build a unified feature set for model training.

    All data loading and merging happens here. Feature computation is delegated
    to registered FeatureGroup classes.
    """

    CACHE_VERSION = "unified_v46_annualized_fin_20260214"
    BS_ITEM_CODES = {
        "equity": "ifrs-full_Equity",
        "assets": "ifrs-full_Assets",
        "operating_cf": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
    }
    PL_ITEM_CODES = {
        "net_income": "ifrs-full_ProfitLoss",
        "gross_profit": "ifrs-full_GrossProfit",
    }
    BROAD_INDEX_CODES = [
        "KOSPI_코스피",
        "KOSPI_코스피_(외국주포함)",
        "KOSDAQ_코스닥",
        "KOSDAQ_코스닥_(외국주포함)",
    ]

    FEATURE_COLUMNS = _FeatureColumnsDescriptor()

    def __init__(self, db_path: str = "krx_stock_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-256000")
        try:
            conn.execute("PRAGMA mmap_size=2147483648")
        except Exception:
            pass
        self._conn = conn
        return conn

    @staticmethod
    def _to_iso(date_yyyymmdd: str) -> str:
        if len(date_yyyymmdd) == 8 and "-" not in date_yyyymmdd:
            return f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:]}"
        return date_yyyymmdd

    def _ensure_indexes(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_stock_date ON daily_prices(stock_code, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_date_mcap ON daily_prices(date, market_cap)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ic_date_stock ON index_constituents(date, stock_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ic_date_index ON index_constituents(date, index_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fp_stock_avail_consol ON financial_periods(stock_code, available_date, consolidation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bs_item_period ON financial_items_bs_cf(item_code_normalized, period_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pl_item_period ON financial_items_pl(item_code_normalized, period_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_code_date ON delisted_stocks(stock_code, delisting_date)")
            conn.commit()

    def _cache_path(self, cache_key: str) -> str:
        os.makedirs(".cache", exist_ok=True)
        digest = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return f".cache/features_unified_{digest}.parquet"

    def _normalize_delist_date(self, s: pd.Series) -> pd.Series:
        text = s.astype(str).str.strip()
        text = text.str.replace("-", "", regex=False)
        return text.where(text.str.len() == 8, None)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_prices(
        self,
        start_date: str,
        end_date: str,
        min_market_cap: int,
        markets: List[str],
        universe_end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        _ = universe_end_date or end_date
        placeholders = ",".join(["?" for _ in markets])
        with self._connect() as conn:
            price_q = f"""
            SELECT stock_code, date, market_type, closing_price, opening_price,
                   high_price, low_price, volume, value, market_cap
            FROM daily_prices
            WHERE date >= ? AND date <= ?
              AND market_type IN ({placeholders})
              AND closing_price > 0
              AND volume > 0
              AND market_cap >= ?
            ORDER BY stock_code, date
            """
            params = [start_date, end_date] + markets + [min_market_cap]
            prices = pd.read_sql_query(price_q, conn, params=params)
            stocks = pd.read_sql_query(
                """
                SELECT stock_code,
                       current_name AS name,
                       current_market_type AS current_market_type
                FROM stocks
                """,
                conn,
            )
        return prices.merge(stocks, on="stock_code", how="left")

    def _exclude_delisted(self, df: pd.DataFrame) -> pd.DataFrame:
        with self._connect() as conn:
            delisted = pd.read_sql_query(
                "SELECT stock_code, delisting_date FROM delisted_stocks WHERE delisting_date IS NOT NULL",
                conn,
            )
        if delisted.empty:
            return df
        delisted["delisting_date"] = self._normalize_delist_date(delisted["delisting_date"])
        delisted = delisted.dropna(subset=["delisting_date"]).drop_duplicates("stock_code", keep="last")
        merged = df.merge(delisted, on="stock_code", how="left")
        keep = merged["delisting_date"].isna() | (merged["date"] < merged["delisting_date"])
        return merged.loc[keep].drop(columns=["delisting_date"])

    def _load_index_membership(self, start_date: str, end_date: str) -> pd.DataFrame:
        iso_start = self._to_iso(start_date)
        iso_end = self._to_iso(end_date)
        conn = self._connect()
        members = pd.read_sql_query(
            """
            SELECT date AS membership_date, stock_code, COUNT(DISTINCT index_code) AS constituent_index_count
            FROM index_constituents
            WHERE date >= ? AND date <= ?
            GROUP BY membership_date, stock_code
            """,
            conn,
            params=[iso_start, iso_end],
        )
        if not members.empty:
            members["membership_date"] = members["membership_date"].astype(str).str.replace("-", "", regex=False)
        return members

    def _load_sector_membership(self, start_date: str, end_date: str) -> pd.DataFrame:
        iso_start = self._to_iso(start_date)
        iso_end = self._to_iso(end_date)
        conn = self._connect()
        members = pd.read_sql_query(
            """
            WITH counts AS (
                SELECT date AS membership_date, index_code, COUNT(DISTINCT stock_code) AS index_member_count
                FROM index_constituents
                WHERE date >= ? AND date <= ?
                GROUP BY membership_date, index_code
            ),
            candidates AS (
                SELECT
                    ic.date AS membership_date,
                    ic.stock_code,
                    ic.index_code,
                    c.index_member_count,
                    CASE
                        WHEN ic.index_code IN (?, ?, ?, ?) THEN 1
                        ELSE 0
                    END AS is_broad
                FROM index_constituents ic
                JOIN counts c
                  ON c.membership_date = ic.date
                 AND c.index_code = ic.index_code
                WHERE ic.date >= ? AND ic.date <= ?
            ),
            ranked AS (
                SELECT
                    membership_date,
                    stock_code,
                    index_code AS sector_index_code,
                    ROW_NUMBER() OVER (
                        PARTITION BY membership_date, stock_code
                        ORDER BY is_broad ASC, index_member_count ASC, LENGTH(index_code) DESC, index_code ASC
                    ) AS rn
                FROM candidates
            )
            SELECT REPLACE(membership_date, '-', '') AS membership_date, stock_code, sector_index_code
            FROM ranked
            WHERE rn = 1
            """,
            conn,
            params=[
                iso_start, iso_end,
                self.BROAD_INDEX_CODES[0], self.BROAD_INDEX_CODES[1],
                self.BROAD_INDEX_CODES[2], self.BROAD_INDEX_CODES[3],
                iso_start, iso_end,
            ],
        )
        return members

    def _load_market_regime(self, start_date: str, end_date: str, target_horizon: int) -> pd.DataFrame:
        with self._connect() as conn:
            idx = pd.read_sql_query(
                """
                SELECT date, closing_index
                FROM index_daily_prices
                WHERE index_code = 'KOSPI_코스피_200'
                  AND date >= ? AND date <= ?
                ORDER BY date
                """,
                conn,
                params=[start_date, end_date],
            )
        if idx.empty:
            return pd.DataFrame(columns=["date", "market_regime_120d", "market_regime_20d", "market_ret_1d", f"market_forward_return_{target_horizon}d"])
        idx["market_regime_120d"] = idx["closing_index"] / idx["closing_index"].rolling(120, min_periods=60).mean() - 1
        idx["market_regime_20d"] = idx["closing_index"] / idx["closing_index"].rolling(20, min_periods=10).mean() - 1
        idx["market_ret_1d"] = idx["closing_index"].pct_change()
        idx[f"market_forward_return_{target_horizon}d"] = idx["closing_index"].shift(-target_horizon) / idx["closing_index"] - 1
        return idx[["date", "market_regime_120d", "market_regime_20d", "market_ret_1d", f"market_forward_return_{target_horizon}d"]]

    def _load_sector_index_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                """
                SELECT date, index_code, closing_index
                FROM index_daily_prices
                WHERE date >= ? AND date <= ?
                  AND (index_code LIKE 'KOSPI_%' OR index_code LIKE 'KOSDAQ_%')
                ORDER BY index_code, date
                """,
                conn,
                params=[start_date, end_date],
            )
        if df.empty:
            return pd.DataFrame(columns=[
                "date", "sector",
                "sector_momentum_21d", "sector_momentum_63d",
                "sector_relative_momentum_20d", "sector_relative_momentum_21d", "sector_relative_momentum_63d",
            ])
        benchmark = (
            df[df["index_code"] == "KOSPI_코스피"][["date", "closing_index"]]
            .rename(columns={"closing_index": "market_index_close"})
            .copy()
        )
        benchmark["market_mom_20d"] = benchmark["market_index_close"].pct_change(20)
        benchmark["market_mom_21d"] = benchmark["market_index_close"].pct_change(21)
        benchmark["market_mom_63d"] = benchmark["market_index_close"].pct_change(63)

        df = df.copy()
        grouped = df.groupby("index_code")
        df["sector_momentum_21d"] = grouped["closing_index"].pct_change(21, fill_method=None)
        df["sector_momentum_20d"] = grouped["closing_index"].pct_change(20, fill_method=None)
        df["sector_momentum_63d"] = grouped["closing_index"].pct_change(63, fill_method=None)
        df = df.merge(
            benchmark[["date", "market_mom_20d", "market_mom_21d", "market_mom_63d"]],
            on="date", how="left",
        )
        df["sector_relative_momentum_20d"] = df["sector_momentum_20d"] - df["market_mom_20d"]
        df["sector_relative_momentum_21d"] = df["sector_momentum_21d"] - df["market_mom_21d"]
        df["sector_relative_momentum_63d"] = df["sector_momentum_63d"] - df["market_mom_63d"]

        return df.rename(columns={"index_code": "sector"})[
            ["date", "sector",
             "sector_momentum_21d", "sector_momentum_63d",
             "sector_relative_momentum_20d", "sector_relative_momentum_21d", "sector_relative_momentum_63d"]
        ]

    def _load_financial_ratios_pit(self, stock_codes: List[str], end_date: str) -> pd.DataFrame:
        if not stock_codes:
            return pd.DataFrame(columns=["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"])

        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS _elig_fin")
            conn.execute("CREATE TEMP TABLE _elig_fin (stock_code TEXT PRIMARY KEY)")
            conn.executemany("INSERT INTO _elig_fin(stock_code) VALUES (?)", [(c,) for c in stock_codes])

            params_common = [
                self.BS_ITEM_CODES["equity"], self.BS_ITEM_CODES["assets"],
                self.BS_ITEM_CODES["operating_cf"], "연결", end_date,
            ]
            bs_df = pd.read_sql_query(
                """
                SELECT
                    fp.id AS period_id, fp.stock_code,
                    REPLACE(fp.available_date, '-', '') AS available_date,
                    fp.fiscal_date, fp.fiscal_month,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS equity,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS assets,
                    MAX(CASE WHEN bs.item_code_normalized = ? THEN bs.amount_current END) AS operating_cf
                FROM financial_periods fp
                JOIN financial_items_bs_cf bs ON bs.period_id = fp.id
                JOIN _elig_fin e ON e.stock_code = fp.stock_code
                WHERE fp.consolidation_type = ?
                  AND REPLACE(fp.available_date, '-', '') <= ?
                  AND bs.item_code_normalized IN (?, ?, ?)
                GROUP BY fp.id, fp.stock_code, fp.available_date
                """,
                conn,
                params=params_common + [
                    self.BS_ITEM_CODES["equity"], self.BS_ITEM_CODES["assets"],
                    self.BS_ITEM_CODES["operating_cf"],
                ],
            )

            params_common_pl = [self.PL_ITEM_CODES["net_income"], self.PL_ITEM_CODES["gross_profit"], "연결", end_date]
            pl_df = pd.read_sql_query(
                """
                SELECT
                    fp.id AS period_id, fp.stock_code,
                    REPLACE(fp.available_date, '-', '') AS available_date,
                    fp.fiscal_date, fp.fiscal_month,
                    MAX(CASE WHEN pl.item_code_normalized = ? THEN pl.amount_current_ytd END) AS net_income,
                    MAX(CASE WHEN pl.item_code_normalized = ? THEN pl.amount_current_ytd END) AS gross_profit
                FROM financial_periods fp
                JOIN financial_items_pl pl ON pl.period_id = fp.id
                JOIN _elig_fin e ON e.stock_code = fp.stock_code
                WHERE fp.consolidation_type = ?
                  AND REPLACE(fp.available_date, '-', '') <= ?
                  AND pl.item_code_normalized IN (?, ?)
                GROUP BY fp.id, fp.stock_code, fp.available_date
                """,
                conn,
                params=params_common_pl + [self.PL_ITEM_CODES["net_income"], self.PL_ITEM_CODES["gross_profit"]],
            )

        if bs_df.empty and pl_df.empty:
            return pd.DataFrame(columns=["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"])

        fin = bs_df.merge(pl_df, on=["period_id", "stock_code", "available_date", "fiscal_date", "fiscal_month"], how="outer")
        fin["equity"] = pd.to_numeric(fin["equity"], errors="coerce")
        fin["assets"] = pd.to_numeric(fin["assets"], errors="coerce")
        fin["operating_cf"] = pd.to_numeric(fin["operating_cf"], errors="coerce")
        fin["net_income"] = pd.to_numeric(fin["net_income"], errors="coerce")
        fin["gross_profit"] = pd.to_numeric(fin["gross_profit"], errors="coerce")
        fin["fiscal_month"] = pd.to_numeric(fin["fiscal_month"], errors="coerce").fillna(12).astype(int)

        # --- Annualize YTD P&L figures ---
        # amount_current_ytd is cumulative: Q1=3mo, Q2=6mo, Q3=9mo, Annual=12mo.
        # Without annualization, Q1 ROE looks 4x lower than annual ROE.
        fiscal_date_month = pd.to_datetime(fin["fiscal_date"], errors="coerce").dt.month
        months_ytd = ((fiscal_date_month - fin["fiscal_month"]) % 12).replace(0, 12)
        annualization_factor = 12.0 / months_ytd.clip(lower=3)
        fin["net_income"] = fin["net_income"] * annualization_factor
        fin["gross_profit"] = fin["gross_profit"] * annualization_factor
        # operating_cf is also YTD on BS/CF statements — annualize for consistency
        fin["operating_cf"] = fin["operating_cf"] * annualization_factor

        # --- Compute ratios from annualized figures ---
        # Reject negative equity (insolvent companies)
        valid_equity = fin["equity"].where(fin["equity"] > 0, np.nan)
        fin["roe"] = fin["net_income"] / valid_equity
        fin["gpa"] = fin["gross_profit"] / fin["assets"].replace(0, np.nan)

        # --- Dedup: prefer reports with more months (annual > Q3 > Q2 > Q1) ---
        fin["_months_ytd"] = months_ytd
        fin = fin.sort_values(["stock_code", "available_date", "_months_ytd"])
        fin = fin.drop_duplicates(["stock_code", "available_date"], keep="last")

        fin = fin[["stock_code", "available_date", "roe", "gpa", "net_income", "operating_cf"]].copy()
        fin = fin.sort_values(["stock_code", "available_date"])
        return fin

    def _merge_financial_features(self, df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
        if fin_df.empty:
            df["roe"] = np.nan
            df["gpa"] = np.nan
            df["net_income"] = np.nan
            df["operating_cf"] = np.nan
            return df

        df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        right = fin_df.copy()
        right["available_dt"] = pd.to_datetime(right["available_date"], format="%Y%m%d", errors="coerce")
        right = right.dropna(subset=["available_dt"]).sort_values(["stock_code", "available_dt"])

        # merge_asof requires left_on to be globally sorted
        df = df.sort_values(["date_dt", "stock_code"])
        merged = pd.merge_asof(
            df,
            right.sort_values(["available_dt", "stock_code"]),
            left_on="date_dt",
            right_on="available_dt",
            by="stock_code",
            direction="backward",
            suffixes=("", "_fin"),
        )

        # --- Staleness guard: if financial data is >15 months old, treat as missing ---
        staleness = (merged["date_dt"] - merged["available_dt"]).dt.days
        is_stale = staleness > 450  # ~15 months
        for col in ["roe", "gpa", "net_income", "operating_cf"]:
            if col in merged.columns:
                merged.loc[is_stale, col] = np.nan

        merged = merged.drop(columns=["date_dt", "available_dt", "available_date"], errors="ignore")

        for col in ["roe", "gpa"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            sector_med = merged.groupby(["date", "sector"])[col].transform("median")
            market_med = merged.groupby("date")[col].transform("median")
            merged[col] = merged[col].fillna(sector_med).fillna(market_med).fillna(0.0)
            merged[col] = merged[col].clip(-5.0, 5.0)
        merged["net_income"] = pd.to_numeric(merged["net_income"], errors="coerce")
        merged["operating_cf"] = pd.to_numeric(merged["operating_cf"], errors="coerce")
        return merged

    # ------------------------------------------------------------------
    # Universe filters
    # ------------------------------------------------------------------

    def _apply_hard_universe_filters(
        self,
        df: pd.DataFrame,
        min_price: int = 2000,
        liquidity_drop_pct: float = 0.20,
    ) -> pd.DataFrame:
        mask = df["closing_price"] >= min_price
        if "avg_value_20d" in df.columns and mask.any():
            liq_cut = df.groupby("date")["avg_value_20d"].transform(
                lambda s: s.quantile(liquidity_drop_pct)
            )
            mask &= df["avg_value_20d"] >= liq_cut
        if "net_income" in df.columns and "operating_cf" in df.columns:
            bad_accrual = (df["net_income"] > 0) & (df["operating_cf"] < 0)
            mask &= ~bad_accrual
        # Exclude stocks with extreme ROE (likely negative/micro equity)
        if "roe" in df.columns:
            mask &= df["roe"].abs() <= 3.0  # |ROE| > 300% = distressed or data issue
        return df.loc[mask]

    # ------------------------------------------------------------------
    # Rolling beta (needs market_ret_1d which comes from regime merge)
    # ------------------------------------------------------------------

    def _compute_rolling_beta(self, df: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
        out = df.sort_values(["stock_code", "date"])
        if "market_ret_1d" not in out.columns:
            out["rolling_beta_60d"] = 1.0
            return out
        out["_xy"] = out["ret_1d"] * out["market_ret_1d"]
        out["_y2"] = out["market_ret_1d"] ** 2
        g = out.groupby("stock_code", sort=False)
        roll_xy = g["_xy"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_x = g["ret_1d"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_y = g["market_ret_1d"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        roll_y2 = g["_y2"].rolling(window, min_periods=min_periods).mean().droplevel(0)
        cov = roll_xy - roll_x * roll_y
        var = roll_y2 - roll_y ** 2
        beta = cov / var.replace(0, np.nan)
        out = out.drop(columns=["_xy", "_y2"])
        out["rolling_beta_60d"] = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-3.0, 3.0)
        return out

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _add_targets(self, df: pd.DataFrame, target_horizon: int) -> pd.DataFrame:
        out = df.sort_values(["stock_code", "date"])
        g = out.groupby("stock_code")
        fwd_col = f"forward_return_{target_horizon}d"
        rank_col = f"target_rank_{target_horizon}d"
        risk_adj_col = f"target_riskadj_{target_horizon}d"
        risk_adj_rank_col = f"target_riskadj_rank_{target_horizon}d"
        residual_col = f"target_residual_{target_horizon}d"
        residual_rank_col = f"target_residual_rank_{target_horizon}d"

        out[fwd_col] = g["closing_price"].shift(-target_horizon) / out["closing_price"] - 1
        out[rank_col] = out.groupby("date")[fwd_col].rank(method="average", pct=True).fillna(0.5)
        vol = out["volatility_21d"] if "volatility_21d" in out.columns else np.nan
        out[risk_adj_col] = out[fwd_col] / pd.to_numeric(vol, errors="coerce").replace(0, np.nan)
        out[risk_adj_rank_col] = out.groupby("date")[risk_adj_col].rank(method="average", pct=True).fillna(0.5)
        market_fwd_col = f"market_forward_return_{target_horizon}d"
        if market_fwd_col in out.columns and "rolling_beta_60d" in out.columns:
            out[residual_col] = out[fwd_col] - (out["rolling_beta_60d"] * out[market_fwd_col])
            out[residual_rank_col] = out.groupby("date")[residual_col].rank(method="average", pct=True).fillna(0.5)
        else:
            out[residual_col] = out[fwd_col]
            out[residual_rank_col] = out[rank_col]
        return out

    # ------------------------------------------------------------------
    # Year-chunk batching
    # ------------------------------------------------------------------

    def _build_year_chunks(self, start_date: str, end_date: str, target_horizon: int) -> List[dict]:
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        warmup_days = 420
        lookahead_days = max(target_horizon + 7, 42)
        chunks: List[dict] = []
        for year in range(start_dt.year, end_dt.year + 1):
            trim_start = max(start_dt, datetime(year, 1, 1))
            trim_end = min(end_dt, datetime(year, 12, 31))
            if trim_start > trim_end:
                continue
            core_start = trim_start - timedelta(days=warmup_days)
            core_end = trim_end + timedelta(days=lookahead_days)
            chunks.append({
                "year": year,
                "core_start": core_start.strftime("%Y%m%d"),
                "core_end": core_end.strftime("%Y%m%d"),
                "trim_start": trim_start.strftime("%Y%m%d"),
                "trim_end": trim_end.strftime("%Y%m%d"),
            })
        return chunks

    # ------------------------------------------------------------------
    # Core range pipeline (loads data, merges, runs feature groups)
    # ------------------------------------------------------------------

    def _prepare_range_core(
        self,
        start_date: str,
        end_date: str,
        target_horizon: int,
        min_market_cap: int,
        markets: List[str],
        universe_end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        raw = self._load_prices(
            start_date, end_date, min_market_cap, markets,
            universe_end_date=universe_end_date or end_date,
        )
        if raw.empty:
            return raw

        data = self._exclude_delisted(raw)

        # --- External data merges ---
        members = self._load_index_membership(start_date, end_date)
        sector_members = self._load_sector_membership(start_date, end_date)
        regime = self._load_market_regime(start_date, end_date, target_horizon)
        sector_index_returns = self._load_sector_index_returns(start_date, end_date)

        data["membership_date"] = data["date"].str[:6] + "01"
        data = data.merge(members, on=["membership_date", "stock_code"], how="left")
        data = data.merge(sector_members, on=["membership_date", "stock_code"], how="left")
        data["sector"] = data["sector_index_code"].fillna("UNMAPPED_SECTOR_INDEX")
        data["constituent_index_count"] = pd.to_numeric(
            data["constituent_index_count"], errors="coerce"
        ).fillna(0.0)

        fin_pit = self._load_financial_ratios_pit(data["stock_code"].unique().tolist(), end_date)
        data = self._merge_financial_features(data, fin_pit)

        # Sort before feature computation
        data = data.sort_values(["stock_code", "date"])

        # --- Phase 1: price-based feature groups (before filters and external merges) ---
        all_groups = resolve_order(get_all_groups())
        phase1_groups = [g for g in all_groups if g.phase == 1]
        phase2_groups = [g for g in all_groups if g.phase == 2]

        for group_cls in phase1_groups:
            group = group_cls()
            data = group.compute(data)

        # --- Universe filters (after volume/price features are computed) ---
        data = self._apply_hard_universe_filters(data, min_price=2000, liquidity_drop_pct=0.20)

        # --- Merge sector index returns ---
        data = data.merge(sector_index_returns, on=["date", "sector"], how="left")

        # --- Merge market regime ---
        data = data.merge(regime, on="date", how="left")
        data["market_regime_120d"] = data["market_regime_120d"].fillna(0.0)
        data["market_ret_1d"] = data["market_ret_1d"].fillna(0.0)
        data[f"market_forward_return_{target_horizon}d"] = data[f"market_forward_return_{target_horizon}d"].fillna(0.0)

        # --- Rolling beta (needs market_ret_1d from regime) ---
        data = self._compute_rolling_beta(data)

        # --- Fill sector momentum NaNs ---
        for col in [
            "sector_momentum_21d", "sector_momentum_63d",
            "sector_relative_momentum_20d", "sector_relative_momentum_21d",
            "sector_relative_momentum_63d",
        ]:
            if col in data.columns:
                data[col] = data[col].fillna(0.0)

        # --- Phase 2: feature groups needing sector/market data ---
        for group_cls in phase2_groups:
            group = group_cls()
            data = group.compute(data)

        # --- Targets ---
        data = self._add_targets(data, target_horizon)
        data = data.drop(columns=["membership_date"], errors="ignore")
        return data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_ml_data(
        self,
        start_date: str,
        end_date: str,
        target_horizon: int = 21,
        min_market_cap: int = 500_000_000_000,
        markets: Optional[List[str]] = None,
        include_fundamental: bool = True,
        include_macro: bool = True,
        use_cache: bool = True,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        del include_fundamental, include_macro

        markets = markets or ["kospi", "kosdaq"]
        self._ensure_indexes()
        workers = max(1, n_workers or 4)

        feature_columns = self.FEATURE_COLUMNS
        cache_key = (
            f"{self.CACHE_VERSION}_{start_date}_{end_date}_{target_horizon}_{min_market_cap}_"
            f"{'_'.join(sorted(markets))}"
        )
        cache_path = self._cache_path(cache_key)
        if use_cache and os.path.exists(cache_path):
            db_mtime = os.path.getmtime(self.db_path)
            if os.path.getmtime(cache_path) > db_mtime:
                cached = pd.read_parquet(cache_path)
                self.logger.info("Loaded features from cache: %s rows", len(cached))
                return cached

        chunks = self._build_year_chunks(start_date, end_date, target_horizon)
        if not chunks:
            return pd.DataFrame()

        years = [c["year"] for c in chunks]
        print(
            f"[Features] year-batch mode years={years} workers={workers} cache={'on' if use_cache else 'off'}",
            flush=True,
        )

        frames: List[pd.DataFrame] = []
        if workers == 1 or len(chunks) == 1:
            for chunk in chunks:
                print(
                    f"[Features] year={chunk['year']} load core={chunk['core_start']}~{chunk['core_end']} "
                    f"trim={chunk['trim_start']}~{chunk['trim_end']}",
                    flush=True,
                )
                chunk_df = self._prepare_range_core(
                    start_date=chunk["core_start"],
                    end_date=chunk["core_end"],
                    target_horizon=target_horizon,
                    min_market_cap=min_market_cap,
                    markets=markets,
                    universe_end_date=chunk["trim_end"],
                )
                if chunk_df.empty:
                    print(f"[Features] year={chunk['year']} produced 0 rows", flush=True)
                    continue
                chunk_df = chunk_df[
                    (chunk_df["date"] >= chunk["trim_start"]) & (chunk_df["date"] <= chunk["trim_end"])
                ]
                print(f"[Features] year={chunk['year']} rows={len(chunk_df):,}", flush=True)
                frames.append(chunk_df)
        else:
            payloads = [
                {
                    "db_path": self.db_path,
                    "core_start": c["core_start"],
                    "core_end": c["core_end"],
                    "trim_start": c["trim_start"],
                    "trim_end": c["trim_end"],
                    "target_horizon": target_horizon,
                    "min_market_cap": min_market_cap,
                    "markets": markets,
                    "year": c["year"],
                    "universe_end_date": c["trim_end"],
                }
                for c in chunks
            ]
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_prepare_year_chunk_worker, p): p["year"] for p in payloads}
                    for fut in as_completed(futures):
                        year = futures[fut]
                        chunk_df = fut.result()
                        if not chunk_df.empty:
                            frames.append(chunk_df)
                            print(f"[Features] year={year} rows={len(chunk_df):,}", flush=True)
                        else:
                            print(f"[Features] year={year} produced 0 rows", flush=True)
            except (PermissionError, OSError) as exc:
                self.logger.warning(
                    "Multiprocessing unavailable (%s). Falling back to sequential.", exc,
                )
                print(f"[Features] multiprocessing unavailable ({exc}); fallback to sequential", flush=True)
                for p in payloads:
                    print(
                        f"[Features] year={p['year']} load core={p['core_start']}~{p['core_end']} "
                        f"trim={p['trim_start']}~{p['trim_end']}",
                        flush=True,
                    )
                    chunk_df = _prepare_year_chunk_worker(p)
                    if not chunk_df.empty:
                        frames.append(chunk_df)
                        print(f"[Features] year={p['year']} rows={len(chunk_df):,}", flush=True)
                    else:
                        print(f"[Features] year={p['year']} produced 0 rows", flush=True)

        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        data = data.sort_values(["date", "stock_code"]).drop_duplicates(["date", "stock_code"], keep="last")

        fwd_col = f"forward_return_{target_horizon}d"
        required = [c for c in feature_columns if c in data.columns] + [fwd_col]
        data = data.dropna(subset=required)

        feature_cols = [c for c in feature_columns if c in data.columns]
        for col in feature_cols:
            lo = data.groupby("date")[col].transform(lambda s: s.quantile(0.01))
            hi = data.groupby("date")[col].transform(lambda s: s.quantile(0.99))
            data[col] = data[col].clip(lower=lo, upper=hi)

        data = data.sort_values(["date", "stock_code"]).reset_index(drop=True)
        print(f"[Features] merged rows={len(data):,}", flush=True)

        if use_cache:
            data.to_parquet(cache_path, index=False)

        self.logger.info("Prepared unified ML dataset: %s rows", len(data))
        return data

    def prepare_prediction_data(
        self,
        end_date: str,
        target_horizon: int = 21,
        min_market_cap: int = 500_000_000_000,
        markets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        markets = markets or ["kospi", "kosdaq"]
        self._ensure_indexes()

        warmup_days = 420
        start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=warmup_days)
        start_date = start_dt.strftime("%Y%m%d")

        print(f"[Prediction] building features {start_date}~{end_date}", flush=True)

        data = self._prepare_range_core(
            start_date=start_date,
            end_date=end_date,
            target_horizon=target_horizon,
            min_market_cap=min_market_cap,
            markets=markets,
        )

        if data.empty:
            return data

        feature_columns = self.FEATURE_COLUMNS
        feature_required = [c for c in feature_columns if c in data.columns]
        data = data.dropna(subset=feature_required)

        for col in feature_required:
            data[col] = data[col].clip(
                lower=np.nanpercentile(data[col], 1),
                upper=np.nanpercentile(data[col], 99),
            )

        latest_date = data["date"].max()
        pred = data[data["date"] == latest_date].copy()
        print(f"[Prediction] date={latest_date}, universe={len(pred)}", flush=True)
        return pred


def _prepare_year_chunk_worker(payload: dict) -> pd.DataFrame:
    fe = FeatureEngineer(payload["db_path"])
    chunk_df = fe._prepare_range_core(
        start_date=payload["core_start"],
        end_date=payload["core_end"],
        target_horizon=payload["target_horizon"],
        min_market_cap=payload["min_market_cap"],
        markets=payload["markets"],
        universe_end_date=payload.get("universe_end_date", payload["core_end"]),
    )
    if chunk_df.empty:
        return chunk_df
    return chunk_df[
        (chunk_df["date"] >= payload["trim_start"]) & (chunk_df["date"] <= payload["trim_end"])
    ].copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build unified feature dataset")
    parser.add_argument("--start", default="20110101")
    parser.add_argument("--end", default="20260213")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    fe = FeatureEngineer("krx_stock_data.db")
    out = fe.prepare_ml_data(
        start_date=args.start,
        end_date=args.end,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
        use_cache=not args.no_cache,
    )
    print(f"rows={len(out):,}, cols={len(out.columns)}")
    print("features:")
    for feature in fe.FEATURE_COLUMNS:
        print(f"- {feature}")
