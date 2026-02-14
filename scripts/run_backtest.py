#!/usr/bin/env python3
"""Unified backtest runner (single model, single feature pipeline)."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.features import FeatureEngineer
from ml.model import MLRanker, walk_forward_split
from ml.models import get_model_class


def _compute_core_stats(results: pd.DataFrame) -> dict:
    """Compute all backtest statistics from results DataFrame."""
    s = {}
    s["n_rebalances"] = len(results)
    s["n_years"] = max(results["year"].nunique(), 1)

    # --- Cumulative ---
    s["total_return"] = (1.0 + results["portfolio_return"]).prod() - 1.0
    s["benchmark_return"] = (1.0 + results["benchmark_return"]).prod() - 1.0
    s["alpha"] = s["total_return"] - s["benchmark_return"]

    # --- Drawdown ---
    cum = (1.0 + results["portfolio_return"]).cumprod()
    drawdown = cum / cum.cummax() - 1.0
    s["max_dd"] = float(drawdown.min())
    s["cum_portfolio"] = cum
    s["cum_benchmark"] = (1.0 + results["benchmark_return"]).cumprod()
    s["drawdown"] = drawdown

    # Underwater duration
    uw = 0
    max_uw = 0
    for flag in (drawdown < 0).tolist():
        uw = uw + 1 if flag else 0
        max_uw = max(max_uw, uw)
    s["max_underwater"] = max_uw

    # --- Annualized ---
    rebals_per_year = max(len(results) / s["n_years"], 1)
    s["ann_vol"] = float(results["portfolio_return"].std() * np.sqrt(rebals_per_year))
    s["ann_return"] = (1.0 + s["total_return"]) ** (1.0 / s["n_years"]) - 1.0
    s["sharpe"] = s["ann_return"] / s["ann_vol"] if s["ann_vol"] > 0 else 0.0
    s["calmar"] = s["ann_return"] / abs(s["max_dd"]) if s["max_dd"] < 0 else np.nan

    # --- Trade statistics ---
    wins = results[results["alpha"] > 0]["alpha"]
    losses = results[results["alpha"] <= 0]["alpha"]
    s["hit_rate"] = float((results["alpha"] > 0).mean())
    s["avg_win"] = float(wins.mean()) if len(wins) > 0 else 0.0
    s["avg_loss"] = float(losses.mean()) if len(losses) > 0 else 0.0
    total_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    total_loss = float(losses.abs().sum()) if len(losses) > 0 else 0.0
    s["profit_factor"] = total_profit / total_loss if total_loss > 0 else np.inf
    s["win_loss_ratio"] = abs(s["avg_win"] / s["avg_loss"]) if s["avg_loss"] != 0 else np.inf

    # --- IC ---
    if "ic_spearman" in results.columns:
        s["ic_mean"] = float(results["ic_spearman"].mean())
        ic_std = float(results["ic_spearman"].std())
        s["ic_ir"] = s["ic_mean"] / ic_std if ic_std > 0 else np.nan
    else:
        s["ic_mean"] = np.nan
        s["ic_ir"] = np.nan

    # --- Market capture ---
    up_mask = results["benchmark_return"] > 0
    down_mask = results["benchmark_return"] < 0
    if up_mask.sum() > 0:
        s["up_capture"] = float(results.loc[up_mask, "portfolio_return"].mean() / results.loc[up_mask, "benchmark_return"].mean())
    else:
        s["up_capture"] = np.nan
    if down_mask.sum() > 0:
        s["down_capture"] = float(results.loc[down_mask, "portfolio_return"].mean() / results.loc[down_mask, "benchmark_return"].mean())
    else:
        s["down_capture"] = np.nan

    # --- Rolling Sharpe ---
    s["rolling_sharpe_12"] = (
        results["portfolio_return"].rolling(12).mean()
        / results["portfolio_return"].rolling(12).std().replace(0, np.nan)
        * np.sqrt(12)
    )

    # --- Rolling Beta (portfolio vs benchmark) ---
    cov_window = 12
    port_r = results["portfolio_return"]
    bench_r = results["benchmark_return"]
    roll_cov = port_r.rolling(cov_window).cov(bench_r)
    roll_var = bench_r.rolling(cov_window).var()
    s["rolling_beta"] = (roll_cov / roll_var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    s["overall_beta"] = float(port_r.cov(bench_r) / bench_r.var()) if bench_r.var() > 0 else np.nan

    # --- Turnover ---
    s["avg_turnover"] = float(results["turnover"].mean()) if "turnover" in results.columns else np.nan
    s["total_tx_cost"] = float(results["transaction_cost"].sum()) if "transaction_cost" in results.columns else np.nan

    # --- Annual stats ---
    annual = results.groupby("year").agg(
        ann_port=("portfolio_return", lambda x: (1 + x).prod() - 1),
        ann_bench=("benchmark_return", lambda x: (1 + x).prod() - 1),
        ann_vol=("portfolio_return", lambda x: x.std() * np.sqrt(max(len(x), 1))),
    )
    annual["ann_alpha"] = annual["ann_port"] - annual["ann_bench"]
    annual["ann_sharpe"] = annual["ann_port"] / annual["ann_vol"].replace(0, np.nan)
    s["annual"] = annual

    # --- Quintile ---
    if {"q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"}.issubset(results.columns):
        s["q_means"] = results[["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"]].mean()
        s["q_mono"] = bool(
            s["q_means"]["q5_ret"] > s["q_means"]["q4_ret"] > s["q_means"]["q3_ret"]
            > s["q_means"]["q2_ret"] > s["q_means"]["q1_ret"]
        )
    else:
        s["q_means"] = None
        s["q_mono"] = False

    return s


def summarize(results: pd.DataFrame, sector_rows: list, output_path: str = "backtest_report.png") -> None:
    """Print enhanced summary + generate visual report."""
    if results.empty:
        print("No backtest results were generated.")
        return

    s = _compute_core_stats(results)
    sector_df = pd.DataFrame(sector_rows) if sector_rows else pd.DataFrame()

    # ======================================================================
    # TEXT SUMMARY
    # ======================================================================
    print("\n" + "=" * 70)
    print("  BACKTEST REPORT")
    print("=" * 70)

    print(f"\n{'--- Overview ---':^70}")
    print(f"  Rebalances: {s['n_rebalances']}  |  Years: {s['n_years']}")
    print(f"  Total Return:     {s['total_return']:>8.2%}   Benchmark: {s['benchmark_return']:>8.2%}")
    print(f"  Alpha:            {s['alpha']:>8.2%}   Hit Rate:  {s['hit_rate']:>8.2%}")
    print(f"  Ann. Return:      {s['ann_return']:>8.2%}   Ann. Vol:  {s['ann_vol']:>8.2%}")
    print(f"  Sharpe:           {s['sharpe']:>8.2f}   Calmar:    {s['calmar']:>8.2f}" if pd.notna(s['calmar']) else f"  Sharpe:           {s['sharpe']:>8.2f}   Calmar:        N/A")
    print(f"  Max Drawdown:     {s['max_dd']:>8.2%}   Max Underwater: {s['max_underwater']} rebals")

    print(f"\n{'--- Trade Statistics (매매 세부 지표) ---':^70}")
    print(f"  Win Rate:         {s['hit_rate']:>8.2%}   (벤치마크 대비 이긴 비율)")
    pf_str = f"{s['profit_factor']:.2f}" if np.isfinite(s['profit_factor']) else "INF"
    print(f"  Profit Factor:    {pf_str:>8s}   (총이익/총손실, 1.5+ 우수, 2.0+ 성배)")
    wl_str = f"{s['win_loss_ratio']:.2f}" if np.isfinite(s['win_loss_ratio']) else "INF"
    print(f"  Win/Loss Ratio:   {wl_str:>8s}   (평균이익/평균손실)")
    print(f"  Avg Win:          {s['avg_win']:>8.2%}   Avg Loss:  {s['avg_loss']:>8.2%}")
    if pd.notna(s['avg_turnover']):
        print(f"  Avg Turnover:     {s['avg_turnover']:>8.2%}   Total Tx Cost: {s['total_tx_cost']:>8.2%}")

    print(f"\n{'--- Market Regime Analysis (하락장 방어력) ---':^70}")
    uc_str = f"{s['up_capture']:.2f}" if pd.notna(s['up_capture']) else "N/A"
    dc_str = f"{s['down_capture']:.2f}" if pd.notna(s['down_capture']) else "N/A"
    print(f"  Up Capture:       {uc_str:>8s}   (시장 1% 상승시 포트폴리오 변동)")
    print(f"  Down Capture:     {dc_str:>8s}   (0.7 이하면 우수한 방어력)")
    beta_str = f"{s['overall_beta']:.2f}" if pd.notna(s['overall_beta']) else "N/A"
    print(f"  Overall Beta:     {beta_str:>8s}   (0.5 이하면 독자적 알파)")

    print(f"\n{'--- IC & Quintile ---':^70}")
    if pd.notna(s['ic_mean']):
        ic_ir_str = f"{s['ic_ir']:.2f}" if pd.notna(s['ic_ir']) else "N/A"
        print(f"  Mean IC:          {s['ic_mean']:>8.4f}   IC IR:     {ic_ir_str:>8s}")
    if s['q_means'] is not None:
        q_str = "  ".join([f"Q{i}={s['q_means'][f'q{i}_ret']:.2%}" for i in range(1, 6)])
        print(f"  Quintile: {q_str}")
        print(f"  Monotonic: {'PASS' if s['q_mono'] else 'FAIL'}")

    print(f"\n{'--- Annual Sharpe (연도별 안정성) ---':^70}")
    for yr, row in s['annual'].iterrows():
        sh = f"{row['ann_sharpe']:.2f}" if pd.notna(row['ann_sharpe']) else "N/A"
        print(f"  {yr}:  Return={row['ann_port']:>7.2%}  Alpha={row['ann_alpha']:>7.2%}  Sharpe={sh:>6s}")

    # Sector attribution text
    if not sector_df.empty:
        print(f"\n{'--- Sector Attribution (섹터 귀인 분석) ---':^70}")
        sec_agg = sector_df.groupby("sector").agg(
            total_contribution=("contribution", "sum"),
            avg_weight=("weight", "mean"),
            appearances=("date", "count"),
        ).sort_values("total_contribution", ascending=False)
        total_contrib = sec_agg["total_contribution"].sum()
        for sec_name, row in sec_agg.head(10).iterrows():
            pct = row["total_contribution"] / total_contrib * 100 if total_contrib != 0 else 0
            short_name = sec_name.split("_")[-1] if "_" in sec_name else sec_name
            print(f"  {short_name:<20s}  기여도={row['total_contribution']:>7.2%}  ({pct:>5.1f}%)  비중={row['avg_weight']:>5.1%}")

    print("\n" + "=" * 70)

    # ======================================================================
    # VISUAL REPORT
    # ======================================================================
    _generate_visual_report(results, s, sector_df, output_path)


def _generate_visual_report(results: pd.DataFrame, s: dict, sector_df: pd.DataFrame, output_path: str) -> None:
    """Generate a multi-panel PNG report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("[Report] matplotlib not installed, skipping visual report.")
        return

    # Try Korean font
    try:
        import matplotlib.font_manager as fm
        korean_fonts = [f.name for f in fm.fontManager.ttflist if any(
            k in f.name for k in ["Nanum", "Malgun", "Apple SD", "NanumGothic", "AppleGothic"]
        )]
        if korean_fonts:
            plt.rcParams["font.family"] = korean_fonts[0]
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    # Color palette
    C_PORT = "#2563EB"   # blue
    C_BENCH = "#9CA3AF"  # gray
    C_ALPHA = "#10B981"  # green
    C_NEG = "#EF4444"    # red
    C_WARN = "#F59E0B"   # amber
    C_BG = "#F8FAFC"

    fig = plt.figure(figsize=(22, 28), facecolor="white", dpi=100)
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.28,
                  left=0.06, right=0.96, top=0.95, bottom=0.03)

    dates = pd.to_datetime(results["date"], format="%Y%m%d", errors="coerce")

    # ── Panel 1: Cumulative Returns + Drawdown ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(C_BG)
    ax1.plot(dates, s["cum_portfolio"], color=C_PORT, linewidth=2, label="Portfolio")
    ax1.plot(dates, s["cum_benchmark"], color=C_BENCH, linewidth=1.5, label="Benchmark", linestyle="--")
    ax1.fill_between(dates, s["cum_portfolio"], s["cum_benchmark"],
                     where=s["cum_portfolio"] >= s["cum_benchmark"],
                     alpha=0.15, color=C_ALPHA, interpolate=True)
    ax1.fill_between(dates, s["cum_portfolio"], s["cum_benchmark"],
                     where=s["cum_portfolio"] < s["cum_benchmark"],
                     alpha=0.15, color=C_NEG, interpolate=True)
    ax1.set_title("Cumulative Returns (Portfolio vs Benchmark)", fontsize=14, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax1.grid(True, alpha=0.3)
    # Drawdown on twin axis
    ax1b = ax1.twinx()
    ax1b.fill_between(dates, s["drawdown"], 0, alpha=0.25, color=C_NEG, label="Drawdown")
    ax1b.set_ylim(min(s["drawdown"].min() * 1.3, -0.05), 0.02)
    ax1b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1b.set_ylabel("Drawdown", fontsize=10, color=C_NEG)
    ax1b.tick_params(axis="y", colors=C_NEG)

    # ── Panel 2: Annual Performance ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(C_BG)
    annual = s["annual"]
    yrs = annual.index.astype(str)
    x = np.arange(len(yrs))
    w = 0.35
    bars_port = ax2.bar(x - w / 2, annual["ann_port"] * 100, w, color=C_PORT, label="Portfolio", zorder=3)
    bars_bench = ax2.bar(x + w / 2, annual["ann_bench"] * 100, w, color=C_BENCH, label="Benchmark", zorder=3)
    # Alpha markers
    for i, (yr, row) in enumerate(annual.iterrows()):
        color = C_ALPHA if row["ann_alpha"] > 0 else C_NEG
        ax2.annotate(f"{row['ann_alpha']:+.1%}", (i, max(row['ann_port'], row['ann_bench']) * 100 + 1),
                     ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(yrs, rotation=45, fontsize=8)
    ax2.set_title("Annual Returns & Alpha", fontsize=13, fontweight="bold")
    ax2.set_ylabel("%", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Annual Sharpe ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(C_BG)
    sharpe_vals = annual["ann_sharpe"].fillna(0)
    colors_sharpe = [C_ALPHA if v > 0.5 else (C_WARN if v > 0 else C_NEG) for v in sharpe_vals]
    ax3.bar(yrs, sharpe_vals, color=colors_sharpe, zorder=3)
    ax3.axhline(0.5, color=C_ALPHA, linewidth=1, linestyle="--", alpha=0.7, label="Sharpe=0.5")
    ax3.axhline(0, color="black", linewidth=0.5)
    for i, v in enumerate(sharpe_vals):
        ax3.annotate(f"{v:.2f}", (i, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax3.set_xticks(range(len(yrs)))
    ax3.set_xticklabels(yrs, rotation=45, fontsize=8)
    ax3.set_title("Annual Sharpe Ratio (연도별 안정성)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Trade Statistics ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(C_BG)
    ax4.axis("off")
    pf_val = s["profit_factor"]
    pf_color = C_ALPHA if pf_val >= 1.5 else (C_WARN if pf_val >= 1.0 else C_NEG)
    wr_color = C_ALPHA if s["hit_rate"] >= 0.5 else C_WARN
    dc_val = s["down_capture"]
    dc_color = C_ALPHA if (pd.notna(dc_val) and dc_val < 0.7) else C_WARN

    stats_lines = [
        ("Trade Statistics (매매 세부 지표)", "", 16, "bold", "black"),
        ("", "", 6, "normal", "black"),
        ("Win Rate (승률)", f"{s['hit_rate']:.1%}", 14, "bold", wr_color),
        ("Profit Factor (수익인자)", f"{pf_val:.2f}" if np.isfinite(pf_val) else "INF", 14, "bold", pf_color),
        ("Win/Loss Ratio (손익비)", f"{s['win_loss_ratio']:.2f}" if np.isfinite(s['win_loss_ratio']) else "INF", 14, "bold", C_PORT),
        ("Avg Win", f"{s['avg_win']:+.2%}", 12, "normal", C_ALPHA),
        ("Avg Loss", f"{s['avg_loss']:+.2%}", 12, "normal", C_NEG),
        ("", "", 10, "normal", "black"),
        ("Market Regime (하락장 방어력)", "", 16, "bold", "black"),
        ("", "", 6, "normal", "black"),
        ("Up Capture", f"{s['up_capture']:.2f}" if pd.notna(s['up_capture']) else "N/A", 14, "bold", C_PORT),
        ("Down Capture", f"{dc_val:.2f}" if pd.notna(dc_val) else "N/A", 14, "bold", dc_color),
        ("Overall Beta", f"{s['overall_beta']:.2f}" if pd.notna(s['overall_beta']) else "N/A", 14, "bold", C_PORT),
    ]
    y_pos = 0.97
    for label, value, fsize, fweight, color in stats_lines:
        if value:
            ax4.text(0.05, y_pos, label, fontsize=fsize, fontweight="normal",
                     transform=ax4.transAxes, va="top")
            ax4.text(0.95, y_pos, value, fontsize=fsize, fontweight=fweight, color=color,
                     transform=ax4.transAxes, va="top", ha="right")
        else:
            ax4.text(0.05, y_pos, label, fontsize=fsize, fontweight=fweight, color=color,
                     transform=ax4.transAxes, va="top")
        y_pos -= 0.075 if fsize >= 14 else 0.04

    # ── Panel 5: Quintile Returns ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(C_BG)
    if s["q_means"] is not None:
        q_vals = [s["q_means"][f"q{i}_ret"] * 100 for i in range(1, 6)]
        q_labels = ["Q1\n(Worst)", "Q2", "Q3", "Q4", "Q5\n(Best)"]
        q_colors = [C_NEG, "#F97316", C_WARN, "#34D399", C_ALPHA]
        bars = ax5.bar(q_labels, q_vals, color=q_colors, zorder=3, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, q_vals):
            ax5.annotate(f"{val:.2f}%", (bar.get_x() + bar.get_width() / 2, val),
                         ha="center", va="bottom" if val >= 0 else "top",
                         fontsize=11, fontweight="bold")
        ax5.axhline(0, color="black", linewidth=0.5)
        mono_text = "MONOTONIC" if s["q_mono"] else "NOT MONOTONIC"
        mono_color = C_ALPHA if s["q_mono"] else C_NEG
        ax5.set_title(f"Quintile Returns — {mono_text}", fontsize=13, fontweight="bold", color=mono_color)
        ax5.set_ylabel("Mean Forward Return (%)", fontsize=10)
    else:
        ax5.text(0.5, 0.5, "No quintile data", ha="center", va="center", fontsize=14)
        ax5.set_title("Quintile Returns", fontsize=13, fontweight="bold")
    ax5.grid(True, axis="y", alpha=0.3)

    # ── Panel 6: Rolling Sharpe + Rolling Beta ──
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.set_facecolor(C_BG)
    roll_sh = s["rolling_sharpe_12"]
    valid = roll_sh.notna()
    if valid.any():
        ax6.plot(dates[valid], roll_sh[valid], color=C_PORT, linewidth=1.5, label="Rolling Sharpe (12p)")
        ax6.axhline(0, color="black", linewidth=0.5)
        ax6.axhline(0.5, color=C_ALPHA, linewidth=1, linestyle="--", alpha=0.5)
        ax6.fill_between(dates[valid], roll_sh[valid], 0,
                         where=roll_sh[valid] > 0, alpha=0.15, color=C_ALPHA, interpolate=True)
        ax6.fill_between(dates[valid], roll_sh[valid], 0,
                         where=roll_sh[valid] <= 0, alpha=0.15, color=C_NEG, interpolate=True)
    # Rolling beta on twin
    roll_b = s["rolling_beta"]
    valid_b = roll_b.notna()
    if valid_b.any():
        ax6b = ax6.twinx()
        ax6b.plot(dates[valid_b], roll_b[valid_b], color=C_WARN, linewidth=1, alpha=0.7, label="Rolling Beta")
        ax6b.axhline(1.0, color=C_WARN, linewidth=0.8, linestyle=":", alpha=0.5)
        ax6b.set_ylabel("Beta", fontsize=10, color=C_WARN)
        ax6b.tick_params(axis="y", colors=C_WARN)
        ax6b.set_ylim(-0.5, 2.5)
    ax6.set_title("Rolling Sharpe & Beta (시장 상관 변화)", fontsize=13, fontweight="bold")
    ax6.set_ylabel("Sharpe", fontsize=10)
    ax6.legend(loc="upper left", fontsize=9)
    ax6.grid(True, alpha=0.3)

    # ── Panel 7: Sector Attribution ──
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.set_facecolor(C_BG)
    if not sector_df.empty:
        sec_agg = sector_df.groupby("sector")["contribution"].sum().sort_values(ascending=True)
        top_sec = sec_agg.tail(12)
        short_names = [n.split("_")[-1] if "_" in n else n for n in top_sec.index]
        colors_sec = [C_ALPHA if v > 0 else C_NEG for v in top_sec.values]
        ax7.barh(short_names, top_sec.values * 100, color=colors_sec, zorder=3, height=0.7)
        ax7.axvline(0, color="black", linewidth=0.5)
        for i, v in enumerate(top_sec.values):
            ax7.annotate(f"{v:.2%}", (v * 100, i), ha="left" if v > 0 else "right",
                         va="center", fontsize=8, fontweight="bold")
        ax7.set_title("Sector Attribution (섹터별 기여도)", fontsize=13, fontweight="bold")
        ax7.set_xlabel("Cumulative Contribution (%)", fontsize=10)

        # HHI annotation
        if "sector_hhi" in results.columns:
            avg_hhi = results["sector_hhi"].mean()
            conc_label = "HIGH" if avg_hhi > 0.25 else ("MED" if avg_hhi > 0.15 else "LOW")
            conc_color = C_NEG if avg_hhi > 0.25 else (C_WARN if avg_hhi > 0.15 else C_ALPHA)
            ax7.annotate(f"Avg HHI={avg_hhi:.3f} ({conc_label})", (0.98, 0.02),
                         xycoords="axes fraction", ha="right", va="bottom",
                         fontsize=10, fontweight="bold", color=conc_color,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax7.text(0.5, 0.5, "No sector data", ha="center", va="center", fontsize=14)
        ax7.set_title("Sector Attribution", fontsize=13, fontweight="bold")
    ax7.grid(True, axis="x", alpha=0.3)

    # ── Panel 8: Up/Down Market Scatter ──
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.set_facecolor(C_BG)
    bench_pct = results["benchmark_return"] * 100
    port_pct = results["portfolio_return"] * 100
    up = results["benchmark_return"] > 0
    ax8.scatter(bench_pct[up], port_pct[up], color=C_ALPHA, alpha=0.5, s=30, label="Bull rebalance", zorder=3)
    ax8.scatter(bench_pct[~up], port_pct[~up], color=C_NEG, alpha=0.5, s=30, label="Bear rebalance", zorder=3)
    lims = [min(bench_pct.min(), port_pct.min()) - 1, max(bench_pct.max(), port_pct.max()) + 1]
    ax8.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="y=x")
    ax8.set_xlim(lims)
    ax8.set_ylim(lims)
    ax8.set_xlabel("Benchmark Return (%)", fontsize=10)
    ax8.set_ylabel("Portfolio Return (%)", fontsize=10)
    ax8.set_title("Up/Down Capture (시장 상승 vs 하락 구간)", fontsize=13, fontweight="bold")
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # ── Panel 9: IC time series ──
    ax9 = fig.add_subplot(gs[4, 1])
    ax9.set_facecolor(C_BG)
    if "ic_spearman" in results.columns:
        ic = results["ic_spearman"]
        ic_colors = [C_ALPHA if v > 0 else C_NEG for v in ic]
        ax9.bar(dates, ic, color=ic_colors, width=20, alpha=0.6, zorder=3)
        ic_rolling = ic.rolling(6).mean()
        valid_ic = ic_rolling.notna()
        if valid_ic.any():
            ax9.plot(dates[valid_ic], ic_rolling[valid_ic], color=C_PORT, linewidth=2, label="6-period MA")
        ax9.axhline(0, color="black", linewidth=0.5)
        ax9.axhline(s["ic_mean"], color=C_WARN, linewidth=1, linestyle="--", alpha=0.7,
                     label=f"Mean IC={s['ic_mean']:.4f}")
        ax9.set_title("Spearman IC Over Time (예측력 변화)", fontsize=13, fontweight="bold")
        ax9.set_ylabel("IC", fontsize=10)
        ax9.legend(fontsize=9)
    else:
        ax9.text(0.5, 0.5, "No IC data", ha="center", va="center", fontsize=14)
        ax9.set_title("Spearman IC", fontsize=13, fontweight="bold")
    ax9.grid(True, alpha=0.3)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved visual report to {out}")


def _run_fold(payload: dict) -> dict:
    """Run one walk-forward fold in a worker process."""
    train_df: pd.DataFrame = payload["train_df"]
    test_df: pd.DataFrame = payload["test_df"]
    info: dict = payload["info"]
    feature_cols: list[str] = payload["feature_cols"]
    target_col: str = payload["target_col"]
    fwd_col: str = payload["fwd_col"]
    top_n: int = payload["top_n"]
    rebalance_days: int = payload["rebalance_days"]
    time_decay: float = payload["time_decay"]
    model_jobs: int = payload["model_jobs"]
    buy_fee_rate: float = payload["buy_fee_rate"]
    sell_fee_rate: float = payload["sell_fee_rate"]
    learning_rate: float = payload["learning_rate"]
    n_estimators: int = payload["n_estimators"]
    patience: int = payload["patience"]
    min_market_cap: int = payload["min_market_cap"]
    stress_mode: bool = payload["stress_mode"]
    vol_exclude_pct: float = payload["vol_exclude_pct"]
    sector_neutral_score: bool = payload["sector_neutral_score"]
    buy_rank: int = payload["buy_rank"]
    hold_rank: int = payload["hold_rank"]
    embargo_days: int = payload["embargo_days"]
    cash_out_enabled: bool = payload.get("cash_out", False)
    model_class_name: str = payload.get("model_class", "lgbm")
    print(
        f"[Fold {info['test_year']}] start "
        f"(train={info['train_period']}, train_rows={len(train_df):,}, test_rows={len(test_df):,})",
        flush=True,
    )

    train_years = sorted(train_df["date"].str[:4].unique())
    val_year = train_years[-1]
    sub_train = train_df[train_df["date"].str[:4] != val_year]
    val_df = train_df[train_df["date"].str[:4] == val_year]
    if sub_train.empty:
        sub_train, val_df = train_df, None
    # Purged training: enforce embargo gap before test period.
    all_dates = sorted(pd.concat([train_df["date"], test_df["date"]]).unique())
    test_start = min(test_df["date"])
    if test_start in all_dates:
        idx = all_dates.index(test_start)
        if idx > embargo_days:
            cutoff = all_dates[idx - embargo_days]
            sub_train = sub_train[sub_train["date"] < cutoff].copy()
            if val_df is not None:
                val_df = val_df[val_df["date"] < cutoff].copy()
    if sub_train.empty:
        sub_train = train_df.copy()
        val_df = None

    ModelClass = get_model_class(model_class_name)
    model = ModelClass(feature_cols=feature_cols, target_col=target_col, time_decay=time_decay)
    params = model.BEST_PARAMS.copy()
    params["n_jobs"] = model_jobs
    params["learning_rate"] = learning_rate
    params["n_estimators"] = n_estimators
    model.patience = patience
    model.train(sub_train, val_df, params=params)
    if val_df is not None and len(val_df) > 100 and fwd_col in val_df.columns:
        val_probe = val_df.copy()
        val_probe["score"] = model.predict(val_probe)
        score_rank_probe = val_probe["score"].rank(method="first", pct=True)
        val_probe["quintile"] = np.ceil(score_rank_probe * 5).clip(1, 5).astype(int)
        qv = val_probe.groupby("quintile")[fwd_col].mean()
        mono_ok = False
        if all(q in qv.index for q in [1, 2, 3, 4, 5]):
            mono_ok = bool(qv.loc[5] > qv.loc[4] > qv.loc[3] > qv.loc[2] > qv.loc[1])
        if not mono_ok:
            print(f"[Fold {info['test_year']}] quintiles not monotonic (diagnostic only, no retry)", flush=True)
    print(f"[Fold {info['test_year']}] model trained", flush=True)

    rows = []
    sector_rows = []
    pick_rows = []
    date_groups = {d: g.copy() for d, g in test_df.groupby("date", sort=True)}
    rebalance_dates = sorted(date_groups.keys())[::rebalance_days]
    prev_holdings: set[str] = set(payload.get("prev_holdings", []))
    for d in rebalance_dates:
        day_df = date_groups[d].copy()
        # PIT universe on rebalance date
        day_df = day_df[day_df["market_cap"] >= min_market_cap].copy()
        if stress_mode and 0 < vol_exclude_pct < 1 and "volatility_21d" in day_df.columns and len(day_df) > 10:
            vol_cut = day_df["volatility_21d"].quantile(1.0 - vol_exclude_pct)
            day_df = day_df[day_df["volatility_21d"] <= vol_cut].copy()
        if len(day_df) < top_n:
            continue
        # Cash-out: when KOSPI200 is below 20-day MA, halve positions
        effective_top_n = top_n
        cash_weight = 0.0
        if cash_out_enabled and "market_regime_20d" in day_df.columns:
            regime_val = day_df["market_regime_20d"].iloc[0]
            if pd.notna(regime_val) and regime_val < 0:
                effective_top_n = max(top_n // 2, 5)
                cash_weight = 1.0 - (effective_top_n / top_n)
        day_df["score"] = model.predict(day_df)
        if sector_neutral_score and "sector" in day_df.columns:
            sec_mean = day_df.groupby("sector")["score"].transform("mean")
            sec_std = day_df.groupby("sector")["score"].transform("std").replace(0, np.nan)
            day_df["score_rank"] = ((day_df["score"] - sec_mean) / sec_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            day_df["score_rank"] = day_df["score"]
        day_df["rank_pos"] = day_df["score_rank"].rank(ascending=False, method="first")
        score_rank = day_df["score_rank"].rank(method="first", pct=True)
        day_df["quintile"] = np.ceil(score_rank * 5).clip(1, 5).astype(int)
        qret = day_df.groupby("quintile")[fwd_col].mean()
        q1 = float(qret.get(1, np.nan))
        q2 = float(qret.get(2, np.nan))
        q3 = float(qret.get(3, np.nan))
        q4 = float(qret.get(4, np.nan))
        q5 = float(qret.get(5, np.nan))
        q_mono = int(q5 > q4 > q3 > q2 > q1) if np.all(pd.notna([q1, q2, q3, q4, q5])) else 0
        ic = day_df[["score_rank", fwd_col]].corr(method="spearman").iloc[0, 1]
        ic = float(ic) if pd.notna(ic) else np.nan
        # Hysteresis buffer:
        # 1) Keep existing holdings while rank <= hold_rank (wide gate)
        # 2) Buy new names ONLY when rank <= buy_rank (tight gate)
        # 3) Fill remaining slots ONLY from rank <= hold_rank (not unrestricted)
        # 4) New stock must score higher than worst keeper to justify cost
        keep_pool = day_df[(day_df["stock_code"].isin(prev_holdings)) & (day_df["rank_pos"] <= hold_rank)].copy()
        already_in = set(keep_pool["stock_code"])
        buy_candidates = day_df[
            (~day_df["stock_code"].isin(already_in)) & (day_df["rank_pos"] <= buy_rank)
        ].copy()

        # Cost-aware replacement: new stock must outscore the worst keeper
        # by enough to justify the round-trip transaction cost.
        if len(keep_pool) > 0 and len(buy_candidates) > 0:
            worst_keeper_score = keep_pool["score_rank"].min()
            score_edge = buy_fee_rate + sell_fee_rate  # ~1% round-trip as score hurdle
            buy_candidates = buy_candidates[
                buy_candidates["score_rank"] > worst_keeper_score + score_edge
            ].copy()

        picks = pd.concat([keep_pool, buy_candidates], ignore_index=True)
        picks = picks.sort_values("score_rank", ascending=False).drop_duplicates("stock_code")

        # Fill remaining slots — but ONLY from rank <= hold_rank (not from entire universe)
        if len(picks) < effective_top_n:
            fill_pool = day_df[
                (~day_df["stock_code"].isin(set(picks["stock_code"])))
                & (day_df["rank_pos"] <= hold_rank)
            ].copy()
            fill_pool = fill_pool.sort_values("score_rank", ascending=False)
            picks = pd.concat([picks, fill_pool.head(effective_top_n - len(picks))], ignore_index=True)
        picks = picks.head(effective_top_n).copy()
        current_holdings = set(picks["stock_code"].tolist())
        stock_ret = float(picks[fwd_col].mean())
        # Blend with cash (0% return) when cash-out is active
        port_ret = stock_ret * (1.0 - cash_weight)
        bench_ret = float(day_df[fwd_col].mean())

        if not prev_holdings:
            turnover = 1.0  # initial deployment from cash
            transaction_cost = buy_fee_rate
        else:
            overlap = len(prev_holdings & current_holdings)
            turnover = 1.0 - (overlap / max(effective_top_n, 1))
            transaction_cost = turnover * (buy_fee_rate + sell_fee_rate)

        net_port_ret = (1.0 + port_ret) * (1.0 - transaction_cost) - 1.0
        prev_holdings = current_holdings
        sec = (
            picks.groupby("sector", as_index=False)
            .agg(
                n=("stock_code", "count"),
                sector_forward_return=(fwd_col, "mean"),
            )
            .sort_values("n", ascending=False)
        )
        sec["weight"] = sec["n"] / max(effective_top_n, 1)
        sec["contribution"] = sec["weight"] * sec["sector_forward_return"]
        top_sector = str(sec.iloc[0]["sector"]) if len(sec) > 0 else "N/A"
        top_sector_weight = float(sec.iloc[0]["weight"]) if len(sec) > 0 else 0.0
        sector_hhi = float((sec["weight"] ** 2).sum()) if len(sec) > 0 else np.nan
        for _, srow in sec.iterrows():
            sector_rows.append(
                {
                    "date": d,
                    "test_year": info["test_year"],
                    "sector": srow["sector"],
                    "weight": float(srow["weight"]),
                    "sector_forward_return": float(srow["sector_forward_return"]),
                    "contribution": float(srow["contribution"]),
                }
            )

        # Collect per-pick details for optional CSV export
        pick_detail_cols = ["stock_code", "name", "sector", "closing_price", "market_cap",
                            "score", "score_rank", "rank_pos"]
        pick_detail_cols = [c for c in pick_detail_cols if c in picks.columns]
        for _, prow in picks.iterrows():
            pick_rows.append({
                "date": d,
                "test_year": info["test_year"],
                **{c: prow[c] for c in pick_detail_cols},
            })

        rows.append(
            {
                "date": d,
                "year": int(d[:4]),
                "portfolio_return": net_port_ret,
                "portfolio_return_gross": port_ret,
                "benchmark_return": bench_ret,
                "alpha": net_port_ret - bench_ret,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "ic_spearman": ic,
                "q1_ret": q1,
                "q2_ret": q2,
                "q3_ret": q3,
                "q4_ret": q4,
                "q5_ret": q5,
                "q_monotonic": q_mono,
                "top_sector": top_sector,
                "top_sector_weight": top_sector_weight,
                "sector_hhi": sector_hhi,
                "top_codes": ",".join(picks["stock_code"].head(10).tolist()),
                "train_period": info["train_period"],
                "test_year": info["test_year"],
            }
        )

    print(
        f"[Fold {info['test_year']}] done "
        f"(rebalance_points={len(rebalance_dates)}, result_rows={len(rows)})",
        flush=True,
    )
    return {
        "test_year": info["test_year"],
        "rows": rows,
        "sector_rows": sector_rows,
        "pick_rows": pick_rows,
        "final_holdings": list(prev_holdings),
    }


def run(args: argparse.Namespace) -> None:
    effective_buy_fee = args.buy_fee
    effective_sell_fee = args.sell_fee
    if getattr(args, "no_sector_neutral", False):
        args.sector_neutral_score = False
    if getattr(args, "no_cash_out", False):
        args.cash_out = False
    effective_sector_neutral = args.sector_neutral_score or args.stress_mode
    if args.stress_mode:
        effective_buy_fee = 1.0
        effective_sell_fee = 1.0

    print(f"[Backtest] loading data {args.start}~{args.end} ...", flush=True)
    fe = FeatureEngineer(args.db)
    df = fe.prepare_ml_data(
        start_date=args.start,
        end_date=args.end,
        target_horizon=args.horizon,
        min_market_cap=args.min_market_cap,
        use_cache=not args.no_cache,
        n_workers=args.workers,
    )

    if df.empty:
        print("No ML data available for the requested range.")
        return
    if args.stress_mode and 0 < args.vol_exclude_pct < 1 and "volatility_21d" in df.columns:
        vol_cut = df.groupby("date")["volatility_21d"].transform(
            lambda s: s.quantile(1.0 - args.vol_exclude_pct)
        )
        df = df[df["volatility_21d"] <= vol_cut].copy()
    print(f"[Backtest] feature rows={len(df):,}, cols={len(df.columns)}", flush=True)

    feature_cols = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in df.columns]
    fwd_col = f"forward_return_{args.horizon}d"
    residual_col = f"target_residual_{args.horizon}d"
    # Cross-sectional z-score of residual returns (forward_return - beta*market_return).
    # Removes market regime effect so the model learns stock-specific alpha.
    zscore_col = f"target_residual_zscore_{args.horizon}d"
    base_col = residual_col if residual_col in df.columns else fwd_col
    if base_col in df.columns:
        grp = df.groupby("date")[base_col]
        df[zscore_col] = (df[base_col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        df[zscore_col] = df[zscore_col].fillna(0.0)
    target_col = zscore_col

    splits = walk_forward_split(df, train_years=args.train_years)
    if not splits:
        print("No walk-forward splits available. Widen date range or reduce train years.")
        return
    cpu_count = os.cpu_count() or 4
    workers = max(1, args.workers)
    split_years = [int(s[2]["test_year"]) for s in splits]

    if args.model_jobs > 0:
        model_jobs = args.model_jobs
    else:
        model_jobs = max(1, cpu_count // workers) if workers > 1 else -1

    # Resolve model params for summary
    ModelClassInfo = get_model_class(args.model)
    model_params = ModelClassInfo.BEST_PARAMS.copy()
    model_params["learning_rate"] = args.learning_rate
    model_params["n_estimators"] = args.n_estimators

    # ── Config Summary ──
    print("\n" + "=" * 70)
    print("  BACKTEST CONFIG")
    print("=" * 70)
    print(f"\n{'--- Data ---':^70}")
    print(f"  Period:           {args.start} ~ {args.end}")
    print(f"  Universe:         market_cap >= {args.min_market_cap:,}")
    print(f"  Rows:             {len(df):,}   Features: {len(feature_cols)}")
    print(f"\n{'--- Model ---':^70}")
    print(f"  Type:             {args.model}")
    print(f"  Objective:        {model_params.get('objective', 'N/A')} (huber_delta={model_params.get('huber_delta', 'N/A')})")
    print(f"  Target:           {target_col}")
    print(f"  Target Source:    {base_col}")
    print(f"  LR / Estimators:  {args.learning_rate} / {args.n_estimators}")
    print(f"  Early Stop:       patience={args.patience}")
    print(f"  Leaves / Leaf:    {model_params.get('num_leaves', 'N/A')} / min_data={model_params.get('min_data_in_leaf', 'N/A')}")
    print(f"  Feature Frac:     {model_params.get('feature_fraction', 'N/A')}")
    print(f"  Time Decay:       {args.time_decay}")
    print(f"\n{'--- Walk-Forward ---':^70}")
    print(f"  Train Window:     {args.train_years} years (rolling)")
    print(f"  Folds:            {len(splits)}   Test Years: {split_years}")
    print(f"  Embargo:          {args.embargo_days} days")
    print(f"\n{'--- Portfolio ---':^70}")
    print(f"  Top N:            {args.top_n}")
    print(f"  Rebalance:        every {args.rebalance_days} trading days")
    print(f"  Buy Rank:         <= {args.buy_rank}   Hold Rank: <= {args.hold_rank}")
    print(f"  Fees:             buy={effective_buy_fee:.2f}%  sell={effective_sell_fee:.2f}%")
    print(f"  Sector Neutral:   {effective_sector_neutral}")
    cash_out_flag = getattr(args, "cash_out", False)
    print(f"  Cash-Out (20d):   {cash_out_flag}")
    if args.stress_mode:
        print(f"  Stress Mode:      ON (vol_exclude={args.vol_exclude_pct:.0%})")
    print("=" * 70 + "\n", flush=True)

    rows = []
    sector_rows = []
    fold_payloads = [
        {
            "train_df": train_df,
            "test_df": test_df,
            "info": info,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "fwd_col": fwd_col,
            "top_n": args.top_n,
            "rebalance_days": args.rebalance_days,
            "time_decay": args.time_decay,
            "model_jobs": model_jobs,
            "buy_fee_rate": effective_buy_fee / 100.0,
            "sell_fee_rate": effective_sell_fee / 100.0,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "patience": args.patience,
            "min_market_cap": args.min_market_cap,
            "stress_mode": args.stress_mode,
            "vol_exclude_pct": args.vol_exclude_pct,
            "sector_neutral_score": effective_sector_neutral,
            "buy_rank": args.buy_rank,
            "hold_rank": args.hold_rank,
            "embargo_days": args.embargo_days,
            "cash_out": args.cash_out,
            "model_class": args.model,
        }
        for train_df, test_df, info in splits
    ]

    if workers == 1 or len(fold_payloads) == 1:
        # Sequential: carry holdings across folds to avoid 100% turnover at fold boundaries
        fold_results = []
        carry_holdings: list[str] = []
        for p in fold_payloads:
            p["prev_holdings"] = carry_holdings
            res = _run_fold(p)
            fold_results.append(res)
            carry_holdings = res.get("final_holdings", [])
    else:
        fold_results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_fold, p) for p in fold_payloads]
            for fut in as_completed(futures):
                fold_results.append(fut.result())
                done_years = sorted([int(r["test_year"]) for r in fold_results])
                print(f"[Backtest] completed folds so far: {done_years}", flush=True)

    fold_results.sort(key=lambda x: x["test_year"])
    pick_rows = []
    for res in fold_results:
        rows.extend(res["rows"])
        sector_rows.extend(res.get("sector_rows", []))
        pick_rows.extend(res.get("pick_rows", []))

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["date", "test_year"]).reset_index(drop=True)
    if not results.empty:
        out_csv = Path(args.output)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_csv, index=False)
        print(f"Saved detailed results to {out_csv}")
        rolling = results[["date", "portfolio_return"]].copy()
        rolling["rolling_12_sharpe"] = (
            rolling["portfolio_return"].rolling(12).mean()
            / rolling["portfolio_return"].rolling(12).std().replace(0, np.nan)
            * np.sqrt(12)
        )
        rolling_out = out_csv.with_name(out_csv.stem + "_rolling_sharpe.csv")
        rolling.to_csv(rolling_out, index=False)
        print(f"Saved rolling Sharpe to {rolling_out}")
        quintile_summary = results[["q1_ret", "q2_ret", "q3_ret", "q4_ret", "q5_ret"]].mean().to_frame("mean_return")
        quintile_out = out_csv.with_name(out_csv.stem + "_quintiles.csv")
        quintile_summary.to_csv(quintile_out)
        print(f"Saved quintile summary to {quintile_out}")
        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            sector_out = out_csv.with_name(out_csv.stem + "_sector_attribution.csv")
            sector_df.to_csv(sector_out, index=False)
            print(f"Saved sector attribution to {sector_out}")
        if args.save_picks and pick_rows:
            picks_df = pd.DataFrame(pick_rows).sort_values(["date", "rank_pos"])
            picks_out = out_csv.with_name(out_csv.stem + "_picks.csv")
            picks_df.to_csv(picks_out, index=False)
            print(f"Saved picks to {picks_out} ({len(picks_df)} rows)")

    if splits:
        latest_split = max(splits, key=lambda x: x[2]["test_year"])
        latest_train_df = latest_split[0]
        train_years = sorted(latest_train_df["date"].str[:4].unique())
        val_year = train_years[-1]
        sub_train = latest_train_df[latest_train_df["date"].str[:4] != val_year]
        val_df = latest_train_df[latest_train_df["date"].str[:4] == val_year]
        if sub_train.empty:
            sub_train, val_df = latest_train_df, None
        FinalModelClass = get_model_class(args.model)
        latest_model = FinalModelClass(feature_cols=feature_cols, target_col=target_col, time_decay=args.time_decay)
        params = latest_model.BEST_PARAMS.copy()
        params["n_jobs"] = max(1, cpu_count // 2)
        params["learning_rate"] = args.learning_rate
        params["n_estimators"] = args.n_estimators
        latest_model.patience = args.patience
        latest_model.train(sub_train, val_df, params=params)
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        latest_model.save(str(model_path))
        print(f"Saved unified model to {model_path}")

    report_path = Path(args.output).with_suffix(".png")
    summarize(results, sector_rows, output_path=str(report_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified model backtest")
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "xgboost", "catboost"],
                        help="Model type to use (default: lgbm)")
    parser.add_argument("--db", default="krx_stock_data.db")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--end", default="20260213")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--rebalance-days", type=int, default=63)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--min-market-cap", type=int, default=500_000_000_000)
    parser.add_argument("--time-decay", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--output", default="backtest_unified_results.csv")
    parser.add_argument("--model-out", default="models/lgbm_unified.pkl")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model-jobs", type=int, default=0, help="LightGBM threads per worker (0=auto)")
    parser.add_argument("--buy-fee", type=float, default=0.5, help="Buy fee percent per trade (default 0.5)")
    parser.add_argument("--sell-fee", type=float, default=0.5, help="Sell fee percent per trade (default 0.5)")
    parser.add_argument("--stress-mode", action="store_true", help="Enable realism stress tests")
    parser.add_argument("--vol-exclude-pct", type=float, default=0.10, help="Exclude top N%% volatility names (default 0.10)")
    parser.add_argument("--sector-neutral-score", action="store_true", default=True,
                        help="Use sector-neutral z-score ranking for picks (default: on)")
    parser.add_argument("--no-sector-neutral", action="store_true",
                        help="Disable sector-neutral scoring")
    parser.add_argument("--buy-rank", type=int, default=10, help="Buy only if rank <= this threshold (default 10)")
    parser.add_argument("--hold-rank", type=int, default=90, help="Keep holding while rank <= this threshold (default 90)")
    parser.add_argument("--embargo-days", type=int, default=21, help="Purged embargo gap in trading days (default 21)")
    parser.add_argument("--cash-out", action="store_true", default=True,
                        help="Go 50%% cash when KOSPI200 below 20d MA (default: on)")
    parser.add_argument("--no-cash-out", action="store_true",
                        help="Disable cash-out logic")
    parser.add_argument("--save-picks", action="store_true", help="Save picked stocks per rebalance date to CSV")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    run(args)


if __name__ == "__main__":
    main()
