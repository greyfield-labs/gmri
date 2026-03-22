"""
Robustness checks for the Macro Regime Index:
  - Threshold sensitivity: re-classify at multiple CPI thresholds
  - Era breakdown: regime distribution across four structural macro eras
  - Filtered vs unfiltered regime comparison
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from classifier import (
    REGIME_ORDER,
    REGIME_SHORT,
    apply_min_duration_filter,
    classify,
    compute_regime_stats,
)
from charts import REGIME_COLORS, FIG_DPI

# ---------------------------------------------------------------------------
# Era definitions — single source of truth; imported by charts.py too
# ---------------------------------------------------------------------------
ERAS = [
    ("Pre-Volcker",          "1971-01-01", "1979-12-31"),
    ("Volcker / Great Mod.", "1980-01-01", "2000-12-31"),
    ("Post-GFC",             "2001-01-01", "2019-12-31"),
    ("Recent",               "2020-01-01", None),
]

ERA_BOUNDARY_DATES = ["1980-01-01", "2001-01-01", "2020-01-01"]


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity_analysis(
    df_base: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> dict[float, pd.DataFrame]:
    """
    Re-classify df_base at each threshold (AND logic).
    Returns {threshold: stats_df}.
    """
    if thresholds is None:
        thresholds = [2.0, 2.5, 3.0]
    results = {}
    for t in thresholds:
        classified = classify(df_base, cpi_threshold=t, require_both_growth=True)
        results[t] = compute_regime_stats(classified)
    return results


def plot_threshold_sensitivity(
    results: dict[float, pd.DataFrame],
    save_path: str | None = None,
) -> plt.Figure:
    thresholds = sorted(results.keys())
    n_thresholds = len(thresholds)
    n_regimes = len(REGIME_ORDER)
    bar_width = 0.18
    x = np.arange(n_thresholds)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, regime in enumerate(REGIME_ORDER):
        pct_values = [results[t].loc[regime, "pct_time"] for t in thresholds]
        offset = (i - n_regimes / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            pct_values,
            width=bar_width,
            color=REGIME_COLORS[regime],
            alpha=0.85,
            label=regime,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, pct_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}% threshold" for t in thresholds], fontsize=10)
    ax.set_ylabel("% of Months in Regime", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_title(
        "Regime Distribution by CPI Inflation Threshold (AND Growth Logic)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


def print_threshold_sensitivity(results: dict[float, pd.DataFrame]) -> None:
    thresholds = sorted(results.keys())
    print("\n" + "=" * 72)
    print(f"{'THRESHOLD SENSITIVITY ANALYSIS':^72}")
    print("=" * 72)
    header = f"{'Regime':<22}" + "".join(f"  {t}% thresh" for t in thresholds)
    print(header)
    print("-" * 72)
    for r in REGIME_ORDER:
        row = f"{r:<22}"
        for t in thresholds:
            pct = results[t].loc[r, "pct_time"]
            row += f"  {pct:>8.1f}%"
        print(row)
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Era breakdown
# ---------------------------------------------------------------------------

def era_breakdown(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Slice df by era and return per-era regime stats."""
    results = {}
    for name, start, end in ERAS:
        era_df = df.loc[start:end].copy() if end else df.loc[start:].copy()
        if len(era_df) == 0:
            continue
        results[name] = compute_regime_stats(era_df)
    return results


def print_era_breakdown(era_results: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 58)
    print(f"{'ERA-BASED REGIME BREAKDOWN':^58}")
    print("=" * 58)
    for era_name, stats in era_results.items():
        n_months = int(stats["pct_time"].sum() / 100 * sum(
            int(s * m / 100) for s, m in zip(
                stats["pct_time"],
                [stats.loc[r, "avg_duration_months"] * stats.loc[r, "n_spells"]
                 for r in REGIME_ORDER]
            )
        )) if False else None  # skip month count for brevity
        print(f"\n  {era_name}")
        print(f"  {'Regime':<22} {'% Time':>7} {'Avg Dur':>9} {'# Spells':>9}")
        print("  " + "-" * 52)
        for regime, row in stats.iterrows():
            print(
                f"  {regime:<22} {row['pct_time']:>6.1f}% "
                f"{row['avg_duration_months']:>7.1f}mo "
                f"{int(row['n_spells']):>8d}"
            )
    print("\n" + "=" * 58 + "\n")


def plot_era_breakdown(
    era_results: dict[str, pd.DataFrame],
    save_path: str | None = None,
) -> plt.Figure:
    era_names = list(era_results.keys())
    n_eras = len(era_names)
    n_regimes = len(REGIME_ORDER)
    bar_width = 0.18
    x = np.arange(n_eras)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, regime in enumerate(REGIME_ORDER):
        pct_values = [era_results[e].loc[regime, "pct_time"] for e in era_names]
        offset = (i - n_regimes / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            pct_values,
            width=bar_width,
            color=REGIME_COLORS[regime],
            alpha=0.85,
            label=regime,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, pct_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(era_names, fontsize=10)
    ax.set_ylabel("% of Months in Regime", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.set_title(
        "Regime Breakdown by Structural Era",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Filtered vs unfiltered comparison
# ---------------------------------------------------------------------------

def print_filtered_vs_unfiltered(df: pd.DataFrame, min_duration: int = 3) -> None:
    unfiltered_stats = compute_regime_stats(df)

    filtered_regime = apply_min_duration_filter(df["REGIME"], min_duration)
    filtered_df = df.copy()
    filtered_df["REGIME"] = filtered_regime
    filtered_stats = compute_regime_stats(filtered_df)

    print("\n" + "=" * 72)
    print(f"{'FILTERED vs UNFILTERED REGIME STATS (min_duration=' + str(min_duration) + 'mo)':^72}")
    print("=" * 72)
    header = (
        f"{'Regime':<22}"
        f"  {'Unfilt %':>8} {'Filt %':>7}"
        f"  {'Unfilt Dur':>10} {'Filt Dur':>9}"
        f"  {'Unfilt N':>8} {'Filt N':>7}"
    )
    print(header)
    print("-" * 72)
    for r in REGIME_ORDER:
        u = unfiltered_stats.loc[r]
        f = filtered_stats.loc[r]
        print(
            f"{r:<22}"
            f"  {u['pct_time']:>7.1f}% {f['pct_time']:>6.1f}%"
            f"  {u['avg_duration_months']:>9.1f}mo {f['avg_duration_months']:>8.1f}mo"
            f"  {int(u['n_spells']):>8d} {int(f['n_spells']):>7d}"
        )

    total_unfilt = sum(unfiltered_stats["n_spells"])
    total_filt = sum(filtered_stats["n_spells"])
    print("-" * 72)
    print(
        f"{'TOTAL spells':<22}"
        f"  {'':>8} {'':>7}"
        f"  {'':>10} {'':>9}"
        f"  {int(total_unfilt):>8d} {int(total_filt):>7d}"
    )
    eliminated = int(total_unfilt) - int(total_filt)
    print(f"\n  Spurious spells eliminated by {min_duration}-month filter: {eliminated}")
    print("=" * 72 + "\n")
