import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import seaborn as sns
import pandas as pd

REGIME_COLORS = {
    "Goldilocks": "#4CAF50",
    "Overheating": "#FF9800",
    "Stagflation": "#F44336",
    "Deflationary Bust": "#2196F3",
}
RECESSION_COLOR = "#BDBDBD"
RECESSION_ALPHA = 0.4
FIG_DPI = 150
CPI_THRESHOLD = 2.5


def _run_length_blocks(series: pd.Series):
    """Yield (value, start_date, end_date) for each contiguous block."""
    block_id = (series != series.shift()).cumsum()
    for _, grp in series.groupby(block_id):
        yield grp.iloc[0], grp.index[0], grp.index[-1]


def _add_recession_shading(ax: plt.Axes, recession: pd.Series) -> None:
    rec = recession.dropna()
    for val, start, end in _run_length_blocks(rec):
        if val == 1:
            ax.axvspan(
                start, end,
                color=RECESSION_COLOR,
                alpha=RECESSION_ALPHA,
                zorder=2,
                linewidth=0,
            )


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def plot_regime_timeline(
    df: pd.DataFrame, save_path: str | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw regime colored bands (zorder=1, under recession shading)
    for regime, start, end in _run_length_blocks(df["REGIME"]):
        ax.axvspan(
            start, end,
            color=REGIME_COLORS[regime],
            alpha=0.45,
            zorder=1,
            linewidth=0,
        )

    # Recession shading on top of regime bands
    _add_recession_shading(ax, df["RECESSION"])

    # Secondary axis: CPI YoY
    ax2 = ax.twinx()
    ax2.plot(
        df.index, df["CPI_YOY"],
        color="black", linewidth=1.1, alpha=0.75, label="CPI YoY %", zorder=3,
    )
    ax2.axhline(
        CPI_THRESHOLD, color="black", linewidth=0.9, linestyle="--",
        alpha=0.6, zorder=3, label=f"{CPI_THRESHOLD}% threshold",
    )
    ax2.set_ylabel("CPI YoY %", fontsize=10)
    ax2.tick_params(axis="y", labelsize=9)

    # Primary axis cosmetics
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("")
    _format_date_axis(ax)
    ax.tick_params(axis="x", labelsize=9)

    # Legend
    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.45, label=r)
        for r in REGIME_COLORS
    ]
    recession_patch = mpatches.Patch(
        color=RECESSION_COLOR, alpha=RECESSION_ALPHA, label="NBER Recession"
    )
    cpi_line = plt.Line2D([0], [0], color="black", linewidth=1.1, alpha=0.75, label="CPI YoY %")
    threshold_line = plt.Line2D(
        [0], [0], color="black", linewidth=0.9, linestyle="--",
        alpha=0.6, label=f"{CPI_THRESHOLD}% inflation threshold",
    )
    ax.legend(
        handles=regime_patches + [recession_patch, cpi_line, threshold_line],
        loc="upper left",
        fontsize=8,
        framealpha=0.85,
        ncol=2,
    )

    ax.set_title("Macro Regime Index: 1970–Present", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_transition_matrix(
    transition_matrix: pd.DataFrame, save_path: str | None = None
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 11},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        "Month-Over-Month Regime Transition Probabilities",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("To Regime", fontsize=10, labelpad=8)
    ax.set_ylabel("From Regime", fontsize=10, labelpad=8)
    plt.setp(ax.xaxis.get_ticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax.yaxis.get_ticklabels(), rotation=0, fontsize=9)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_signal_chart(
    df: pd.DataFrame, save_path: str | None = None
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    # --- Top subplot: GDP Gap ---
    ax1.plot(
        df.index, df["GDP_GAP"],
        color="#90A4AE", linewidth=0.8, linestyle="--", alpha=0.7, label="GDP Gap (raw)",
    )
    ax1.plot(
        df.index, df["GDP_GAP_SMOOTH"],
        color="#37474F", linewidth=1.5, label="GDP Gap (3m smoothed)",
    )
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax1.fill_between(
        df.index, df["GDP_GAP_SMOOTH"], 0,
        where=df["GDP_GAP_SMOOTH"] > 0,
        color=REGIME_COLORS["Goldilocks"], alpha=0.2, label="Above trend",
    )
    ax1.fill_between(
        df.index, df["GDP_GAP_SMOOTH"], 0,
        where=df["GDP_GAP_SMOOTH"] <= 0,
        color=REGIME_COLORS["Deflationary Bust"], alpha=0.2, label="Below trend",
    )
    _add_recession_shading(ax1, df["RECESSION"])
    ax1.set_title("GDP Output Gap (% of Potential GDP)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Output Gap (%)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # --- Bottom subplot: CPI YoY ---
    ax2.plot(
        df.index, df["CPI_YOY"],
        color="#B71C1C", linewidth=1.3, label="CPI YoY %",
    )
    ax2.axhline(
        CPI_THRESHOLD, color="#F44336", linewidth=1.1, linestyle="--",
        label=f"{CPI_THRESHOLD}% threshold",
    )
    ax2.fill_between(
        df.index, df["CPI_YOY"], CPI_THRESHOLD,
        where=df["CPI_YOY"] > CPI_THRESHOLD,
        color="#F44336", alpha=0.15, label="Above threshold",
    )
    _add_recession_shading(ax2, df["RECESSION"])
    ax2.set_title("CPI Inflation YoY %", fontsize=11, fontweight="bold")
    ax2.set_ylabel("CPI YoY %", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # Shared x-axis formatting
    _format_date_axis(ax2)
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_xlim(df.index[0], df.index[-1])

    # Add recession legend entry to both subplots
    recession_patch = mpatches.Patch(
        color=RECESSION_COLOR, alpha=RECESSION_ALPHA, label="NBER Recession"
    )
    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + [recession_patch],
            fontsize=8, loc="upper right", framealpha=0.85,
        )

    fig.suptitle(
        "Macro Regime Signals: GDP Gap & CPI Inflation",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def plot_regime_timeline_v2(
    df: pd.DataFrame,
    era_results: dict | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Enhanced timeline chart with era boundary lines, era name labels,
    and per-era regime % annotations. Requires robustness.ERAS and REGIME_SHORT.
    """
    from robustness import ERAS, ERA_BOUNDARY_DATES
    from classifier import REGIME_SHORT, REGIME_ORDER

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Regime colored bands ---
    for regime, start, end in _run_length_blocks(df["REGIME"]):
        ax.axvspan(
            start, end,
            color=REGIME_COLORS[regime],
            alpha=0.45,
            zorder=1,
            linewidth=0,
        )

    # --- NBER recession shading ---
    _add_recession_shading(ax, df["RECESSION"])

    # --- Era boundary vertical lines ---
    for boundary in ERA_BOUNDARY_DATES:
        bdate = pd.Timestamp(boundary)
        if df.index[0] <= bdate <= df.index[-1]:
            ax.axvline(
                bdate, color="#333333", linewidth=1.2,
                linestyle="--", alpha=0.7, zorder=4,
            )

    # --- Era name labels and regime % annotations ---
    if era_results:
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        era_dates = [(pd.Timestamp(s), pd.Timestamp(e) if e else df.index[-1])
                     for _, s, e in ERAS]

        for (era_name, start_str, end_str), stats in zip(ERAS, era_results.values()):
            era_start = pd.Timestamp(start_str)
            era_end = pd.Timestamp(end_str) if end_str else df.index[-1]
            # Clip to actual data range
            era_start = max(era_start, df.index[0])
            era_end = min(era_end, df.index[-1])
            x_mid = era_start + (era_end - era_start) / 2

            # Era name
            ax.text(
                x_mid, 0.99, era_name,
                transform=blended,
                ha="center", va="top",
                fontsize=8, fontweight="bold",
                color="#333333", zorder=5,
            )

            # Per-era regime percentages — two lines of two regimes each
            line1 = "  ".join(
                f"{REGIME_SHORT[r]}: {stats.loc[r, 'pct_time']:.0f}%"
                for r in REGIME_ORDER[:2]
            )
            line2 = "  ".join(
                f"{REGIME_SHORT[r]}: {stats.loc[r, 'pct_time']:.0f}%"
                for r in REGIME_ORDER[2:]
            )
            ax.text(
                x_mid, 0.92, f"{line1}\n{line2}",
                transform=blended,
                ha="center", va="top",
                fontsize=6.5, color="#444444",
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, linewidth=0),
            )

    # --- Secondary axis: CPI YoY ---
    ax2 = ax.twinx()
    ax2.plot(
        df.index, df["CPI_YOY"],
        color="black", linewidth=1.1, alpha=0.75, label="CPI YoY %", zorder=3,
    )
    ax2.axhline(
        CPI_THRESHOLD, color="black", linewidth=0.9, linestyle="--",
        alpha=0.6, zorder=3, label=f"{CPI_THRESHOLD}% threshold",
    )
    ax2.set_ylabel("CPI YoY %", fontsize=10)
    ax2.tick_params(axis="y", labelsize=9)

    # --- Primary axis cosmetics ---
    ax.set_xlim(df.index[0], df.index[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("")
    _format_date_axis(ax)
    ax.tick_params(axis="x", labelsize=9)

    # --- Legend ---
    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.45, label=r)
        for r in REGIME_COLORS
    ]
    recession_patch = mpatches.Patch(
        color=RECESSION_COLOR, alpha=RECESSION_ALPHA, label="NBER Recession"
    )
    era_line = plt.Line2D(
        [0], [0], color="#333333", linewidth=1.2, linestyle="--",
        alpha=0.7, label="Era boundary",
    )
    cpi_line = plt.Line2D([0], [0], color="black", linewidth=1.1, alpha=0.75, label="CPI YoY %")
    threshold_line = plt.Line2D(
        [0], [0], color="black", linewidth=0.9, linestyle="--",
        alpha=0.6, label=f"{CPI_THRESHOLD}% inflation threshold",
    )
    ax.legend(
        handles=regime_patches + [recession_patch, era_line, cpi_line, threshold_line],
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        ncol=2,
    )

    ax.set_title(
        "Macro Regime Index: 1970–Present (with Era Breakdown)",
        fontsize=14, fontweight="bold", pad=12,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


def print_stats_table(stats: pd.DataFrame) -> None:
    print("\n" + "=" * 58)
    print(f"{'MACRO REGIME SUMMARY STATISTICS':^58}")
    print("=" * 58)
    header = f"{'Regime':<22} {'% Time':>7} {'Avg Dur':>9} {'# Spells':>9}"
    print(header)
    print("-" * 58)
    for regime, row in stats.iterrows():
        print(
            f"{regime:<22} {row['pct_time']:>6.1f}% "
            f"{row['avg_duration_months']:>7.1f}mo "
            f"{int(row['n_spells']):>8d}"
        )
    print("=" * 58 + "\n")
