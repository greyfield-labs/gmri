import numpy as np
import pandas as pd

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]

# Short labels for chart annotations
REGIME_SHORT = {
    "Goldilocks": "GL",
    "Overheating": "OV",
    "Stagflation": "SG",
    "Deflationary Bust": "DB",
}


def compute_gdp_gap(df: pd.DataFrame) -> pd.Series:
    gap = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    gap.name = "GDP_GAP"
    return gap


def smooth_gdp_gap(gdp_gap: pd.Series, window: int = 3) -> pd.Series:
    smoothed = gdp_gap.rolling(window=window, min_periods=1).mean()
    smoothed.name = "GDP_GAP_SMOOTH"
    return smoothed


def compute_unemployment_gap(df: pd.DataFrame) -> pd.Series:
    gap = df["UNRATE"] - df["NROU"]
    gap.name = "UNEMP_GAP"
    return gap


def classify_growth(
    gdp_gap_smooth: pd.Series,
    unemp_gap: pd.Series,
    require_both: bool = True,
) -> pd.Series:
    """
    Flag growth as above trend.
    require_both=True (default): GDP gap > 0 AND unemployment gap < 0 must both hold.
    require_both=False (original OR logic): either signal alone is sufficient.
    """
    if require_both:
        above = (gdp_gap_smooth > 0) & (unemp_gap < 0)
    else:
        above = (gdp_gap_smooth > 0) | (unemp_gap < 0)
    above.name = "GROWTH_ABOVE_TREND"
    return above


def compute_signal_conflict(df: pd.DataFrame) -> pd.Series:
    """
    True for months where GDP gap and unemployment gap give opposite growth signals.
    Conflict = one says above trend, the other says below trend.
    """
    gdp_says_above = df["GDP_GAP_SMOOTH"] > 0
    unemp_says_above = df["UNEMP_GAP"] < 0
    conflict = gdp_says_above != unemp_says_above
    conflict.name = "SIGNAL_CONFLICT"
    return conflict


def classify_inflation(df: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
    above = df["CPI_YOY"] > threshold
    above.name = "INFLATION_ABOVE"
    return above


def assign_regime(
    growth_above: pd.Series, inflation_above: pd.Series
) -> pd.Series:
    conditions = [
        growth_above & ~inflation_above,
        growth_above & inflation_above,
        ~growth_above & inflation_above,
        ~growth_above & ~inflation_above,
    ]
    choices = REGIME_ORDER
    regime = np.select(conditions, choices, default="Unknown")
    return pd.Series(
        pd.Categorical(regime, categories=REGIME_ORDER),
        index=growth_above.index,
        name="REGIME",
    )


def apply_min_duration_filter(regime: pd.Series, min_duration: int = 3) -> pd.Series:
    """
    Absorb spells shorter than min_duration months into the preceding regime.
    Iterates until stable. The very first spell is never absorbed (no predecessor).
    """
    filtered = regime.astype(str).copy()
    for _ in range(20):
        block_id = (filtered != filtered.shift()).cumsum()
        changed = False
        for _bid, group in filtered.groupby(block_id, sort=True):
            if len(group) < min_duration:
                loc = filtered.index.get_loc(group.index[0])
                if loc > 0:
                    prev_val = filtered.iloc[loc - 1]
                    filtered[group.index] = prev_val
                    changed = True
        if not changed:
            break
    result = pd.Series(
        pd.Categorical(filtered, categories=REGIME_ORDER),
        index=regime.index,
        name="REGIME_FILTERED",
    )
    return result


def _run_length_encode(series: pd.Series) -> pd.Series:
    """Return a group ID for each contiguous block of equal values."""
    return (series != series.shift()).cumsum()


def compute_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    regime = df["REGIME"]
    total_months = len(regime)

    block_id = _run_length_encode(regime)
    blocks = (
        pd.DataFrame({"regime": regime, "block": block_id})
        .groupby("block")
        .agg(regime=("regime", "first"), length=("regime", "count"))
        .reset_index(drop=True)
    )

    stats_rows = []
    for r in REGIME_ORDER:
        r_blocks = blocks[blocks["regime"] == r]["length"]
        n_spells = len(r_blocks)
        pct_time = (regime == r).sum() / total_months * 100
        avg_duration = r_blocks.mean() if n_spells > 0 else 0.0
        stats_rows.append(
            {
                "regime": r,
                "pct_time": round(pct_time, 1),
                "avg_duration_months": round(avg_duration, 1),
                "n_spells": n_spells,
            }
        )

    return pd.DataFrame(stats_rows).set_index("regime")


def compute_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    from_vals = regime.iloc[:-1].values
    to_vals = regime.iloc[1:].values
    matrix = pd.crosstab(
        pd.Categorical(from_vals, categories=REGIME_ORDER),
        pd.Categorical(to_vals, categories=REGIME_ORDER),
        normalize="index",
    )
    matrix.index.name = "From"
    matrix.columns.name = "To"
    return matrix


def print_signal_conflict_report(df: pd.DataFrame) -> None:
    conflict = df["SIGNAL_CONFLICT"]
    pct = conflict.mean() * 100

    print("\n" + "=" * 58)
    print(f"{'SIGNAL CONFLICT REPORT':^58}")
    print("=" * 58)
    print(f"  GDP gap vs unemployment gap disagree: {pct:.1f}% of months")
    print()

    # Breakdown: how conflict months are distributed across AND-logic regimes
    conflict_df = df[conflict]
    print(f"  Distribution of {len(conflict_df)} conflict months by AND-logic regime:")
    regime_counts = conflict_df["REGIME"].value_counts()
    for r in REGIME_ORDER:
        count = regime_counts.get(r, 0)
        share = count / len(conflict_df) * 100 if len(conflict_df) > 0 else 0
        print(f"    {r:<22} {count:>4d} months ({share:>5.1f}%)")

    # Also show what OR-logic would have called those months
    or_growth = classify_growth(df["GDP_GAP_SMOOTH"], df["UNEMP_GAP"], require_both=False)
    and_growth = df["GROWTH_ABOVE_TREND"]
    upgraded = conflict & or_growth & ~and_growth
    downgraded = conflict & ~or_growth & and_growth
    print(f"\n  OR→AND reclassification impact:")
    print(f"    Months OR called above-trend but AND does not: {upgraded.sum():>4d}")
    print(f"    Months AND calls above-trend but OR did not:   {downgraded.sum():>4d}")
    print("=" * 58 + "\n")


def classify(
    df: pd.DataFrame,
    cpi_threshold: float = 2.5,
    gdp_gap_window: int = 3,
    require_both_growth: bool = True,
    min_duration: int | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["GDP_GAP"] = compute_gdp_gap(df)
    df["GDP_GAP_SMOOTH"] = smooth_gdp_gap(df["GDP_GAP"], window=gdp_gap_window)
    df["UNEMP_GAP"] = compute_unemployment_gap(df)
    df["GROWTH_ABOVE_TREND"] = classify_growth(
        df["GDP_GAP_SMOOTH"], df["UNEMP_GAP"], require_both=require_both_growth
    )
    df["INFLATION_ABOVE"] = classify_inflation(df, threshold=cpi_threshold)
    df["REGIME"] = assign_regime(df["GROWTH_ABOVE_TREND"], df["INFLATION_ABOVE"])
    df["SIGNAL_CONFLICT"] = compute_signal_conflict(df)
    if min_duration is not None:
        df["REGIME_FILTERED"] = apply_min_duration_filter(df["REGIME"], min_duration)
    return df
