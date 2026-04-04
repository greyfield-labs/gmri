"""
GMRI Parameter Sensitivity Grid
=================================
Runs a full 5×3×4 = 60-configuration grid search across:
  - CPI thresholds:    2.0, 2.25, 2.5, 2.75, 3.0 (%)
  - GDP smooth window: 1, 3, 6 (months)
  - Min duration:      1, 2, 3, 6 (months)

Data is fetched once. All 60 classifications run in-memory.

Outputs:
  validate/sensitivity_grid.csv   — one row per configuration
  Printed fragility ranking of the three parameters

Run from project root:  python validate/sensitivity.py
"""

import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
CPI_THRESHOLDS  = [2.00, 2.25, 2.50, 2.75, 3.00]
GDP_WINDOWS     = [1, 3, 6]
MIN_DURATIONS   = [1, 2, 3, 6]

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
START_DATE   = "1970-01-01"

# ---------------------------------------------------------------------------
# Data fetching (identical to replication.py — no imports from src/)
# ---------------------------------------------------------------------------

def get_fred_client() -> Fred:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(f"FRED_API_KEY not found. Expected .env at: {env_path}")
    return Fred(api_key=api_key)


def fetch(fred: Fred, series_id: str, start: str) -> pd.Series:
    s = fred.get_series(series_id, observation_start=start).dropna()
    s.name = series_id
    return s


def to_month_end(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def interp_quarterly(s: pd.Series, method: str = "cubic") -> pd.Series:
    monthly = s.resample("ME").asfreq()
    monthly = monthly.interpolate(method=method)
    monthly = monthly.ffill()
    return monthly


def fetch_master_df(fred: Fred) -> pd.DataFrame:
    """Fetch and assemble the base DataFrame. Called once."""
    print("  Fetching CPIAUCSL...")
    cpi_raw  = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("  Fetching UNRATE...")
    unrate   = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("  Fetching GDPC1...")
    gdpc1    = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), method="cubic")
    print("  Fetching NROU...")
    nrou     = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), method="linear")
    print("  Fetching GDPPOT...")
    gdppot   = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), method="cubic")

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"

    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:].dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])
    return df


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_one(
    df: pd.DataFrame,
    cpi_threshold: float,
    gdp_window: int,
    min_duration: int,
) -> pd.Series:
    """Return the filtered regime Series for one parameter combination."""
    # GDP gap and smoothing
    gdp_gap = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    gdp_gap_smooth = gdp_gap.rolling(window=gdp_window, min_periods=1).mean()

    # Unemployment gap
    unemp_gap = df["UNRATE"] - df["NROU"]

    # AND logic for growth
    growth_above  = (gdp_gap_smooth > 0) & (unemp_gap < 0)
    infl_above    = df["CPI_YOY"] > cpi_threshold

    # Regime assignment
    conditions = [
        growth_above & ~infl_above,
        growth_above &  infl_above,
       ~growth_above &  infl_above,
       ~growth_above & ~infl_above,
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")
    regime = pd.Series(
        pd.Categorical(raw, categories=REGIME_ORDER),
        index=df.index, name="REGIME",
    )

    # Min-duration filter
    if min_duration > 1:
        regime = apply_min_duration_filter(regime, min_duration)

    return regime


def apply_min_duration_filter(regime: pd.Series, min_duration: int) -> pd.Series:
    filtered = regime.astype(str).copy()
    for _ in range(20):
        block_id = (filtered != filtered.shift()).cumsum()
        changed = False
        for _bid, group in filtered.groupby(block_id, sort=True):
            if len(group) < min_duration:
                loc = filtered.index.get_loc(group.index[0])
                if loc > 0:
                    filtered[group.index] = filtered.iloc[loc - 1]
                    changed = True
        if not changed:
            break
    return pd.Series(
        pd.Categorical(filtered, categories=REGIME_ORDER),
        index=regime.index, name="REGIME",
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(regime: pd.Series) -> dict:
    total = len(regime)
    regime_str = regime.astype(str)

    # % time per regime
    pct = {r: (regime == r).sum() / total * 100 for r in REGIME_ORDER}

    # Transition count (month-to-month regime changes)
    n_transitions = int((regime_str != regime_str.shift()).sum()) - 1  # -1 for first row

    # Transition matrix — diagonal persistence per regime
    from_vals = regime.iloc[:-1]
    to_vals   = regime.iloc[1:]
    trans = pd.crosstab(
        pd.Categorical(from_vals, categories=REGIME_ORDER),
        pd.Categorical(to_vals,   categories=REGIME_ORDER),
        normalize="index",
    )
    diag = {r: float(trans.loc[r, r]) if r in trans.index and r in trans.columns else np.nan
            for r in REGIME_ORDER}
    avg_diag = float(np.nanmean(list(diag.values())))

    return {
        "pct_goldilocks":        round(pct["Goldilocks"],        2),
        "pct_overheating":       round(pct["Overheating"],       2),
        "pct_stagflation":       round(pct["Stagflation"],       2),
        "pct_deflationary_bust": round(pct["Deflationary Bust"], 2),
        "n_transitions":         n_transitions,
        "diag_goldilocks":       round(diag["Goldilocks"],        4),
        "diag_overheating":      round(diag["Overheating"],       4),
        "diag_stagflation":      round(diag["Stagflation"],       4),
        "diag_deflationary_bust":round(diag["Deflationary Bust"], 4),
        "avg_diag_persistence":  round(avg_diag,                  4),
    }


# ---------------------------------------------------------------------------
# Fragility ranking
# ---------------------------------------------------------------------------

def compute_fragility_ranking(grid: pd.DataFrame) -> None:
    """
    For each parameter, compute its effect on each target metric using the
    "range of group means" method: vary one parameter across its levels while
    averaging over all other parameter combinations. The effect size is
    max(group_mean) - min(group_mean).

    A larger range = more fragile (more sensitive) to that parameter choice.
    """
    params = {
        "cpi_threshold": CPI_THRESHOLDS,
        "gdp_window":    GDP_WINDOWS,
        "min_duration":  MIN_DURATIONS,
    }
    target_metrics = {
        "Goldilocks frequency (%)":     "pct_goldilocks",
        "Total transition count":        "n_transitions",
        "Avg diagonal persistence":      "avg_diag_persistence",
    }

    print("\n" + "=" * 70)
    print(f"{'PARAMETER SENSITIVITY ANALYSIS':^70}")
    print("=" * 70)

    # ---- Per-target metric: group means table + effect size ----
    effect_sizes: dict[str, dict[str, float]] = {m: {} for m in target_metrics}

    for metric_label, col in target_metrics.items():
        print(f"\n  TARGET: {metric_label}")
        print(f"  {'Parameter':<20}  {'Levels →':>10}", end="")

        for param, levels in params.items():
            group_means = grid.groupby(param)[col].mean()

            # Header row: parameter name and its levels
            print(f"\n  {param:<20}", end="")
            for lvl in levels:
                print(f"  {group_means.loc[lvl]:>7.2f}", end="")

            effect = group_means.max() - group_means.min()
            effect_sizes[metric_label][param] = effect
            print(f"   │ range={effect:.2f}", end="")

        print()

    # ---- Fragility ranking per metric ----
    print("\n" + "-" * 70)
    print(f"  {'FRAGILITY RANKING (most → least impactful)':^68}")
    print("-" * 70)

    # Collect overall ranking across all three metrics
    overall_normalised: dict[str, float] = {p: 0.0 for p in params}

    for metric_label, sizes in effect_sizes.items():
        ranked = sorted(sizes.items(), key=lambda x: x[1], reverse=True)
        max_effect = ranked[0][1] if ranked[0][1] > 0 else 1.0
        print(f"\n  {metric_label}:")
        for rank, (param, effect) in enumerate(ranked, 1):
            bar = "█" * max(1, round(effect / max_effect * 20))
            print(f"    {rank}. {param:<20}  effect={effect:>7.3f}  {bar}")
            overall_normalised[param] += effect / max_effect  # normalise to [0,1]

    # ---- Single overall ranking across all three metrics ----
    print("\n" + "-" * 70)
    print(f"  {'OVERALL FRAGILITY RANKING (sum of normalised effects)':^68}")
    print("-" * 70)
    overall_ranked = sorted(overall_normalised.items(), key=lambda x: x[1], reverse=True)
    for rank, (param, score) in enumerate(overall_ranked, 1):
        bar = "█" * max(1, round(score / overall_ranked[0][1] * 30))
        print(f"    {rank}. {param:<20}  score={score:.3f}  {bar}")

    print()
    print("  Interpretation:")
    print("    score = sum of (effect / max_effect) across 3 target metrics")
    print("    Higher score → index is more sensitive to this parameter.")
    print("=" * 70 + "\n")

    return overall_ranked  # (param, score) list, most → least impactful


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grid() -> pd.DataFrame:
    print("=" * 60)
    print("  GMRI PARAMETER SENSITIVITY GRID")
    print("=" * 60)
    print(f"  Grid: {len(CPI_THRESHOLDS)} CPI thresholds × "
          f"{len(GDP_WINDOWS)} GDP windows × "
          f"{len(MIN_DURATIONS)} min durations "
          f"= {len(CPI_THRESHOLDS)*len(GDP_WINDOWS)*len(MIN_DURATIONS)} configurations")
    print()

    # Fetch data once
    print("Fetching data from FRED (once)...")
    fred = get_fred_client()
    df_base = fetch_master_df(fred)
    print(f"  Data: {df_base.index[0].strftime('%Y-%m')} to "
          f"{df_base.index[-1].strftime('%Y-%m')} ({len(df_base)} months)\n")

    # Run grid
    print("Running 60 configurations...")
    rows = []
    configs = list(itertools.product(CPI_THRESHOLDS, GDP_WINDOWS, MIN_DURATIONS))

    for i, (cpi_t, gdp_w, min_d) in enumerate(configs, 1):
        regime = classify_one(df_base, cpi_threshold=cpi_t,
                              gdp_window=gdp_w, min_duration=min_d)
        metrics = compute_metrics(regime)
        rows.append({
            "cpi_threshold": cpi_t,
            "gdp_window":    gdp_w,
            "min_duration":  min_d,
            **metrics,
        })
        if i % 10 == 0 or i == len(configs):
            print(f"  [{i:>2}/{len(configs)}] done")

    grid = pd.DataFrame(rows)

    # Save CSV
    out_path = Path(__file__).parent / "sensitivity_grid.csv"
    grid.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved: {out_path}  ({len(grid)} rows × {len(grid.columns)} columns)")

    # Print fragility ranking
    compute_fragility_ranking(grid)

    return grid


if __name__ == "__main__":
    run_grid()
