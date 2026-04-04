"""
GMRI Markov Stability Test
============================
Tests whether regime transition probabilities are stable across time using
two complementary approaches:

  1. Rolling 10-year windows (1-year step): plots diagonal persistence
     over time to visualise structural breaks.

  2. Era chi-squared test: compares each era's observed transition counts
     against full-sample expected counts. Rejects H0 of stable transitions
     at p < 0.05.

Eras:
  Pre-Volcker      1971-01 – 1979-12
  Great Moderation 1980-01 – 2000-12
  Post-GFC         2001-01 – 2019-12
  Recent           2020-01 – 2025-09

Run from project root:  python validate/markov_stability.py
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Parameters — must match production main.py exactly
# ---------------------------------------------------------------------------
START_DATE    = "1970-01-01"
CPI_THRESHOLD = 2.5
GDP_WINDOW    = 3
MIN_DURATION  = 3

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
SHORT = {"Goldilocks": "GL", "Overheating": "OV",
         "Stagflation": "SG", "Deflationary Bust": "DB"}

REGIME_COLORS = {
    "Goldilocks":        "#4CAF50",
    "Overheating":       "#FF9800",
    "Stagflation":       "#F44336",
    "Deflationary Bust": "#2196F3",
}

ERAS = [
    ("Pre-Volcker",      "1971-01", "1979-12"),
    ("Great Moderation", "1980-01", "2000-12"),
    ("Post-GFC",         "2001-01", "2019-12"),
    ("Recent",           "2020-01", None),
]

ROLLING_WINDOW_YEARS = 10
ROLLING_STEP_YEARS   = 1
FIG_DPI = 150

# ---------------------------------------------------------------------------
# Data fetching (no src/ imports)
# ---------------------------------------------------------------------------

def get_fred_client() -> Fred:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(f"FRED_API_KEY not found. Expected .env at: {env_path}")
    return Fred(api_key=key)


def fetch(fred, series_id, start):
    s = fred.get_series(series_id, observation_start=start).dropna()
    s.name = series_id
    return s


def to_month_end(s):
    s = s.copy()
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s


def interp_quarterly(s, method="cubic"):
    monthly = s.resample("ME").asfreq()
    monthly = monthly.interpolate(method=method)
    return monthly.ffill()


def build_regime_series(fred) -> pd.Series:
    """Fetch, classify, and return the filtered monthly regime series."""
    print("  CPIAUCSL...", end=" ", flush=True)
    cpi_raw = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("UNRATE...",    end=" ", flush=True)
    unrate  = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("GDPC1...",     end=" ", flush=True)
    gdpc1   = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), "cubic")
    print("NROU...",      end=" ", flush=True)
    nrou    = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), "linear")
    print("GDPPOT...")
    gdppot  = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), "cubic")

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:].dropna()

    # Classification
    gdp_gap = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    gdp_sm  = gdp_gap.rolling(window=GDP_WINDOW, min_periods=1).mean()
    u_gap   = df["UNRATE"] - df["NROU"]

    growth_above = (gdp_sm > 0) & (u_gap < 0)
    infl_above   = df["CPI_YOY"] > CPI_THRESHOLD

    conditions = [
        growth_above & ~infl_above,
        growth_above &  infl_above,
       ~growth_above &  infl_above,
       ~growth_above & ~infl_above,
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")

    # Min-duration filter
    filtered = pd.Series(raw, index=df.index, dtype=str)
    for _ in range(20):
        block_id = (filtered != filtered.shift()).cumsum()
        changed = False
        for _bid, group in filtered.groupby(block_id, sort=True):
            if len(group) < MIN_DURATION:
                loc = filtered.index.get_loc(group.index[0])
                if loc > 0:
                    filtered[group.index] = filtered.iloc[loc - 1]
                    changed = True
        if not changed:
            break

    return pd.Series(
        pd.Categorical(filtered, categories=REGIME_ORDER),
        index=df.index, name="REGIME",
    )


# ---------------------------------------------------------------------------
# Transition matrix helpers
# ---------------------------------------------------------------------------

def transition_counts(regime: pd.Series) -> pd.DataFrame:
    """Raw transition count matrix (all pairs, including self)."""
    from_v = regime.astype(str).iloc[:-1]
    to_v   = regime.astype(str).iloc[1:]
    # Build manually so index/columns are always the full REGIME_ORDER strings
    mat = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER, dtype=float)
    for f, t in zip(from_v, to_v):
        if f in mat.index and t in mat.columns:
            mat.loc[f, t] += 1
    mat.index.name   = "From"
    mat.columns.name = "To"
    return mat


def transition_probs(counts: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise count matrix to probabilities. Rows summing to 0 → NaN."""
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(row_sums, axis=0)


def diagonal_values(probs: pd.DataFrame) -> dict[str, float]:
    """Extract diagonal persistence probabilities."""
    return {r: float(probs.loc[r, r]) if r in probs.index else np.nan
            for r in REGIME_ORDER}


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

def compute_rolling_persistence(regime: pd.Series) -> pd.DataFrame:
    """
    For each rolling 10-year window stepped by 1 year, compute the diagonal
    persistence for each regime. Returns a DataFrame indexed by window
    mid-point date.
    """
    records = []
    start_yr = regime.index[0].year
    end_yr   = regime.index[-1].year

    for yr in range(start_yr, end_yr - ROLLING_WINDOW_YEARS + 2, ROLLING_STEP_YEARS):
        w_start = pd.Timestamp(f"{yr}-01-01")   + pd.offsets.MonthEnd(0)
        w_end   = pd.Timestamp(f"{yr + ROLLING_WINDOW_YEARS - 1}-12-31") + pd.offsets.MonthEnd(0)

        window = regime.loc[w_start:w_end]
        if len(window) < 24:   # need at least 2 years of data
            continue

        counts = transition_counts(window)
        probs  = transition_probs(counts)
        diag   = diagonal_values(probs)

        mid_date = w_start + (w_end - w_start) / 2
        records.append({"window_mid": mid_date, **diag})

    return pd.DataFrame(records).set_index("window_mid")


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------

def chi2_era_test(
    regime: pd.Series,
    full_counts: pd.DataFrame,
) -> list[dict]:
    """
    For each era, compute observed transition counts and expected counts
    derived from the full-sample row-conditional probabilities scaled to
    the era's row totals.

    Test statistic: sum over (from, to) pairs of (O - E)^2 / E,
    excluding cells where E < 1 (sparse cells bias chi-squared).

    Degrees of freedom = (n_regimes - 1)^2 — one free parameter per row
    (row probabilities sum to 1), applied to each non-zero row.
    We use df = number of included cells - number of rows with data,
    which is the standard chi-squared contingency df for a transition matrix.
    """
    full_probs = transition_probs(full_counts)
    results = []

    for era_name, era_start, era_end in ERAS:
        s = pd.Timestamp(era_start) + pd.offsets.MonthEnd(0)
        e = (pd.Timestamp(era_end)  + pd.offsets.MonthEnd(0)) if era_end else regime.index[-1]
        era_regime = regime.loc[s:e]

        if len(era_regime) < 12:
            results.append({"era": era_name, "error": "insufficient data"})
            continue

        obs = transition_counts(era_regime)

        # Expected counts: era row total × full-sample transition probability
        era_row_totals = obs.sum(axis=1)
        exp = full_probs.mul(era_row_totals, axis=0)

        # Accumulate chi-squared statistic over valid (O, E) pairs
        chi2_stat = 0.0
        n_cells   = 0
        n_rows    = 0
        rows_seen = set()

        for from_r in REGIME_ORDER:
            row_has_data = False
            for to_r in REGIME_ORDER:
                o = float(obs.loc[from_r, to_r])
                e_val = float(exp.loc[from_r, to_r])
                if e_val < 1.0:
                    continue   # skip sparse cells
                chi2_stat += (o - e_val) ** 2 / e_val
                n_cells   += 1
                row_has_data = True
            if row_has_data:
                rows_seen.add(from_r)

        n_rows = len(rows_seen)
        df = max(n_cells - n_rows, 1)   # df = cells - constraints
        p_value = 1.0 - chi2.cdf(chi2_stat, df=df)

        # Per-cell breakdown (largest contributors)
        cell_contribs = []
        for from_r in REGIME_ORDER:
            for to_r in REGIME_ORDER:
                o     = float(obs.loc[from_r, to_r])
                e_val = float(exp.loc[from_r, to_r])
                if e_val < 1.0:
                    continue
                contrib = (o - e_val) ** 2 / e_val
                cell_contribs.append({
                    "from": from_r, "to": to_r,
                    "observed": o, "expected": round(e_val, 1),
                    "contrib": round(contrib, 3),
                    "direction": "over" if o > e_val else "under",
                })
        cell_contribs.sort(key=lambda x: x["contrib"], reverse=True)

        results.append({
            "era":         era_name,
            "n_months":    len(era_regime),
            "chi2_stat":   round(chi2_stat, 3),
            "df":          df,
            "p_value":     p_value,
            "reject_h0":   p_value < 0.05,
            "obs":         obs,
            "exp":         exp,
            "top_cells":   cell_contribs[:5],
        })

    return results


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_rolling_persistence(rolling_df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"Rolling {ROLLING_WINDOW_YEARS}-Year Diagonal Persistence (1-Year Step)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    era_boundaries = [
        pd.Timestamp("1980-01-01"),
        pd.Timestamp("2001-01-01"),
        pd.Timestamp("2020-01-01"),
    ]
    era_labels = ["Pre-Volcker | Great Mod.", "Great Mod. | Post-GFC", "Post-GFC | Recent"]

    for ax, regime in zip(axes.flat, REGIME_ORDER):
        col = regime
        vals = rolling_df[col].dropna()

        ax.plot(vals.index, vals.values,
                color=REGIME_COLORS[regime], linewidth=1.8, label=regime)
        ax.fill_between(vals.index, vals.values, alpha=0.15,
                        color=REGIME_COLORS[regime])

        # Smoothed trend
        if len(vals) >= 5:
            trend = vals.rolling(5, center=True, min_periods=3).mean()
            ax.plot(trend.index, trend.values,
                    color=REGIME_COLORS[regime], linewidth=2.8,
                    linestyle="--", alpha=0.7, label="5-window trend")

        # Era boundaries
        for boundary, label in zip(era_boundaries, era_labels):
            if vals.index[0] <= boundary <= vals.index[-1]:
                ax.axvline(boundary, color="#555555", linewidth=0.9,
                           linestyle=":", alpha=0.7)

        # Reference lines
        ax.axhline(0.90, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.axhline(0.95, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)

        ax.set_title(regime, fontsize=10, fontweight="bold",
                     color=REGIME_COLORS[regime])
        ax.set_ylim(0.5, 1.02)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

        # Annotation: mean ± std
        mean_v = vals.mean()
        std_v  = vals.std()
        ax.text(0.03, 0.06, f"μ={mean_v:.3f}  σ={std_v:.3f}",
                transform=ax.transAxes, fontsize=8,
                color=REGIME_COLORS[regime], va="bottom")

        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    regime: pd.Series,
    rolling_df: pd.DataFrame,
    full_counts: pd.DataFrame,
    era_results: list[dict],
) -> list[str]:
    lines: list[str] = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    sep  = "=" * 70
    dash = "-" * 70

    log(sep)
    log("  GMRI MARKOV STABILITY TEST")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Parameters: CPI={CPI_THRESHOLD}%  GDP window={GDP_WINDOW}mo  "
        f"Min duration={MIN_DURATION}mo  Growth=AND")
    log(sep)
    log()

    # ------------------------------------------------------------------
    # Full-sample transition matrix
    # ------------------------------------------------------------------
    log("FULL-SAMPLE TRANSITION MATRIX (probabilities)")
    log(dash)
    full_probs = transition_probs(full_counts)
    hdr = f"  {'From → To':<22}" + "".join(f"  {SHORT[c]:>7}" for c in REGIME_ORDER)
    log(hdr)
    log("  " + "-" * 54)
    for r in REGIME_ORDER:
        row = f"  {r:<22}"
        for c in REGIME_ORDER:
            val = full_probs.loc[r, c]
            marker = " *" if r == c else "  "
            row += f"  {val:>5.3f}{marker}"
        log(row)
    log("  (* diagonal persistence)")
    log()

    log("FULL-SAMPLE DIAGONAL PERSISTENCE")
    log(dash)
    full_diag = diagonal_values(full_probs)
    for r in REGIME_ORDER:
        bar_len = max(1, round(full_diag[r] * 30))
        bar = "█" * bar_len
        log(f"  {r:<22}  {full_diag[r]:.4f}  {bar}")
    log()

    # ------------------------------------------------------------------
    # Rolling window summary
    # ------------------------------------------------------------------
    log("ROLLING PERSISTENCE SUMMARY")
    log(dash)
    log(f"  Window: {ROLLING_WINDOW_YEARS} years  |  Step: {ROLLING_STEP_YEARS} year  |  "
        f"Windows computed: {len(rolling_df)}")
    log()
    hdr2 = f"  {'Regime':<22}  {'Mean':>6}  {'Std':>6}  {'Min':>6}  {'Max':>6}  {'Range':>6}"
    log(hdr2)
    log("  " + "-" * 60)
    for r in REGIME_ORDER:
        col  = rolling_df[r].dropna()
        mean = col.mean()
        std  = col.std()
        mn   = col.min()
        mx   = col.max()
        rng  = mx - mn
        log(f"  {r:<22}  {mean:>6.4f}  {std:>6.4f}  {mn:>6.4f}  {mx:>6.4f}  {rng:>6.4f}")
    log()

    # Identify the most volatile regime
    ranges = {r: (rolling_df[r].max() - rolling_df[r].min()) for r in REGIME_ORDER}
    most_volatile = max(ranges, key=ranges.get)
    log(f"  Most volatile diagonal: {most_volatile} "
        f"(range={ranges[most_volatile]:.4f})")
    log()

    # ------------------------------------------------------------------
    # Era chi-squared results
    # ------------------------------------------------------------------
    log("ERA-LEVEL TRANSITION MATRICES AND CHI-SQUARED TESTS")
    log(dash)
    log("  H0: era transition probabilities equal full-sample probabilities")
    log("  Rejection threshold: p < 0.05")
    log()

    any_rejected = False
    for res in era_results:
        if "error" in res:
            log(f"  {res['era']}: {res['error']}")
            continue

        verdict   = "REJECT H0 ✗" if res["reject_h0"] else "FAIL TO REJECT ✓"
        p_str     = f"{res['p_value']:.4f}" if res["p_value"] >= 0.0001 else "<0.0001"
        flag      = " ← UNSTABLE" if res["reject_h0"] else ""
        if res["reject_h0"]:
            any_rejected = True

        log(f"  {res['era']}  ({res['n_months']} months)")
        log(f"    χ²={res['chi2_stat']:.3f}  df={res['df']}  "
            f"p={p_str}  →  {verdict}{flag}")
        log()

        # Observed vs expected probability table for this era
        obs_probs = transition_probs(res["obs"])
        full_p    = transition_probs(full_counts)

        log(f"    Transition probabilities — era vs full-sample:")
        hdr3 = f"    {'From → To':<22}" + "".join(
            f"  {SHORT[c]:>12}" for c in REGIME_ORDER
        )
        log(hdr3)
        log("    " + "-" * 72)
        for r in REGIME_ORDER:
            row = f"    {r:<22}"
            for c in REGIME_ORDER:
                era_p  = obs_probs.loc[r, c]
                full_p_val = full_p.loc[r, c]
                diff   = era_p - full_p_val
                sign   = "+" if diff >= 0 else ""
                marker = "*" if abs(diff) >= 0.10 else " "
                row += f"  {era_p:.3f}({sign}{diff:+.3f}){marker}"
            log(row)
        log("    (* diff ≥ 0.10 from full-sample)")
        log()

        # Top contributing cells
        if res["top_cells"]:
            log(f"    Top χ² contributors:")
            for cell in res["top_cells"][:3]:
                log(f"      {SHORT[cell['from']]}→{SHORT[cell['to']]}  "
                    f"O={cell['observed']:.0f}  E={cell['expected']:.1f}  "
                    f"contrib={cell['contrib']:.3f}  ({cell['direction']}represented)")
        log()

    # ------------------------------------------------------------------
    # Overall stability verdict
    # ------------------------------------------------------------------
    log(sep)
    log("  OVERALL STABILITY VERDICT")
    log(sep)
    log()

    rejected_eras = [r["era"] for r in era_results
                     if "error" not in r and r["reject_h0"]]

    if not any_rejected:
        log("  STABLE: No era rejects H0 at p < 0.05.")
        log("  Transition probabilities are statistically consistent across eras.")
        log("  The Markov chain assumption is supported by the data.")
    else:
        log(f"  UNSTABLE: {len(rejected_eras)} era(s) reject H0 at p < 0.05:")
        for era in rejected_eras:
            log(f"    - {era}")
        log()
        log("  This means transition probabilities differ significantly across")
        log("  structural eras. Regime persistence and transition paths changed")
        log("  over time — a single stationary Markov chain is a simplification.")
        log()
        log("  Implication for white paper:")
        log("  Consider era-conditional transition matrices rather than a single")
        log("  full-sample matrix. The rolling persistence chart (rolling_persistence.png)")
        log("  shows where the structural breaks occurred.")

    log()
    log("  Rolling persistence chart: validate/rolling_persistence.png")
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching and classifying data...")
    fred   = get_fred_client()
    regime = build_regime_series(fred)
    print(f"  {len(regime)} months: "
          f"{regime.index[0].strftime('%Y-%m')} – {regime.index[-1].strftime('%Y-%m')}\n")

    print("Computing full-sample transition matrix...")
    full_counts = transition_counts(regime)

    print(f"Computing rolling {ROLLING_WINDOW_YEARS}-year persistence "
          f"(1-year step)...")
    rolling_df = compute_rolling_persistence(regime)
    print(f"  {len(rolling_df)} windows computed\n")

    print("Running era chi-squared tests...")
    era_results = chi2_era_test(regime, full_counts)

    print("Building report...")
    lines = build_report(regime, rolling_df, full_counts, era_results)

    out_path = Path(__file__).parent / "markov_stability.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")

    chart_path = str(Path(__file__).parent / "rolling_persistence.png")
    print("Saving rolling persistence chart...")
    plot_rolling_persistence(rolling_df, save_path=chart_path)


if __name__ == "__main__":
    main()
