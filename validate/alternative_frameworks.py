"""
GMRI Alternative Framework Comparison
=======================================
Benchmarks the GMRI growth signal against three alternatives while holding
the CPI inflation signal (CPI YoY > 2.5%) and min-duration filter (3 months)
identical across all frameworks.

Frameworks compared:
  1. Baseline GMRI  — GDP gap (vs CBO potential, 3-mo smoothed) AND unemployment gap
  2. HP Filter      — statsmodels hpfilter(GDPC1, lamb=1600); cycle > 0 = above trend
  3. NBER Binary    — USREC == 0 (expansion = above trend, recession = below)
  4. CFNAI          — Chicago Fed National Activity Index >= 0 = above trend

All four frameworks are evaluated over the same date range (inner join of all
series after CPI YoY lag). No imports from the existing src/ codebase.

Outputs:
  validate/alternative_frameworks.txt  — comparison tables + transition matrices
  validate/framework_comparison.png    — 3-panel bar chart

Run from project root:  python validate/alternative_frameworks.py
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE    = "1970-01-01"
CPI_THRESHOLD = 2.5
MIN_DURATION  = 3
HP_LAMBDA     = 1600
FIG_DPI       = 150

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
SHORT = {"Goldilocks": "GL", "Overheating": "OV",
         "Stagflation": "SG", "Deflationary Bust": "DB"}

FRAMEWORK_COLORS = ["#37474F", "#E65100", "#1565C0", "#2E7D32"]
FRAMEWORK_LABELS = ["Baseline\nGMRI", "HP Filter\n(λ=1600)", "NBER\nBinary", "CFNAI\n(≥0)"]
FRAMEWORK_KEYS   = ["baseline", "hp_filter", "nber", "cfnai"]

REGIME_COLORS = {
    "Goldilocks":        "#4CAF50",
    "Overheating":       "#FF9800",
    "Stagflation":       "#F44336",
    "Deflationary Bust": "#2196F3",
}

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def get_fred_client() -> Fred:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(f"FRED_API_KEY not found. Expected .env at: {env_path}")
    return Fred(api_key=key)


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
    return monthly.ffill()


def fetch_all(fred: Fred) -> dict[str, pd.Series]:
    """Fetch every series needed. Returns dict of aligned month-end Series."""
    print("  CPIAUCSL...", end=" ", flush=True)
    cpi_raw  = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("UNRATE...",    end=" ", flush=True)
    unrate   = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("GDPC1...",     end=" ", flush=True)
    gdpc1    = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), "cubic")
    print("NROU...",      end=" ", flush=True)
    nrou     = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), "linear")
    print("GDPPOT...",    end=" ", flush=True)
    gdppot   = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), "cubic")
    print("USREC...",     end=" ", flush=True)
    usrec    = to_month_end(fetch(fred, "USREC",  start=START_DATE))
    print("CFNAI...")
    cfnai    = to_month_end(fetch(fred, "CFNAI",  start=START_DATE))

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"
    usrec.name   = "USREC"
    cfnai.name   = "CFNAI"

    return dict(
        cpi_yoy=cpi_yoy, unrate=unrate, gdpc1=gdpc1,
        nrou=nrou, gdppot=gdppot, usrec=usrec, cfnai=cfnai,
    )


def build_common_df(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join all series, trim to START_DATE, drop any remaining NaN rows."""
    df = pd.concat(list(series.values()), axis=1, join="inner")
    df = df.loc[START_DATE:].dropna()
    return df


# ---------------------------------------------------------------------------
# Growth signal builders
# ---------------------------------------------------------------------------

def growth_baseline(df: pd.DataFrame) -> pd.Series:
    gdp_gap    = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    gdp_smooth = gdp_gap.rolling(window=3, min_periods=1).mean()
    unemp_gap  = df["UNRATE"] - df["NROU"]
    above = (gdp_smooth > 0) & (unemp_gap < 0)
    above.name = "growth_above"
    return above


def growth_hp_filter(df: pd.DataFrame) -> pd.Series:
    """
    Apply HP filter (lambda=1600) to the monthly-interpolated GDPC1 series.
    lambda=1600 is the quarterly convention; on monthly data it under-smooths
    the trend, leaving a noisier cycle component.
    """
    gdpc1_clean = df["GDPC1"].dropna()
    cycle, _trend = hpfilter(gdpc1_clean, lamb=HP_LAMBDA)
    # Re-index cycle onto full df index (may differ if dropna trimmed anything)
    cycle = cycle.reindex(df.index).ffill().bfill()
    above = (cycle > 0)
    above.name = "growth_above"
    return above


def growth_nber(df: pd.DataFrame) -> pd.Series:
    """USREC == 0 (expansion) → above trend; USREC == 1 (recession) → below trend."""
    above = (df["USREC"] == 0)
    above.name = "growth_above"
    return above


def growth_cfnai(df: pd.DataFrame) -> pd.Series:
    """CFNAI >= 0 → above trend; CFNAI < 0 → below trend."""
    above = (df["CFNAI"] >= 0)
    above.name = "growth_above"
    return above


# ---------------------------------------------------------------------------
# Classification helpers (no src/ imports)
# ---------------------------------------------------------------------------

def assign_regime(growth_above: pd.Series, infl_above: pd.Series) -> pd.Series:
    conditions = [
        growth_above & ~infl_above,
        growth_above &  infl_above,
       ~growth_above &  infl_above,
       ~growth_above & ~infl_above,
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")
    return pd.Series(
        pd.Categorical(raw, categories=REGIME_ORDER),
        index=growth_above.index, name="REGIME",
    )


def apply_min_duration_filter(regime: pd.Series, min_duration: int = 3) -> pd.Series:
    filtered = regime.astype(str).copy()
    for _ in range(20):
        block_id = (filtered != filtered.shift()).cumsum()
        changed  = False
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


def classify_framework(
    df: pd.DataFrame,
    growth_fn,
    min_duration: int = MIN_DURATION,
) -> pd.Series:
    growth_above = growth_fn(df)
    infl_above   = df["CPI_YOY"] > CPI_THRESHOLD
    regime_raw   = assign_regime(growth_above, infl_above)
    return apply_min_duration_filter(regime_raw, min_duration)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def transition_count_matrix(regime: pd.Series) -> pd.DataFrame:
    """Manual count matrix — avoids pd.crosstab Categorical label bugs."""
    mat = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER, dtype=float)
    from_v = regime.astype(str).iloc[:-1]
    to_v   = regime.astype(str).iloc[1:]
    for f, t in zip(from_v, to_v):
        if f in mat.index and t in mat.columns:
            mat.loc[f, t] += 1
    mat.index.name   = "From"
    mat.columns.name = "To"
    return mat


def transition_probs(counts: pd.DataFrame) -> pd.DataFrame:
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(row_sums, axis=0)


def compute_metrics(regime: pd.Series) -> dict:
    total = len(regime)
    pct   = {r: (regime == r).sum() / total * 100 for r in REGIME_ORDER}

    # Transitions = actual regime changes (excluding self-stays)
    regime_str    = regime.astype(str)
    n_transitions = int((regime_str != regime_str.shift()).sum()) - 1

    counts = transition_count_matrix(regime)
    probs  = transition_probs(counts)
    diag   = {r: float(probs.loc[r, r]) for r in REGIME_ORDER}
    avg_diag = float(np.nanmean(list(diag.values())))

    return {
        "pct_goldilocks":        round(pct["Goldilocks"],        1),
        "pct_overheating":       round(pct["Overheating"],       1),
        "pct_stagflation":       round(pct["Stagflation"],       1),
        "pct_deflationary_bust": round(pct["Deflationary Bust"], 1),
        "n_transitions":         n_transitions,
        "diag_goldilocks":       round(diag["Goldilocks"],        4),
        "diag_overheating":      round(diag["Overheating"],       4),
        "diag_stagflation":      round(diag["Stagflation"],       4),
        "diag_deflationary_bust":round(diag["Deflationary Bust"], 4),
        "avg_diag":              round(avg_diag,                  4),
        "counts":                counts,
        "probs":                 probs,
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_comparison(results: dict[str, dict], baseline_key: str, save_path: str) -> None:
    labels = FRAMEWORK_LABELS
    keys   = FRAMEWORK_KEYS
    colors = FRAMEWORK_COLORS

    gl_vals    = [results[k]["pct_goldilocks"]  for k in keys]
    tr_vals    = [results[k]["n_transitions"]   for k in keys]
    diag_vals  = [results[k]["avg_diag"]        for k in keys]

    baseline_gl   = results[baseline_key]["pct_goldilocks"]
    baseline_tr   = results[baseline_key]["n_transitions"]
    baseline_diag = results[baseline_key]["avg_diag"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Alternative Growth Signal Comparison  (CPI threshold and duration filter held constant)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    specs = [
        (axes[0], gl_vals,   baseline_gl,   "Goldilocks Frequency (%)",
         "% of months in Goldilocks", REGIME_COLORS["Goldilocks"]),
        (axes[1], tr_vals,   baseline_tr,   "Total Regime Transitions",
         "Count of month-to-month changes", "#546E7A"),
        (axes[2], diag_vals, baseline_diag, "Avg Diagonal Persistence",
         "Mean of 4 diagonal transition probs", "#7B1FA2"),
    ]

    x = np.arange(len(keys))

    for ax, vals, base_val, title, ylabel, accent in specs:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55,
                      edgecolor="white", linewidth=0.8)

        # Baseline reference line
        ax.axhline(base_val, color=colors[0], linewidth=1.4,
                   linestyle="--", alpha=0.7, label=f"Baseline: {base_val:.2f}")

        # Value labels above each bar
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1f}" if isinstance(val, float) else str(val)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Delta annotations below bar tops (vs baseline)
        for bar, val, key in zip(bars, vals, keys):
            if key == baseline_key:
                continue
            delta = val - base_val
            sign  = "+" if delta >= 0 else ""
            fmt_d = f"{sign}{delta:.1f}" if isinstance(delta, float) else f"{sign}{delta}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f"Δ{fmt_d}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        # Y-axis: percentage format for GL panel, otherwise auto
        if "%" in ylabel:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

        # Add some headroom above bars for labels
        cur_top = max(vals)
        ax.set_ylim(0, cur_top * 1.22)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(results: dict[str, dict], df: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 72
    dash = "-" * 72

    log(sep)
    log("  GMRI ALTERNATIVE FRAMEWORK COMPARISON")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Common date range: {df.index[0].strftime('%Y-%m')} – "
        f"{df.index[-1].strftime('%Y-%m')}  ({len(df)} months)")
    log(f"  CPI threshold: {CPI_THRESHOLD}%  |  Min duration: {MIN_DURATION}mo  "
        f"  (identical across all frameworks)")
    log(sep)
    log()

    # Note on HP filter
    log("  NOTE: HP filter uses lambda=1600 (quarterly convention) applied to")
    log("  monthly-interpolated GDPC1. This under-smooths the trend on monthly")
    log("  data, leaving a noisier cycle component → expect more transitions.")
    log()

    # ------------------------------------------------------------------
    # Framework descriptions
    # ------------------------------------------------------------------
    log("FRAMEWORK DEFINITIONS")
    log(dash)
    descs = [
        ("Baseline GMRI",
         "GDP gap = (GDPC1 − GDPPOT)/GDPPOT × 100, 3-mo rolling smooth;\n"
         "   AND unemployment gap (UNRATE − NROU) < 0. Dual-signal AND logic."),
        ("HP Filter (λ=1600)",
         "Hodrick-Prescott filter applied to monthly GDPC1 with λ=1600.\n"
         "   Growth above trend ⟺ HP cycle component > 0."),
        ("NBER Binary",
         "USREC indicator from FRED. Growth above trend ⟺ USREC = 0\n"
         "   (NBER expansion). Binary step function — no gradation."),
        ("CFNAI (≥0)",
         "Chicago Fed National Activity Index. Growth above trend ⟺ CFNAI ≥ 0.\n"
         "   85-indicator composite; negative = below historical trend."),
    ]
    for label, desc in zip(FRAMEWORK_LABELS, descs):
        log(f"  {label.replace(chr(10), ' ')}")
        log(f"   {desc}")
        log()

    # ------------------------------------------------------------------
    # Regime distribution table
    # ------------------------------------------------------------------
    log("REGIME DISTRIBUTION")
    log(dash)
    hdr = (f"  {'Framework':<20}  {'GL%':>6}  {'OV%':>6}  {'SG%':>6}  {'DB%':>6}"
           f"  {'N_trans':>8}  {'Avg_diag':>9}")
    log(hdr)
    log("  " + "-" * 68)

    base = results["baseline"]
    for key, label in zip(FRAMEWORK_KEYS, FRAMEWORK_LABELS):
        r = results[key]
        lbl = label.replace("\n", " ")
        log(f"  {lbl:<20}  {r['pct_goldilocks']:>6.1f}  {r['pct_overheating']:>6.1f}"
            f"  {r['pct_stagflation']:>6.1f}  {r['pct_deflationary_bust']:>6.1f}"
            f"  {r['n_transitions']:>8d}  {r['avg_diag']:>9.4f}")
    log()

    # ------------------------------------------------------------------
    # Delta table
    # ------------------------------------------------------------------
    log("DELTA VS BASELINE GMRI")
    log(dash)
    hdr2 = (f"  {'Framework':<20}  {'ΔGL%':>7}  {'ΔOV%':>7}  {'ΔSG%':>7}  {'ΔDB%':>7}"
            f"  {'ΔN_trans':>9}  {'ΔAvg_diag':>10}")
    log(hdr2)
    log("  " + "-" * 72)

    for key, label in zip(FRAMEWORK_KEYS[1:], FRAMEWORK_LABELS[1:]):
        r   = results[key]
        lbl = label.replace("\n", " ")
        dgl  = r["pct_goldilocks"]        - base["pct_goldilocks"]
        dov  = r["pct_overheating"]       - base["pct_overheating"]
        dsg  = r["pct_stagflation"]       - base["pct_stagflation"]
        ddb  = r["pct_deflationary_bust"] - base["pct_deflationary_bust"]
        dtr  = r["n_transitions"]         - base["n_transitions"]
        ddiag = r["avg_diag"]             - base["avg_diag"]
        log(f"  {lbl:<20}  {dgl:>+7.1f}  {dov:>+7.1f}  {dsg:>+7.1f}  {ddb:>+7.1f}"
            f"  {dtr:>+9d}  {ddiag:>+10.4f}")
    log()

    # ------------------------------------------------------------------
    # Per-framework transition matrices
    # ------------------------------------------------------------------
    log("TRANSITION PROBABILITY MATRICES")
    log(dash)

    for key, label in zip(FRAMEWORK_KEYS, FRAMEWORK_LABELS):
        r     = results[key]
        probs = r["probs"]
        counts = r["counts"]
        lbl   = label.replace("\n", " ")
        log(f"\n  {lbl}")
        col_hdr = f"  {'From → To':<22}" + "".join(f"  {SHORT[c]:>9}" for c in REGIME_ORDER)
        log(col_hdr)
        log("  " + "-" * 62)
        for from_r in REGIME_ORDER:
            row_str = f"  {from_r:<22}"
            for to_r in REGIME_ORDER:
                p = probs.loc[from_r, to_r]
                n = int(counts.loc[from_r, to_r])
                marker = "*" if from_r == to_r else " "
                row_str += f"  {p:.3f}({n:>3}){marker}"
            log(row_str)
        row_totals = counts.sum(axis=1)
        totals_str = f"  {'Row totals':<22}" + "".join(
            f"  {'':>5}{int(row_totals[r]):>3} " for r in REGIME_ORDER
        )
        log("  " + "-" * 62)
        log(totals_str)
    log()

    # ------------------------------------------------------------------
    # Summary interpretation
    # ------------------------------------------------------------------
    log(sep)
    log("  INTERPRETATION")
    log(sep)
    log()

    # Compute which framework is closest to / farthest from baseline on GL%
    gl_base = base["pct_goldilocks"]
    gl_deltas = {k: abs(results[k]["pct_goldilocks"] - gl_base)
                 for k in FRAMEWORK_KEYS[1:]}
    closest = min(gl_deltas, key=gl_deltas.get)
    farthest = max(gl_deltas, key=gl_deltas.get)

    label_map = dict(zip(FRAMEWORK_KEYS, [l.replace("\n"," ") for l in FRAMEWORK_LABELS]))

    log(f"  Goldilocks frequency range across frameworks: "
        f"{min(results[k]['pct_goldilocks'] for k in FRAMEWORK_KEYS):.1f}% – "
        f"{max(results[k]['pct_goldilocks'] for k in FRAMEWORK_KEYS):.1f}%")
    log(f"  Most similar to baseline: {label_map[closest]} "
        f"(Δ={gl_deltas[closest]:+.1f}pp)")
    log(f"  Most divergent from baseline: {label_map[farthest]} "
        f"(Δ={results[farthest]['pct_goldilocks'] - gl_base:+.1f}pp)")
    log()

    tr_base = base["n_transitions"]
    log(f"  Transition count range: "
        f"{min(results[k]['n_transitions'] for k in FRAMEWORK_KEYS)} – "
        f"{max(results[k]['n_transitions'] for k in FRAMEWORK_KEYS)}")
    log(f"  Baseline: {tr_base}  |  "
        f"HP Filter: {results['hp_filter']['n_transitions']}  |  "
        f"NBER: {results['nber']['n_transitions']}  |  "
        f"CFNAI: {results['cfnai']['n_transitions']}")
    log()

    diag_base = base["avg_diag"]
    log(f"  Avg persistence range: "
        f"{min(results[k]['avg_diag'] for k in FRAMEWORK_KEYS):.4f} – "
        f"{max(results[k]['avg_diag'] for k in FRAMEWORK_KEYS):.4f}")
    log()
    log("  Key finding: if alternative frameworks produce materially different")
    log("  Goldilocks frequencies (>5pp) or transition counts (>20%), the")
    log("  GMRI regime assignments are sensitive to the growth signal choice")
    log("  and that assumption should be disclosed in the white paper.")
    log()
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  GMRI ALTERNATIVE FRAMEWORK COMPARISON")
    print("=" * 60)

    print("\nFetching data from FRED (once)...")
    fred   = get_fred_client()
    series = fetch_all(fred)
    df     = build_common_df(series)
    print(f"  Common range: {df.index[0].strftime('%Y-%m')} – "
          f"{df.index[-1].strftime('%Y-%m')}  ({len(df)} months)\n")

    # Build each framework's regime series
    print("Classifying all four frameworks...")
    framework_fns = {
        "baseline":  growth_baseline,
        "hp_filter": growth_hp_filter,
        "nber":      growth_nber,
        "cfnai":     growth_cfnai,
    }

    results: dict[str, dict] = {}
    for key, fn in framework_fns.items():
        label = FRAMEWORK_LABELS[FRAMEWORK_KEYS.index(key)].replace("\n", " ")
        print(f"  {label}...")
        regime          = classify_framework(df, fn)
        results[key]    = compute_metrics(regime)

    # Build and save report
    print("\nBuilding report...")
    lines = build_report(results, df)

    out_path = Path(__file__).parent / "alternative_frameworks.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")

    # Build and save chart
    print("Saving chart...")
    chart_path = str(Path(__file__).parent / "framework_comparison.png")
    plot_comparison(results, baseline_key="baseline", save_path=chart_path)


if __name__ == "__main__":
    main()
