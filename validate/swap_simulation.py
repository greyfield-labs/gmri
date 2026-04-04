"""
GMRI Swap Payout Simulation
============================
Simulates 10,000 macro regime paths of 36 months each using the empirical
full-sample Markov transition matrix as the engine.

For each path, compute the payout on a regime swap:
  payout = (actual_months_in_regime − strike) × $2,000,000

Three swaps are analysed:
  - Stagflation      : strike = 14 months  (user-specified)
  - Goldilocks       : strike = round(hist_freq × 36)
  - Deflationary Bust: strike = round(hist_freq × 36)

Statistics reported per swap:
  expected payout, std deviation, P5, P95,
  P(|payout| > $20M), max positive payout, max negative payout.

Outputs:
  validate/swap_simulation.txt   — full numeric results
  validate/payout_distribution.png — 3-panel histogram

Run from project root:  python validate/swap_simulation.py
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE    = "1970-01-01"
CPI_THRESHOLD = 2.5
GDP_WINDOW    = 3
MIN_DURATION  = 3

N_PATHS     = 10_000
PATH_LENGTH = 36          # months per simulated path
RATE        = 2_000_000   # $2M per regime-month deviation
SG_STRIKE   = 14          # Stagflation strike — user-specified
SIM_SEED    = 42

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
REGIME_COLORS = {
    "Goldilocks":        "#4CAF50",
    "Overheating":       "#FF9800",
    "Stagflation":       "#F44336",
    "Deflationary Bust": "#2196F3",
}
SHORT = {
    "Goldilocks":        "GL",
    "Overheating":       "OV",
    "Stagflation":       "SG",
    "Deflationary Bust": "DB",
}

# ---------------------------------------------------------------------------
# FRED helpers (no imports from src/)
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
    return monthly.ffill()


def fetch_base_df(fred: Fred) -> pd.DataFrame:
    print("  CPIAUCSL...", end=" ", flush=True)
    cpi_raw = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("UNRATE...", end=" ", flush=True)
    unrate  = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("GDPC1...", end=" ", flush=True)
    gdpc1   = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), "cubic")
    print("NROU...", end=" ", flush=True)
    nrou    = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), "linear")
    print("GDPPOT...")
    gdppot  = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), "cubic")

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:].dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])
    return df


# ---------------------------------------------------------------------------
# Regime classification (verbatim from markov_stability.py)
# ---------------------------------------------------------------------------

def classify_regime(df: pd.DataFrame, gdp_window: int = GDP_WINDOW) -> pd.Series:
    gdp_gap    = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    gdp_smooth = gdp_gap.rolling(window=gdp_window, min_periods=1).mean()
    unemp_gap  = df["UNRATE"] - df["NROU"]

    growth_above = (gdp_smooth > 0) & (unemp_gap < 0)
    infl_above   = df["CPI_YOY"] > CPI_THRESHOLD

    conditions = [
        growth_above & ~infl_above,
        growth_above &  infl_above,
       ~growth_above &  infl_above,
       ~growth_above & ~infl_above,
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")
    return pd.Series(
        pd.Categorical(raw, categories=REGIME_ORDER),
        index=df.index, name="REGIME",
    )


def apply_min_duration_filter(regime: pd.Series, min_duration: int) -> pd.Series:
    if min_duration <= 1:
        return regime
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
# Transition matrix
# ---------------------------------------------------------------------------

def build_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """
    Row-stochastic probability matrix from empirical regime sequence.
    Uses manual loop (avoids pd.crosstab Categorical index bug).
    Rows with zero counts filled with uniform 0.25 fallback.
    """
    from_v = regime.astype(str).iloc[:-1]
    to_v   = regime.astype(str).iloc[1:]
    mat = pd.DataFrame(0, index=REGIME_ORDER, columns=REGIME_ORDER, dtype=float)
    for f, t in zip(from_v, to_v):
        if f in mat.index and t in mat.columns:
            mat.loc[f, t] += 1
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    prob = mat.div(row_sums, axis=0).fillna(0.25)
    # Validate
    row_totals = prob.sum(axis=1)
    assert np.allclose(row_totals, 1.0, atol=1e-9), \
        f"Transition matrix rows do not sum to 1: {row_totals.to_dict()}"
    return prob


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    P: np.ndarray,
    start_dist: np.ndarray,
    n_paths: int = N_PATHS,
    path_length: int = PATH_LENGTH,
    seed: int = SIM_SEED,
) -> np.ndarray:
    """
    Simulate regime paths using the Markov chain defined by P.

    Parameters
    ----------
    P           : (4, 4) row-stochastic numpy array
    start_dist  : (4,) starting regime distribution (must sum to 1)
    n_paths     : number of Monte Carlo paths
    path_length : months per path
    seed        : RNG seed for reproducibility

    Returns
    -------
    paths : (n_paths, path_length) int8 array of regime indices 0–3
    """
    rng  = np.random.default_rng(seed)
    cumP = np.cumsum(P, axis=1)  # (4, 4) cumulative probability matrix

    paths = np.empty((n_paths, path_length), dtype=np.int8)
    paths[:, 0] = rng.choice(len(REGIME_ORDER), size=n_paths, p=start_dist)

    # Pre-draw all uniform random numbers in one call (vectorized)
    u = rng.random((n_paths, path_length - 1))

    for t in range(1, path_length):
        prev = paths[:, t - 1]                        # (n_paths,) regime indices
        # For each path: count how many cumulative thresholds the draw exceeds
        # cumP[prev] has shape (n_paths, 4); u[:, t-1:t] broadcasts to (n_paths, 4)
        paths[:, t] = (u[:, t - 1: t] > cumP[prev]).sum(axis=1).astype(np.int8)

    return paths


# ---------------------------------------------------------------------------
# Payout and statistics
# ---------------------------------------------------------------------------

def compute_payouts(paths: np.ndarray, regime_idx: int, strike: int) -> np.ndarray:
    """
    Payout for party long the regime swap.
    payout[i] = (months_in_regime[i] − strike) × RATE
    Positive = regime appeared more than strike; negative = less.
    """
    months = (paths == regime_idx).sum(axis=1).astype(float)
    return (months - strike) * RATE


def compute_stats(payouts: np.ndarray) -> dict:
    return {
        "expected":        float(np.mean(payouts)),
        "std":             float(np.std(payouts)),
        "p5":              float(np.percentile(payouts,  5)),
        "p95":             float(np.percentile(payouts, 95)),
        "prob_exceed_20M": float(np.mean(np.abs(payouts) > 20_000_000)),
        "max_positive":    float(np.max(payouts)),
        "max_negative":    float(np.min(payouts)),
        "max_abs":         float(np.max(np.abs(payouts))),
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def fmt_m(x: float) -> str:
    """Format a dollar value in $M with sign, e.g. '$−12.34M'."""
    sign = "−" if x < 0 else ""
    return f"${sign}{abs(x) / 1e6:.2f}M"


def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def build_report(
    trans_prob: pd.DataFrame,
    hist_freq: dict,
    swaps: list[dict],
) -> list[str]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 68
    dash = "-" * 68

    log(sep)
    log("  GMRI SWAP PAYOUT SIMULATION")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Paths: {N_PATHS:,} × {PATH_LENGTH} months  |  Rate: ${RATE:,.0f} / regime-month")
    log(sep)
    log()

    # --- Empirical transition matrix ---
    log("  Empirical Transition Matrix (full-sample, row-stochastic):")
    col_hdr = f"  {'From → To':<22}" + "".join(f"  {SHORT[c]:>7}" for c in REGIME_ORDER)
    log(col_hdr)
    log("  " + "-" * 50)
    for from_r in REGIME_ORDER:
        row_str = f"  {from_r:<22}"
        for to_r in REGIME_ORDER:
            val = trans_prob.loc[from_r, to_r]
            row_str += f"  {val:>7.4f}"
        log(row_str)
    log()

    # --- Frequencies and strikes ---
    log("  Regime Frequencies & Strikes:")
    log(f"  {'Regime':<22}  {'Hist Freq':>9}  {'36-mo Exp':>9}  {'Strike':>8}")
    log("  " + "-" * 52)
    for s in swaps:
        exp_months = hist_freq[s["regime"]] * PATH_LENGTH
        strike_str = f"{s['strike']}  (user-specified)" if s["regime"] == "Stagflation" \
                     else str(s["strike"])
        log(f"  {s['regime']:<22}  {hist_freq[s['regime']]*100:>8.1f}%  {exp_months:>9.1f}  {strike_str:>8}")
    log()

    # --- Per-swap results ---
    for s in swaps:
        st = s["stats"]
        log(dash)
        log(f"  {s['regime'].upper()} SWAP  "
            f"(strike = {s['strike']} months, rate = ${RATE/1e6:.0f}M/mo)")
        log(dash)
        log(f"  Expected payout:            {fmt_m(st['expected']):>12}")
        log(f"  Std deviation:              {fmt_m(st['std']):>12}")
        log(f"  5th percentile:             {fmt_m(st['p5']):>12}")
        log(f"  95th percentile:            {fmt_m(st['p95']):>12}")
        log(f"  P(|payout| > $20M):         {fmt_pct(st['prob_exceed_20M']):>12}")
        log(f"  Maximum positive payout:    {fmt_m(st['max_positive']):>12}")
        log(f"  Maximum negative payout:    {fmt_m(st['max_negative']):>12}")
        log(f"  Maximum absolute payout:    {fmt_m(st['max_abs']):>12}")
        log()

    # --- Summary comparison table ---
    log(sep)
    log(f"  {'SUMMARY COMPARISON':^66}")
    log(sep)
    log()
    hdr = (f"  {'Swap':<22}  {'Strike':>6}  {'E[pay]':>9}  {'Std':>8}  "
           f"{'P5':>10}  {'P95':>10}  {'P(>$20M)':>8}")
    log(hdr)
    log("  " + "-" * 78)
    for s in swaps:
        st = s["stats"]
        log(f"  {s['regime']:<22}  {s['strike']:>6}  "
            f"{fmt_m(st['expected']):>9}  {fmt_m(st['std']):>8}  "
            f"{fmt_m(st['p5']):>10}  {fmt_m(st['p95']):>10}  "
            f"{fmt_pct(st['prob_exceed_20M']):>8}")
    log()

    # --- Interpretation ---
    log(sep)
    log(f"  {'INTERPRETATION':^66}")
    log(sep)
    log()
    most_vol = max(swaps, key=lambda s: s["stats"]["std"])
    most_tail = max(swaps, key=lambda s: s["stats"]["prob_exceed_20M"])
    fair_swap = min(swaps, key=lambda s: abs(s["stats"]["expected"]))
    log(f"  Most volatile swap (highest std):  {most_vol['regime']} "
        f"({fmt_m(most_vol['stats']['std'])} std)")
    log(f"  Most tail-risky (P > $20M):        {most_tail['regime']} "
        f"({fmt_pct(most_tail['stats']['prob_exceed_20M'])})")
    log(f"  Closest to fair (E[pay] ≈ 0):      {fair_swap['regime']} "
        f"({fmt_m(fair_swap['stats']['expected'])} expected)")
    log()
    log("  Note: Expected payouts near zero indicate fair strikes — the swap")
    log("  premium reflects path variance, not directional bias.")
    log()
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Histogram chart
# ---------------------------------------------------------------------------

def plot_histograms(swaps: list[dict], save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, s in zip(axes, swaps):
        payouts = s["payouts"]
        st      = s["stats"]
        color   = REGIME_COLORS[s["regime"]]
        regime  = s["regime"]
        strike  = s["strike"]

        # Histogram
        ax.hist(payouts / 1e6, bins=50, color=color, alpha=0.75, edgecolor="none")

        # ±$20M tail shading
        ymax = ax.get_ylim()[1]
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max)
        ax.axvspan(x_min, -20, color="grey", alpha=0.15, zorder=0)
        ax.axvspan( 20, x_max, color="grey", alpha=0.15, zorder=0)

        # Zero payout line
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Zero payout")

        # ±$20M boundary lines
        for bound in [-20, 20]:
            ax.axvline(bound, color="grey", linewidth=0.8, linestyle=":", alpha=0.8)

        # Mean line
        mean_m = st["expected"] / 1e6
        ax.axvline(mean_m, color="black", linewidth=1.5, linestyle="-",
                   label=f"Mean: {mean_m:+.2f}M")

        # P5 / P95 lines
        p5_m  = st["p5"]  / 1e6
        p95_m = st["p95"] / 1e6
        ax.axvline(p5_m,  color="navy", linewidth=1.0, linestyle="-.",
                   label=f"P5: {p5_m:+.1f}M")
        ax.axvline(p95_m, color="darkred", linewidth=1.0, linestyle="-.",
                   label=f"P95: {p95_m:+.1f}M")

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:+.0f}M"
        ))
        ax.set_xlabel("Payout ($M)", fontsize=10)
        ax.set_ylabel("Frequency" if s is swaps[0] else "", fontsize=10)
        ax.set_title(
            f"{regime} Swap\n(strike = {strike} mo, rate = $2M/mo)",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.tick_params(axis="x", labelsize=8, rotation=30)

    plt.suptitle(
        f"GMRI Swap Payout Distributions  —  {N_PATHS:,} paths × {PATH_LENGTH} months",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  GMRI SWAP PAYOUT SIMULATION")
    print("=" * 60)

    # --- Step 1: Fetch and classify ---
    print("\nFetching data from FRED...")
    fred   = get_fred_client()
    df     = fetch_base_df(fred)
    print(f"  {len(df)} months: "
          f"{df.index[0].strftime('%Y-%m')} – {df.index[-1].strftime('%Y-%m')}\n")

    regime = classify_regime(df, gdp_window=GDP_WINDOW)
    regime = apply_min_duration_filter(regime, min_duration=MIN_DURATION)
    print(f"  Regime series: {len(regime)} months classified\n")

    # --- Step 2: Transition matrix ---
    trans_prob = build_transition_matrix(regime)
    P = trans_prob.values.astype(float)   # (4, 4) numpy array

    # --- Step 3: Historical frequencies and strikes ---
    hist_freq = {r: float((regime.astype(str) == r).mean()) for r in REGIME_ORDER}
    GL_STRIKE = round(hist_freq["Goldilocks"]        * PATH_LENGTH)
    DB_STRIKE = round(hist_freq["Deflationary Bust"] * PATH_LENGTH)

    print("  Historical frequencies and derived strikes:")
    for r in REGIME_ORDER:
        exp = hist_freq[r] * PATH_LENGTH
        print(f"    {r:<22}  {hist_freq[r]*100:5.1f}%  → {exp:.1f} mo expected in 36")
    print()

    # Starting distribution = empirical frequencies
    start_dist = np.array([hist_freq[r] for r in REGIME_ORDER])
    start_dist /= start_dist.sum()   # normalize defensively

    # --- Step 4: Simulate ---
    print(f"  Simulating {N_PATHS:,} paths of {PATH_LENGTH} months (seed={SIM_SEED})...")
    paths = simulate_paths(P, start_dist, N_PATHS, PATH_LENGTH, SIM_SEED)
    print(f"  Paths shape: {paths.shape}\n")

    # --- Step 5–6: Payouts and stats for the three swaps ---
    swap_specs = [
        {"regime": "Stagflation",       "strike": SG_STRIKE},
        {"regime": "Goldilocks",        "strike": GL_STRIKE},
        {"regime": "Deflationary Bust", "strike": DB_STRIKE},
    ]
    swaps = []
    for spec in swap_specs:
        regime_idx = REGIME_ORDER.index(spec["regime"])
        payouts    = compute_payouts(paths, regime_idx, spec["strike"])
        stats      = compute_stats(payouts)
        swaps.append({**spec, "payouts": payouts, "stats": stats})
        print(f"  {spec['regime']:<22} strike={spec['strike']:>2}  "
              f"E[pay]={fmt_m(stats['expected'])}  "
              f"std={fmt_m(stats['std'])}  "
              f"P(>$20M)={fmt_pct(stats['prob_exceed_20M'])}")
    print()

    # --- Step 7: Text report ---
    lines = build_report(trans_prob, hist_freq, swaps)
    txt_path = Path(__file__).parent / "swap_simulation.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {txt_path}")

    # --- Step 8: Histogram chart ---
    png_path = str(Path(__file__).parent / "payout_distribution.png")
    plot_histograms(swaps, png_path)


if __name__ == "__main__":
    main()
