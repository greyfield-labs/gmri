"""
GMRI Independent Replication Test
==================================
Reproduces the Macro Regime Index from scratch using raw FRED API data.
No imports from the existing codebase.

Run from project root:  python validate/replication.py
Run from validate/:     python replication.py
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Parameters — hardcoded to match the production run in main.py
# ---------------------------------------------------------------------------
START_DATE     = "1970-01-01"
CPI_THRESHOLD  = 2.5       # % — inflation above/below threshold
GDP_GAP_WINDOW = 3         # months — rolling smoothing window
MIN_DURATION   = 3         # months — minimum spell length before filter
TOLERANCE      = 2.0       # ±pp — pass/fail tolerance for regime % time

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]

# Reference values from the production run (main.py output 2026-04-03)
REFERENCE = {
    "Goldilocks":        9.3,
    "Overheating":      28.0,   # production: 27.5
    "Stagflation":      40.0,
    "Deflationary Bust": 23.0,  # production: 23.1
}

# Expected intermediate diagnostic values
EXPECTED_DIAG = {
    "n_months":          657,
    "spells_before":      55,
    "spells_after":       37,
    "spells_eliminated":  18,
}

# ---------------------------------------------------------------------------
# Helpers — data fetching
# ---------------------------------------------------------------------------

def get_fred_client() -> Fred:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            f"FRED_API_KEY not found. Expected .env at: {env_path}"
        )
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
    """Expand quarterly series to monthly via interpolation then forward-fill."""
    monthly = s.resample("ME").asfreq()
    monthly = monthly.interpolate(method=method)
    monthly = monthly.ffill()
    return monthly


# ---------------------------------------------------------------------------
# Helpers — classification
# ---------------------------------------------------------------------------

def compute_gdp_gap(gdpc1: pd.Series, gdppot: pd.Series) -> pd.Series:
    gap = (gdpc1 - gdppot) / gdppot * 100
    gap.name = "GDP_GAP"
    return gap


def smooth_series(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window=window, min_periods=1).mean()


def compute_unemp_gap(unrate: pd.Series, nrou: pd.Series) -> pd.Series:
    gap = unrate - nrou
    gap.name = "UNEMP_GAP"
    return gap


def growth_above_trend(gdp_gap_smooth: pd.Series, unemp_gap: pd.Series) -> pd.Series:
    """AND logic: both GDP gap > 0 AND unemployment gap < 0 required."""
    return (gdp_gap_smooth > 0) & (unemp_gap < 0)


def inflation_above_threshold(cpi_yoy: pd.Series, threshold: float = 2.5) -> pd.Series:
    return cpi_yoy > threshold


def assign_regime(growth_above: pd.Series, inflation_above: pd.Series) -> pd.Series:
    conditions = [
        growth_above & ~inflation_above,    # Goldilocks
        growth_above & inflation_above,     # Overheating
        ~growth_above & inflation_above,    # Stagflation
        ~growth_above & ~inflation_above,   # Deflationary Bust
    ]
    regime = np.select(conditions, REGIME_ORDER, default="Unknown")
    return pd.Series(
        pd.Categorical(regime, categories=REGIME_ORDER),
        index=growth_above.index,
        name="REGIME",
    )


# ---------------------------------------------------------------------------
# Helpers — min-duration filter (exact match to production algorithm)
# ---------------------------------------------------------------------------

def apply_min_duration_filter(regime: pd.Series, min_duration: int = 3) -> pd.Series:
    """
    Absorb spells shorter than min_duration into the preceding regime.
    Iterates up to 20 passes until no short spells remain.
    The first spell is never absorbed (no predecessor).
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
    return pd.Series(
        pd.Categorical(filtered, categories=REGIME_ORDER),
        index=regime.index,
        name="REGIME",
    )


def count_spells(regime: pd.Series) -> int:
    """Count total number of contiguous same-value spells."""
    return int((regime.astype(str) != regime.astype(str).shift()).sum())


# ---------------------------------------------------------------------------
# Helpers — statistics
# ---------------------------------------------------------------------------

def regime_stats(regime: pd.Series) -> pd.DataFrame:
    total = len(regime)
    block_id = (regime.astype(str) != regime.astype(str).shift()).cumsum()
    blocks = (
        pd.DataFrame({"r": regime, "b": block_id})
        .groupby("b")
        .agg(regime=("r", "first"), length=("r", "count"))
        .reset_index(drop=True)
    )
    rows = []
    for r in REGIME_ORDER:
        lengths = blocks[blocks["regime"] == r]["length"]
        rows.append({
            "regime": r,
            "pct_time": round((regime == r).sum() / total * 100, 1),
            "avg_duration": round(float(lengths.mean()), 1) if len(lengths) > 0 else 0.0,
            "n_spells": len(lengths),
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Failure diagnosis
# ---------------------------------------------------------------------------

def diagnose_failure(regime: str, replicated: float, reference: float) -> str:
    direction = "too high" if replicated > reference else "too low"
    hints = {
        "Goldilocks": {
            "too low":  "AND growth logic may be eliminating valid growth months; "
                        "check NROU interpolation (linear vs cubic) — shifts unemployment gap",
            "too high": "CPI threshold may be too high (try 2.0%); "
                        "verify growth logic is AND not OR",
        },
        "Overheating": {
            "too low":  "CPI threshold may be too high (try 2.0%); "
                        "verify GDPPOT cubic interpolation is not undershooting",
            "too high": "CPI threshold may be too low; "
                        "check DIAG-3 GDPPOT spot-check against 1990-Q3 reference",
        },
        "Stagflation": {
            "too low":  "CPI threshold may be too high, deflating Stagflation count; "
                        "check DIAG-6 inflation-months count",
            "too high": "NROU interpolation mismatch — using cubic instead of linear "
                        "shifts unemployment gap, reclassifying Deflationary Bust as Stagflation; "
                        "verify DIAG-5 unemp gap at 2009-12 is ~4pp",
        },
        "Deflationary Bust": {
            "too low":  "NROU over-interpolated (cubic instead of linear) — "
                        "drives unemployment gap negative, masking below-trend growth",
            "too high": "GDP gap spending too long below zero; "
                        "check GDPC1 or GDPPOT cubic interpolation accuracy via DIAG-3",
        },
    }
    return hints.get(regime, {}).get(direction, f"Unexpected divergence ({direction})")


# ---------------------------------------------------------------------------
# Main replication pipeline
# ---------------------------------------------------------------------------

def run_replication() -> bool:
    lines: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep = "=" * 55

    log(sep)
    log("  GMRI INDEPENDENT REPLICATION REPORT")
    log(f"  Generated: {timestamp}")
    log(sep)
    log()
    log("PARAMETERS")
    log(f"  Start date:       {START_DATE} (1971-01 after CPI YoY lag)")
    log(f"  CPI threshold:    {CPI_THRESHOLD}%")
    log(f"  GDP gap window:   {GDP_GAP_WINDOW} months (rolling mean)")
    log(f"  Growth logic:     AND (GDP gap > 0 AND unemp gap < 0)")
    log(f"  Min duration:     {MIN_DURATION} months")
    log(f"  Tolerance:        ±{TOLERANCE}pp")
    log()

    # ------------------------------------------------------------------
    # STEP 1: Fetch and assemble data
    # ------------------------------------------------------------------
    log("STEP 1 — Fetching data from FRED")
    fred = get_fred_client()

    cpi_raw  = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    unrate   = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    gdpc1_q  = to_month_end(fetch(fred, "GDPC1",    start="1947-01-01"))
    nrou_q   = to_month_end(fetch(fred, "NROU",     start="1947-01-01"))
    gdppot_q = to_month_end(fetch(fred, "GDPPOT",   start="1947-01-01"))

    # Interpolate quarterly → monthly
    gdpc1  = interp_quarterly(gdpc1_q,  method="cubic")
    gdppot = interp_quarterly(gdppot_q, method="cubic")
    nrou   = interp_quarterly(nrou_q,   method="linear")

    gdpc1.name  = "GDPC1"
    gdppot.name = "GDPPOT"
    nrou.name   = "NROU"

    # CPI year-over-year
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    # Assemble master DataFrame
    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:]
    df = df.dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])

    n_months = len(df)
    date_start = df.index[0].strftime("%Y-%m")
    date_end   = df.index[-1].strftime("%Y-%m")

    # Diagnostics
    log("DATA DIAGNOSTICS")
    diag1_ok = (n_months == EXPECTED_DIAG["n_months"])
    log(f"  [DIAG-1] Date range: {date_start} to {date_end} "
        f"({n_months} months)  ← expect {EXPECTED_DIAG['n_months']}"
        f"  {'✓' if diag1_ok else '✗ MISMATCH'}")

    cpi_1980 = df.loc["1980-01-01":"1980-12-31", "CPI_YOY"].max()
    log(f"  [DIAG-2] CPI YoY max in 1980: {cpi_1980:.1f}%  ← expect ~14%")

    spot_date = "1990-09-30"
    if spot_date in df.index:
        g3 = df.loc[spot_date, "GDPC1"]
        p3 = df.loc[spot_date, "GDPPOT"]
        log(f"  [DIAG-3] GDPC1 vs GDPPOT at 1990-09: "
            f"GDPC1={g3:,.1f}  GDPPOT={p3:,.1f}  gap={(g3-p3)/p3*100:.2f}%")
    else:
        log(f"  [DIAG-3] 1990-09-30 not in index, skipping spot-check")

    log()

    # ------------------------------------------------------------------
    # STEP 2: Classify
    # ------------------------------------------------------------------
    log("STEP 2 — Classification")

    gdp_gap        = compute_gdp_gap(df["GDPC1"], df["GDPPOT"])
    gdp_gap_smooth = smooth_series(gdp_gap, window=GDP_GAP_WINDOW)
    unemp_gap      = compute_unemp_gap(df["UNRATE"], df["NROU"])
    growth_above   = growth_above_trend(gdp_gap_smooth, unemp_gap)
    infl_above     = inflation_above_threshold(df["CPI_YOY"], threshold=CPI_THRESHOLD)
    regime_raw     = assign_regime(growth_above, infl_above)

    log(f"  [DIAG-4] GDP gap range: [{gdp_gap_smooth.min():.2f}%, {gdp_gap_smooth.max():.2f}%]"
        f"  ← expect roughly -5% to +4%")

    dec_2009 = "2009-12-31"
    if dec_2009 in df.index:
        ug_val = unemp_gap.loc[dec_2009]
        log(f"  [DIAG-5] Unemployment gap at 2009-12: {ug_val:.2f}pp"
            f"  ← expect ~+4pp (labor market deeply slack)")
    else:
        log(f"  [DIAG-5] 2009-12-31 not in index, skipping unemployment gap spot-check")

    n_infl = int(infl_above.sum())
    pct_infl = n_infl / n_months * 100
    log(f"  [DIAG-6] Months with CPI > {CPI_THRESHOLD}%: {n_infl} ({pct_infl:.1f}%)"
        f"  ← sanity check")

    log()

    # ------------------------------------------------------------------
    # STEP 3: Min-duration filter
    # ------------------------------------------------------------------
    log("STEP 3 — Min-duration filter")

    spells_before = count_spells(regime_raw)
    regime_filtered = apply_min_duration_filter(regime_raw, min_duration=MIN_DURATION)
    spells_after = count_spells(regime_filtered)
    spells_eliminated = spells_before - spells_after

    ok7 = (spells_before == EXPECTED_DIAG["spells_before"])
    ok8 = (spells_after  == EXPECTED_DIAG["spells_after"])
    ok9 = (spells_eliminated == EXPECTED_DIAG["spells_eliminated"])

    log(f"  [DIAG-7] Regime spells before filter: {spells_before}"
        f"  ← expect {EXPECTED_DIAG['spells_before']}  {'✓' if ok7 else '✗ MISMATCH'}")
    log(f"  [DIAG-8] Regime spells after  filter: {spells_after}"
        f"  ← expect {EXPECTED_DIAG['spells_after']}  {'✓' if ok8 else '✗ MISMATCH'}")
    log(f"  [DIAG-9] Spells eliminated:           {spells_eliminated}"
        f"  ← expect {EXPECTED_DIAG['spells_eliminated']}  {'✓' if ok9 else '✗ MISMATCH'}")

    log()

    # ------------------------------------------------------------------
    # STEP 4: Compute regime statistics
    # ------------------------------------------------------------------
    log("STEP 4 — Regime statistics")
    stats = regime_stats(regime_filtered)

    log(f"  {'Regime':<22} {'% Time':>7} {'Avg Dur':>9} {'# Spells':>9}")
    log("  " + "-" * 52)
    for r in REGIME_ORDER:
        row = stats.loc[r]
        log(f"  {r:<22} {row['pct_time']:>6.1f}% "
            f"{row['avg_duration']:>7.1f}mo "
            f"{int(row['n_spells']):>8d}")

    log()

    # ------------------------------------------------------------------
    # STEP 5: Compare against reference
    # ------------------------------------------------------------------
    log(f"REPLICATION RESULTS (tolerance ±{TOLERANCE}pp)")
    log("-" * 55)

    all_pass = True
    n_pass = 0
    for r in REGIME_ORDER:
        rep   = stats.loc[r, "pct_time"]
        ref   = REFERENCE[r]
        delta = abs(rep - ref)
        status = "PASS" if delta <= TOLERANCE else "FAIL"
        if status == "PASS":
            n_pass += 1
        else:
            all_pass = False

        msg = (f"  [{status}] {r:<22}: "
               f"replicated={rep:>5.1f}%  "
               f"reference={ref:>5.1f}%  "
               f"Δ={delta:.1f}pp")
        log(msg)

        if status == "FAIL":
            hint = diagnose_failure(r, rep, ref)
            log(f"           ^ Possible cause: {hint}")

    log()
    verdict = "PASS" if all_pass else "FAIL"
    log(f"OVERALL: {verdict} ({n_pass}/{len(REGIME_ORDER)} regimes within ±{TOLERANCE}pp)")

    # Additional diagnostic flag if spell counts diverge
    if not (ok7 and ok8):
        log()
        log("WARNING: Intermediate spell counts diverge from expected values.")
        log("  Most likely cause: interpolation method mismatch.")
        log("  Verify GDPC1/GDPPOT use method='cubic' and NROU uses method='linear'.")

    log(sep)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report_path = Path(__file__).parent / "replication_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    return all_pass


if __name__ == "__main__":
    success = run_replication()
    raise SystemExit(0 if success else 1)
