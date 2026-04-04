"""
GMRI Diagonal Transition Audit
================================
Tests the "zero diagonal transition" claim across three methodology versions:

  V1 — Raw:      no smoothing (window=1),  no duration filter (min=1)
  V2 — Smoothed: smoothing (window=3),     no duration filter (min=1)
  V3 — Full:     smoothing (window=3),     duration filter (min=3)

The four audited transitions are:
  1. Goldilocks     → Stagflation        (both signals flip simultaneously)
  2. Goldilocks     → Deflationary Bust  (growth flips, inflation stays below)
  3. Stagflation    → Goldilocks         (both signals flip simultaneously)
  4. Deflationary Bust → Overheating     (both signals flip simultaneously)

In the 2×2 regime matrix:

              │ Inflation ≤ 2.5%    │ Inflation > 2.5%
  ────────────┼─────────────────────┼──────────────────
  Growth ↑    │  Goldilocks   (GL)  │  Overheating (OV)
  Growth ↓    │  Deflationary (DB)  │  Stagflation (SG)

Transitions 1, 3, 4 are true diagonals (both signals flip).
Transition 2 is a "skip" — growth flips but inflation stays sub-threshold;
included because GL→DB bypasses OV or SG and is economically implausible
as a single-month jump.

Run from project root:  python validate/diagonal_transitions.py
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE    = "1970-01-01"
CPI_THRESHOLD = 2.5

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
SHORT = {
    "Goldilocks":       "GL",
    "Overheating":      "OV",
    "Stagflation":      "SG",
    "Deflationary Bust":"DB",
}

# The four transitions to audit — (from, to)
AUDIT_TRANSITIONS = [
    ("Goldilocks",        "Stagflation"),
    ("Goldilocks",        "Deflationary Bust"),
    ("Stagflation",       "Goldilocks"),
    ("Deflationary Bust", "Overheating"),
]

VERSIONS = [
    {"label": "V1 — Raw (no smooth, no filter)",     "gdp_window": 1, "min_duration": 1},
    {"label": "V2 — Smoothed only (no filter)",      "gdp_window": 3, "min_duration": 1},
    {"label": "V3 — Full methodology (smooth+filter)","gdp_window": 3, "min_duration": 3},
]

# ---------------------------------------------------------------------------
# Data fetching (no imports from src/)
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
    cpi_raw  = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("UNRATE...", end=" ", flush=True)
    unrate   = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("GDPC1...", end=" ", flush=True)
    gdpc1    = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), "cubic")
    print("NROU...", end=" ", flush=True)
    nrou     = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), "linear")
    print("GDPPOT...")
    gdppot   = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), "cubic")

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:].dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_regime(df: pd.DataFrame, gdp_window: int) -> pd.Series:
    """Assign raw (unfiltered) regime series."""
    gdp_gap   = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
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
    """Absorb spells shorter than min_duration into the preceding regime."""
    if min_duration <= 1:
        return regime  # no-op
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
# Transition counting
# ---------------------------------------------------------------------------

def count_transitions(regime: pd.Series) -> dict[tuple[str, str], list[str]]:
    """
    Return dict mapping (from_regime, to_regime) → list of date strings
    for every month-to-month regime change in the series.
    Only records actual transitions (regime[t] != regime[t-1]).
    """
    counts: dict[tuple[str, str], list[str]] = {}
    prev = None
    for date, val in regime.items():
        val_str = str(val)
        if prev is not None and val_str != prev:
            key = (prev, val_str)
            counts.setdefault(key, [])
            counts[key].append(date.strftime("%Y-%m"))
        prev = val_str
    return counts


def full_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """Raw count matrix (not normalised) for all regime pairs."""
    from_vals = regime.astype(str).iloc[:-1]
    to_vals   = regime.astype(str).iloc[1:]
    mask = from_vals.values != to_vals.values  # only actual changes
    mat = pd.crosstab(
        pd.Categorical(from_vals[mask], categories=REGIME_ORDER),
        pd.Categorical(to_vals[mask],   categories=REGIME_ORDER),
    )
    mat.index.name = "From"; mat.columns.name = "To"
    return mat


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def transition_label(from_r: str, to_r: str) -> str:
    return f"{SHORT[from_r]} → {SHORT[to_r]}  ({from_r} → {to_r})"


def build_report(df_base: pd.DataFrame) -> tuple[list[str], bool, bool]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 68
    dash = "-" * 68

    log(sep)
    log("  GMRI DIAGONAL TRANSITION AUDIT")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(sep)
    log()
    log("  2×2 Regime Matrix:")
    log("              │ Inflation ≤ 2.5%      │ Inflation > 2.5%")
    log("  ────────────┼───────────────────────┼────────────────────")
    log("  Growth ↑    │  Goldilocks   (GL)    │  Overheating  (OV)")
    log("  Growth ↓    │  Deflationary (DB)    │  Stagflation  (SG)")
    log()
    log("  Audited transitions:")
    for i, (fr, to) in enumerate(AUDIT_TRANSITIONS, 1):
        kind = "diagonal" if fr in ("Goldilocks", "Stagflation", "Deflationary Bust") \
               and (fr, to) in [("Goldilocks","Stagflation"),("Stagflation","Goldilocks"),
                                  ("Deflationary Bust","Overheating")] else "skip"
        log(f"    {i}. {transition_label(fr, to)}  [{kind}]")
    log()

    # --- Per-version results ---
    version_results: list[dict] = []

    for v in VERSIONS:
        log(dash)
        log(f"  {v['label']}")
        log(dash)

        regime = classify_regime(df_base, gdp_window=v["gdp_window"])
        regime = apply_min_duration_filter(regime, min_duration=v["min_duration"])

        transition_counts = count_transitions(regime)
        total_transitions = sum(len(dates) for dates in transition_counts.values())

        log(f"  Total regime changes in series: {total_transitions}")
        log()

        # Full transition count matrix
        mat = full_transition_matrix(regime)
        log("  Full transition count matrix (actual changes only, excludes self-stays):")
        col_header = f"  {'From → To':<22}" + "".join(f"  {SHORT[c]:>4}" for c in REGIME_ORDER)
        log(col_header)
        log("  " + "-" * 40)
        for from_r in REGIME_ORDER:
            row_str = f"  {from_r:<22}"
            for to_r in REGIME_ORDER:
                if from_r == to_r:
                    row_str += f"  {'—':>4}"
                else:
                    val = int(mat.loc[from_r, to_r]) if from_r in mat.index and to_r in mat.columns else 0
                    row_str += f"  {val:>4}"
            log(row_str)
        log()

        # Audited transitions
        audit_row: dict[tuple[str,str], int] = {}
        log("  Audited diagonal/skip transitions:")
        any_nonzero = False
        for fr, to in AUDIT_TRANSITIONS:
            dates = transition_counts.get((fr, to), [])
            count = len(dates)
            audit_row[(fr, to)] = count
            status = "ZERO ✓" if count == 0 else f"NON-ZERO ✗  [{', '.join(dates[:5])}{'…' if len(dates)>5 else ''}]"
            log(f"    {transition_label(fr, to):<50} count={count:>3}  {status}")
            if count > 0:
                any_nonzero = True

        version_results.append({
            "label": v["label"],
            "gdp_window": v["gdp_window"],
            "min_duration": v["min_duration"],
            "audit": audit_row,
            "any_nonzero": any_nonzero,
            "total_transitions": total_transitions,
        })
        log()

    # --- Summary comparison table ---
    log(sep)
    log(f"  {'SUMMARY TABLE — Audited Transition Counts':^66}")
    log(sep)
    log()

    # Header
    v_labels = ["V1 (raw)", "V2 (smooth)", "V3 (full)"]
    header = f"  {'Transition':<46}" + "".join(f"  {lbl:>11}" for lbl in v_labels)
    log(header)
    log("  " + "-" * (46 + 3 * 13))

    all_zero_v3 = True
    any_nonzero_v1 = False
    any_nonzero_v2 = False

    for fr, to in AUDIT_TRANSITIONS:
        row_str = f"  {transition_label(fr, to):<46}"
        for vr in version_results:
            c = vr["audit"][(fr, to)]
            cell = f"{c}" if c > 0 else "0 ✓"
            row_str += f"  {cell:>11}"
        log(row_str)

    log()

    for fr, to in AUDIT_TRANSITIONS:
        if version_results[0]["audit"][(fr, to)] > 0:
            any_nonzero_v1 = True
        if version_results[1]["audit"][(fr, to)] > 0:
            any_nonzero_v2 = True
        if version_results[2]["audit"][(fr, to)] > 0:
            all_zero_v3 = False

    # --- Conclusion ---
    log(sep)
    log(f"  {'CONCLUSION':^66}")
    log(sep)
    log()

    v3_zero  = all_zero_v3
    v1_clean = not any_nonzero_v1
    v2_clean = not any_nonzero_v2

    if v1_clean and v2_clean and v3_zero:
        verdict = (
            "HOLDS IN ALL THREE VERSIONS.\n"
            "  The zero-diagonal-transition property is structural — it is not an\n"
            "  artifact of smoothing or filtering. The classification boundaries\n"
            "  are stable enough that both signals never simultaneously flip in a\n"
            "  single month, regardless of preprocessing choices."
        )
    elif not v3_zero:
        verdict = (
            "FAILS ENTIRELY.\n"
            "  Diagonal transitions occur even in the full production methodology.\n"
            "  The zero-diagonal-transition claim cannot be made in the white paper\n"
            "  without qualification. Review the months flagged above."
        )
    elif (any_nonzero_v1 or any_nonzero_v2) and v3_zero:
        which = []
        if any_nonzero_v1:
            which.append("V1 (raw)")
        if any_nonzero_v2:
            which.append("V2 (smoothed only)")
        verdict = (
            f"HOLDS ONLY WITH FILTERING APPLIED.\n"
            f"  Diagonal transitions appear in {' and '.join(which)}, but are\n"
            f"  eliminated when the {3}-month minimum-duration filter is applied (V3).\n"
            f"  The claim depends on the filter — this should be disclosed in the\n"
            f"  white paper. Without filtering, the methodology produces implausible\n"
            f"  single-month diagonal jumps driven by data noise."
        )
    else:
        verdict = "INDETERMINATE — review individual version counts above."

    log(f"  The zero-diagonal-transition claim:")
    log(f"  {verdict}")
    log()

    # Add per-version zero/nonzero summary
    log("  Per-version status:")
    for vr, label in zip(version_results, v_labels):
        status = "ALL ZERO ✓" if not vr["any_nonzero"] else "HAS NON-ZERO ✗"
        log(f"    {label:<14}  {status}")
    log()
    log(sep)

    return lines, v3_zero, (any_nonzero_v1 or any_nonzero_v2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching data from FRED...")
    fred = get_fred_client()
    df_base = fetch_base_df(fred)
    print(f"  {len(df_base)} months: "
          f"{df_base.index[0].strftime('%Y-%m')} – {df_base.index[-1].strftime('%Y-%m')}\n")

    lines, v3_clean, pre_filter_dirty = build_report(df_base)

    out_path = Path(__file__).parent / "diagonal_audit.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
