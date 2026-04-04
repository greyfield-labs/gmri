"""
GMRI Historical Event Validation
==================================
Loads regime classifications from the production classifier and checks
assignments against six well-documented macroeconomic episodes.

Classifier parameters used (matching production main.py):
  CPI threshold:  2.5%
  GDP gap window: 3 months
  Growth logic:   AND (both signals required)
  Min duration:   3 months

Validation periods:
  1973-10 – 1975-03   Oil Shock              expected: Stagflation
  1981-07 – 1982-11   Volcker Recession      expected: Stagflation
  1995-01 – 2000-03   Tech Expansion         expected: Goldilocks or Overheating
  2008-09 – 2009-06   GFC                    expected: Deflationary Bust
  2020-02 – 2020-05   COVID Shock            expected: Deflationary Bust
  2021-03 – 2022-09   Inflation Surge        expected: Overheating or Stagflation

For each period the script reports:
  - Month-by-month regime assignments with signal values
  - Dominant regime (most frequent over the window)
  - Match/mismatch against expected regime set
  - Month-level mismatch flag with explanation of which signal diverged

Run from project root:  python validate/historical_validation.py
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Classifier parameters — must match production main.py exactly
# ---------------------------------------------------------------------------
START_DATE    = "1970-01-01"
CPI_THRESHOLD = 2.5
GDP_WINDOW    = 3
MIN_DURATION  = 3

REGIME_ORDER = ["Goldilocks", "Overheating", "Stagflation", "Deflationary Bust"]
SHORT = {
    "Goldilocks":        "GL",
    "Overheating":       "OV",
    "Stagflation":       "SG",
    "Deflationary Bust": "DB",
}

# ---------------------------------------------------------------------------
# Validation periods — (label, start, end, expected_set, economic_rationale)
# ---------------------------------------------------------------------------
PERIODS = [
    {
        "label":    "Oil Shock",
        "start":    "1973-10",
        "end":      "1975-03",
        "expected": {"Stagflation"},
        "rationale": (
            "OPEC embargo quadrupled oil prices; CPI surged above 10% while real GDP "
            "contracted. Textbook stagflation: supply shock drove both high inflation "
            "and below-trend growth simultaneously."
        ),
    },
    {
        "label":    "Volcker Recession",
        "start":    "1981-07",
        "end":      "1982-11",
        "expected": {"Stagflation"},
        "rationale": (
            "Fed Funds rate peaked near 20% to break the inflation spiral. "
            "Real GDP fell sharply (two-quarter contraction in 1982) while CPI "
            "remained above 5%. Growth below trend, inflation above threshold."
        ),
    },
    {
        "label":    "Tech Expansion",
        "start":    "1995-01",
        "end":      "2000-03",
        "expected": {"Goldilocks", "Overheating"},
        "rationale": (
            "Productivity boom drove GDP well above potential. CPI was subdued "
            "early (Goldilocks), then crept toward and above 2.5% late in the "
            "cycle as labour markets tightened (Overheating). Both are acceptable."
        ),
    },
    {
        "label":    "GFC",
        "start":    "2008-09",
        "end":      "2009-06",
        "expected": {"Deflationary Bust"},
        "rationale": (
            "Lehman collapse triggered the deepest post-war recession. Real GDP "
            "fell 4% peak-to-trough; CPI went briefly negative in mid-2009 on "
            "collapsing commodity prices. Growth below trend, inflation below threshold."
        ),
    },
    {
        "label":    "COVID Shock",
        "start":    "2020-02",
        "end":      "2020-05",
        "expected": {"Deflationary Bust"},
        "rationale": (
            "Lockdowns caused the sharpest GDP collapse on record (-31.4% annualised "
            "Q2 2020). CPI briefly turned negative as oil prices crashed. "
            "Growth massively below trend, inflation below threshold."
        ),
    },
    {
        "label":    "Inflation Surge",
        "start":    "2021-03",
        "end":      "2022-09",
        "expected": {"Overheating", "Stagflation"},
        "rationale": (
            "Post-COVID fiscal stimulus and supply-chain disruptions pushed CPI to "
            "9.1% by mid-2022. Early in the window GDP gap was positive (Overheating); "
            "by late 2022 growth was flagging as the Fed hiked (Stagflation). Both acceptable."
        ),
    },
]

# ---------------------------------------------------------------------------
# Data fetching (independent — no src/ imports)
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


def build_classified_df(fred: Fred) -> pd.DataFrame:
    """Fetch, preprocess, and classify. Returns full DataFrame with all signal columns."""
    print("  CPIAUCSL...", end=" ", flush=True)
    cpi_raw  = to_month_end(fetch(fred, "CPIAUCSL", start=START_DATE))
    print("UNRATE...",   end=" ", flush=True)
    unrate   = to_month_end(fetch(fred, "UNRATE",   start=START_DATE))
    print("GDPC1...",    end=" ", flush=True)
    gdpc1    = interp_quarterly(to_month_end(fetch(fred, "GDPC1",  start="1947-01-01")), "cubic")
    print("NROU...",     end=" ", flush=True)
    nrou     = interp_quarterly(to_month_end(fetch(fred, "NROU",   start="1947-01-01")), "linear")
    print("GDPPOT...")
    gdppot   = interp_quarterly(to_month_end(fetch(fred, "GDPPOT", start="1947-01-01")), "cubic")

    gdpc1.name = "GDPC1"; nrou.name = "NROU"; gdppot.name = "GDPPOT"
    cpi_yoy = (cpi_raw.pct_change(12) * 100).dropna()
    cpi_yoy.name = "CPI_YOY"

    df = pd.concat([cpi_yoy, unrate, gdpc1, nrou, gdppot], axis=1, join="inner")
    df = df.loc[START_DATE:].dropna(subset=["CPI_YOY", "UNRATE", "GDPC1", "NROU", "GDPPOT"])

    # --- Derived signals ---
    df["GDP_GAP"]       = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    df["GDP_GAP_SMOOTH"]= df["GDP_GAP"].rolling(window=GDP_WINDOW, min_periods=1).mean()
    df["UNEMP_GAP"]     = df["UNRATE"] - df["NROU"]
    df["GROWTH_ABOVE"]  = (df["GDP_GAP_SMOOTH"] > 0) & (df["UNEMP_GAP"] < 0)
    df["INFL_ABOVE"]    = df["CPI_YOY"] > CPI_THRESHOLD

    # --- Raw regime ---
    conditions = [
        df["GROWTH_ABOVE"] & ~df["INFL_ABOVE"],
        df["GROWTH_ABOVE"] &  df["INFL_ABOVE"],
       ~df["GROWTH_ABOVE"] &  df["INFL_ABOVE"],
       ~df["GROWTH_ABOVE"] & ~df["INFL_ABOVE"],
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")
    df["REGIME_RAW"] = pd.Categorical(raw, categories=REGIME_ORDER)

    # --- Filtered regime ---
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
    df["REGIME"] = pd.Categorical(filtered, categories=REGIME_ORDER)

    return df


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def signal_diagnosis(row: pd.Series, expected_set: set[str]) -> str:
    """
    Given a row with signal columns and an expected regime set, return a
    short explanation of which signal(s) caused a mismatch.
    """
    growth_word  = "above-trend" if row["GROWTH_ABOVE"]  else "below-trend"
    infl_word    = "above-threshold" if row["INFL_ABOVE"] else "below-threshold"
    actual       = str(row["REGIME"])

    reasons = []
    # For each expected regime, check which signals disagree
    for exp in expected_set:
        exp_growth = exp in ("Goldilocks", "Overheating")
        exp_infl   = exp in ("Overheating", "Stagflation")
        parts = []
        if exp_growth != bool(row["GROWTH_ABOVE"]):
            parts.append(
                f"growth is {growth_word} "
                f"(GDP gap={row['GDP_GAP_SMOOTH']:+.2f}%, "
                f"unemp gap={row['UNEMP_GAP']:+.2f}pp)"
            )
        if exp_infl != bool(row["INFL_ABOVE"]):
            parts.append(f"CPI YoY={row['CPI_YOY']:.1f}% "
                         f"({'above' if row['INFL_ABOVE'] else 'below'} {CPI_THRESHOLD}%)")
        if parts:
            reasons.append("; ".join(parts))

    return " | ".join(reasons) if reasons else f"actual={actual}"


def validate_period(df: pd.DataFrame, period: dict) -> dict:
    """
    Validate one period. Returns a results dict with per-month detail.
    Uses month-end dates for slicing (YYYY-MM → last day of that month).
    """
    start = pd.Timestamp(period["start"]) + pd.offsets.MonthEnd(0)
    end   = pd.Timestamp(period["end"])   + pd.offsets.MonthEnd(0)

    window = df.loc[start:end].copy()
    if window.empty:
        return {"error": f"No data in range {period['start']}–{period['end']}"}

    expected = period["expected"]
    n_months = len(window)

    # Dominant regime
    regime_counts = window["REGIME"].value_counts()
    dominant      = str(regime_counts.index[0])
    dominant_pct  = regime_counts.iloc[0] / n_months * 100

    # Per-month match flags
    window = window.copy()
    window["MATCH"] = window["REGIME"].astype(str).isin(expected)

    n_match    = int(window["MATCH"].sum())
    n_mismatch = n_months - n_match
    match_pct  = n_match / n_months * 100

    # Month-level mismatch detail
    mismatches = []
    for date, row in window[~window["MATCH"]].iterrows():
        mismatches.append({
            "date":       date.strftime("%Y-%m"),
            "actual":     str(row["REGIME"]),
            "cpi_yoy":    round(row["CPI_YOY"],  2),
            "gdp_gap_sm": round(row["GDP_GAP_SMOOTH"], 3),
            "unemp_gap":  round(row["UNEMP_GAP"], 2),
            "growth_above": bool(row["GROWTH_ABOVE"]),
            "infl_above":   bool(row["INFL_ABOVE"]),
            "diagnosis":  signal_diagnosis(row, expected),
        })

    # Full per-month table for detail section
    monthly_detail = []
    for date, row in window.iterrows():
        monthly_detail.append({
            "date":       date.strftime("%Y-%m"),
            "regime":     str(row["REGIME"]),
            "cpi_yoy":    round(row["CPI_YOY"],  2),
            "gdp_gap_sm": round(row["GDP_GAP_SMOOTH"], 3),
            "unemp_gap":  round(row["UNEMP_GAP"], 2),
            "growth_above": bool(row["GROWTH_ABOVE"]),
            "infl_above":   bool(row["INFL_ABOVE"]),
            "match":        bool(row["MATCH"]),
        })

    # Period-level verdict
    dominant_match = dominant in expected
    overall_pass   = match_pct >= 75.0   # ≥75% of months in expected regime → PASS

    return {
        "label":          period["label"],
        "start":          period["start"],
        "end":            period["end"],
        "expected":       expected,
        "n_months":       n_months,
        "dominant":       dominant,
        "dominant_pct":   round(dominant_pct, 1),
        "dominant_match": dominant_match,
        "n_match":        n_match,
        "n_mismatch":     n_mismatch,
        "match_pct":      round(match_pct, 1),
        "overall_pass":   overall_pass,
        "mismatches":     mismatches,
        "monthly_detail": monthly_detail,
        "rationale":      period["rationale"],
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(results: list[dict]) -> list[str]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 72
    dash = "-" * 72

    log(sep)
    log("  GMRI HISTORICAL EVENT VALIDATION")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Parameters: CPI threshold={CPI_THRESHOLD}%  |  "
        f"GDP window={GDP_WINDOW}mo  |  Min duration={MIN_DURATION}mo  |  "
        f"Growth logic=AND")
    log(sep)
    log()

    # -----------------------------------------------------------------------
    # Per-period detail
    # -----------------------------------------------------------------------
    for r in results:
        if "error" in r:
            log(f"  {r['label']}: ERROR — {r['error']}")
            continue

        pass_str    = "PASS ✓" if r["overall_pass"] else "FAIL ✗"
        exp_str     = " or ".join(sorted(r["expected"]))
        dominant_ok = "✓" if r["dominant_match"] else "✗"

        log(dash)
        log(f"  {r['label']}  ({r['start']} – {r['end']})    [{pass_str}]")
        log(dash)
        log(f"  Economic context:")
        # Wrap rationale at 68 chars
        words = r["rationale"].split()
        line_buf = "    "
        for word in words:
            if len(line_buf) + len(word) + 1 > 70:
                log(line_buf)
                line_buf = "    " + word
            else:
                line_buf += (" " if line_buf != "    " else "") + word
        if line_buf.strip():
            log(line_buf)
        log()
        log(f"  Expected regime(s):  {exp_str}")
        log(f"  Dominant regime:     {r['dominant']} ({r['dominant_pct']:.0f}% of months)  {dominant_ok}")
        log(f"  Months in period:    {r['n_months']}")
        log(f"  Months matching:     {r['n_match']} / {r['n_months']} "
            f"({r['match_pct']:.0f}%)  ← ≥75% required to pass")

        # Month-by-month table
        log()
        log("  Month-by-month assignments:")
        hdr = f"  {'Date':<9} {'Regime':<18} {'CPI YoY':>8} {'GDP gap':>8} {'U gap':>7}  {'Match'}"
        log(hdr)
        log("  " + "-" * 64)
        for m in r["monthly_detail"]:
            flag    = "✓" if m["match"] else "✗"
            gr_mark = "↑" if m["growth_above"] else "↓"
            in_mark = "↑" if m["infl_above"]   else "↓"
            log(
                f"  {m['date']:<9} "
                f"{m['regime']:<18} "
                f"{m['cpi_yoy']:>7.1f}% "
                f"{m['gdp_gap_sm']:>+7.2f}% "
                f"{m['unemp_gap']:>+6.2f}pp "
                f" {flag}  G{gr_mark} I{in_mark}"
            )

        # Mismatch detail
        if r["mismatches"]:
            log()
            log(f"  Mismatch detail ({r['n_mismatch']} month(s)):")
            for mm in r["mismatches"]:
                log(f"    {mm['date']}  actual={mm['actual']:<18}  {mm['diagnosis']}")
        log()

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    log(sep)
    log(f"  {'SUMMARY TABLE':^70}")
    log(sep)
    log()

    col_period   = 22
    col_expected = 26
    col_dominant = 20
    col_match    = 10
    col_verdict  = 8

    hdr = (
        f"  {'Period':<{col_period}}"
        f"  {'Expected':<{col_expected}}"
        f"  {'Dominant':<{col_dominant}}"
        f"  {'Match %':>{col_match}}"
        f"  {'Verdict':>{col_verdict}}"
    )
    log(hdr)
    log("  " + "-" * (col_period + col_expected + col_dominant + col_match + col_verdict + 10))

    n_pass = 0
    for r in results:
        if "error" in r:
            log(f"  {r['label']:<{col_period}}  ERROR")
            continue
        verdict = "PASS ✓" if r["overall_pass"] else "FAIL ✗"
        exp_str = " or ".join(sorted(r["expected"]))
        dominant_ok = "✓" if r["dominant_match"] else "✗"
        log(
            f"  {r['label']:<{col_period}}"
            f"  {exp_str:<{col_expected}}"
            f"  {r['dominant'] + ' ' + dominant_ok:<{col_dominant}}"
            f"  {r['match_pct']:>{col_match}.0f}%"
            f"  {verdict:>{col_verdict}}"
        )
        if r["overall_pass"]:
            n_pass += 1

    log()
    total = sum(1 for r in results if "error" not in r)
    log(f"  Overall: {n_pass}/{total} periods pass (dominant regime in expected set "
        f"and ≥75% months match)")
    log()

    # -----------------------------------------------------------------------
    # Methodology note
    # -----------------------------------------------------------------------
    log(sep)
    log(f"  {'METHODOLOGY NOTES':^70}")
    log(sep)
    log()
    log("  Signal legend in month-by-month tables:")
    log("    G↑ = growth above trend  (GDP gap > 0 AND unemp gap < 0)")
    log("    G↓ = growth below trend  (GDP gap ≤ 0 OR unemp gap ≥ 0)")
    log("    I↑ = inflation above threshold  (CPI YoY > 2.5%)")
    log("    I↓ = inflation below threshold  (CPI YoY ≤ 2.5%)")
    log()
    log("  Regime mapping:")
    log("    G↑ + I↓ = Goldilocks        G↑ + I↑ = Overheating")
    log("    G↓ + I↑ = Stagflation       G↓ + I↓ = Deflationary Bust")
    log()
    log("  Pass criterion: dominant regime ∈ expected set AND ≥75% of months")
    log("  in expected set. The 75% threshold allows for boundary months at")
    log("  the start/end of an episode without failing the whole period.")
    log()
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching and classifying data from FRED...")
    fred = get_fred_client()
    df = build_classified_df(fred)
    print(f"  Classified: {df.index[0].strftime('%Y-%m')} – "
          f"{df.index[-1].strftime('%Y-%m')}  ({len(df)} months)\n")

    print("Validating historical periods...")
    results = [validate_period(df, p) for p in PERIODS]

    lines = build_report(results)

    out_path = Path(__file__).parent / "historical_validation.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
