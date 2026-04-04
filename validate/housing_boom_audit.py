"""
GMRI Housing Boom Audit (2003–2007)
=====================================
Defends or qualifies the paper's claim that the Post-GFC era (2001–2019)
is 53% Deflationary Bust.  The most contestable sub-period is the 2003–2007
housing boom, which many economists would characterise as Overheating or
Goldilocks, not Deflationary Bust.

This script:
  1. Fetches all five FRED series and runs the full GMRI pipeline.
  2. Saves validate/regime_classification.csv — the canonical monthly
     regime series used by reproducibility_check.py.
  3. Filters to January 2003 – December 2007 (60 months).
  4. Prints a month-by-month table with regime, CPI YoY, GDP gap (smoothed),
     and unemployment gap.
  5. Cross-checks the AND growth logic: counts months where GDP gap > 0
     but unemployment gap ≥ 0 — the condition that forces a below-trend
     growth signal despite positive GDP gap.
  6. Prints a plain-language conclusion.

Output: validate/housing_boom_audit.txt
        validate/regime_classification.csv  (full sample, used by repro script)

Run from project root:  python validate/housing_boom_audit.py
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
AUDIT_START   = "2003-01-01"
AUDIT_END     = "2007-12-31"
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
# FRED helpers
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
    print("  Fetching CPIAUCSL...", end=" ", flush=True)
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
    return df.loc[START_DATE:].dropna()


# ---------------------------------------------------------------------------
# Classification (verbatim from other validate scripts)
# ---------------------------------------------------------------------------

def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add GDP_GAP_RAW, GDP_GAP_SMOOTH, UNEMP_GAP columns."""
    df = df.copy()
    df["GDP_GAP_RAW"]    = (df["GDPC1"] - df["GDPPOT"]) / df["GDPPOT"] * 100
    df["GDP_GAP_SMOOTH"] = df["GDP_GAP_RAW"].rolling(window=GDP_WINDOW, min_periods=1).mean()
    df["UNEMP_GAP"]      = df["UNRATE"] - df["NROU"]
    return df


def classify_regime(df: pd.DataFrame) -> pd.Series:
    growth = (df["GDP_GAP_SMOOTH"] > 0) & (df["UNEMP_GAP"] < 0)
    infl   = df["CPI_YOY"] > CPI_THRESHOLD
    conditions = [
        growth & ~infl,
        growth &  infl,
       ~growth &  infl,
       ~growth & ~infl,
    ]
    raw = np.select(conditions, REGIME_ORDER, default="Unknown")
    return pd.Series(
        pd.Categorical(raw, categories=REGIME_ORDER),
        index=df.index, name="REGIME",
    )


def apply_min_duration_filter(regime: pd.Series, min_duration: int = MIN_DURATION) -> pd.Series:
    if min_duration <= 1:
        return regime
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


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, regime: pd.Series) -> list[str]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 80
    dash = "-" * 80

    log(sep)
    log("  GMRI HOUSING BOOM AUDIT — 2003–2007")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(sep)
    log()
    log("  Claim under review:")
    log("    The white paper states the Post-GFC era (2001–2019) is 53%")
    log("    Deflationary Bust.  The 2003–2007 housing boom is the most")
    log("    contestable sub-period: many economists would characterise it as")
    log("    Overheating or Goldilocks, not Deflationary Bust.")
    log()
    log("  GMRI growth signal (AND logic):")
    log("    growth_above = (GDP gap smooth > 0) AND (unemployment gap < 0)")
    log("    This requires BOTH a positive output gap AND employment above trend.")
    log()

    # Filter to audit window
    mask  = (df.index >= AUDIT_START) & (df.index <= AUDIT_END)
    df_w  = df.loc[mask].copy()
    reg_w = regime.loc[mask]

    n_months = len(df_w)
    log(f"  Audit window: {AUDIT_START[:7]} – {AUDIT_END[:7]}  ({n_months} months)")
    log()

    # -----------------------------------------------------------------------
    # Month-by-month table
    # -----------------------------------------------------------------------
    log(dash)
    log(f"  {'MONTH-BY-MONTH DETAIL':^78}")
    log(dash)
    log()
    log(f"  {'Month':<8}  {'Regime':<18}  {'CPI YoY':>8}  "
        f"{'GDP gap':>9}  {'Unemp gap':>10}  GDP>0  U<0  Logic")
    log("  " + "-" * 72)

    for date, row in df_w.iterrows():
        r         = str(reg_w.loc[date])
        cpi       = row["CPI_YOY"]
        gdp_s     = row["GDP_GAP_SMOOTH"]
        unemp     = row["UNEMP_GAP"]
        gdp_pos   = "Y" if gdp_s > 0 else "N"
        unemp_neg = "Y" if unemp < 0 else "N"

        # AND logic outcome
        if gdp_s > 0 and unemp < 0:
            logic = "ABOVE-TREND (both)"
        elif gdp_s > 0 and unemp >= 0:
            logic = "BELOW-TREND (U gap blocks)"
        elif gdp_s <= 0 and unemp < 0:
            logic = "BELOW-TREND (GDP gap blocks)"
        else:
            logic = "BELOW-TREND (both)"

        log(f"  {date.strftime('%Y-%m'):<8}  {r:<18}  {cpi:>7.2f}%  "
            f"{gdp_s:>+8.2f}%  {unemp:>+9.2f}pp  {gdp_pos:<5}  {unemp_neg:<4}  {logic}")
    log()

    # -----------------------------------------------------------------------
    # Regime breakdown
    # -----------------------------------------------------------------------
    log(dash)
    log(f"  {'REGIME BREAKDOWN — 2003–2007':^78}")
    log(dash)
    log()
    log(f"  {'Regime':<22}  {'Months':>6}  {'% of window':>12}")
    log("  " + "-" * 44)
    for r in REGIME_ORDER:
        n   = (reg_w == r).sum()
        pct = n / n_months * 100
        log(f"  {r:<22}  {n:>6}  {pct:>11.1f}%")
    log()

    # Summary stats
    avg_cpi   = df_w["CPI_YOY"].mean()
    avg_gdp   = df_w["GDP_GAP_SMOOTH"].mean()
    avg_unemp = df_w["UNEMP_GAP"].mean()
    log(f"  Average CPI YoY (2003–2007):         {avg_cpi:>+7.2f}%")
    log(f"  Average GDP gap smooth (2003–2007):  {avg_gdp:>+7.2f}%")
    log(f"  Average unemployment gap (2003–2007):{avg_unemp:>+7.2f}pp")
    log()

    # -----------------------------------------------------------------------
    # AND logic cross-check
    # -----------------------------------------------------------------------
    log(dash)
    log(f"  {'AND LOGIC CROSS-CHECK':^78}")
    log(dash)
    log()
    log("  The key question: in how many months did GDP gap > 0 but the AND")
    log("  logic still classified growth as BELOW-TREND because unemployment")
    log("  remained above NAIRU (unemployment gap ≥ 0)?")
    log()

    gdp_pos_mask   = df_w["GDP_GAP_SMOOTH"] > 0
    unemp_pos_mask = df_w["UNEMP_GAP"] >= 0    # gap ≥ 0 → unemployment above NAIRU
    unemp_neg_mask = df_w["UNEMP_GAP"] < 0

    # Four quadrants of AND logic during audit window
    q_both_above    = (gdp_pos_mask & unemp_neg_mask).sum()   # true above-trend
    q_gdp_blocked   = (gdp_pos_mask & unemp_pos_mask).sum()   # unemployment blocks growth
    q_unemp_blocked = (~gdp_pos_mask & unemp_neg_mask).sum()  # GDP gap blocks growth
    q_both_below    = (~gdp_pos_mask & unemp_pos_mask).sum()  # both below trend

    log(f"  {'Signal quadrant':<45}  {'Months':>6}  {'%':>6}")
    log("  " + "-" * 62)
    log(f"  {'GDP gap > 0  AND  unemployment gap < 0  (both above)':<45}  "
        f"{q_both_above:>6}  {q_both_above/n_months*100:>5.1f}%")
    log(f"  {'GDP gap > 0  BUT  unemployment gap ≥ 0  (U gap blocks)':<45}  "
        f"{q_gdp_blocked:>6}  {q_gdp_blocked/n_months*100:>5.1f}%")
    log(f"  {'GDP gap ≤ 0  AND  unemployment gap < 0  (GDP gap blocks)':<45}  "
        f"{q_unemp_blocked:>6}  {q_unemp_blocked/n_months*100:>5.1f}%")
    log(f"  {'GDP gap ≤ 0  AND  unemployment gap ≥ 0  (both below)':<45}  "
        f"{q_both_below:>6}  {q_both_below/n_months*100:>5.1f}%")
    log()

    # List the months where unemployment gap blocks an otherwise-positive GDP signal
    if q_gdp_blocked > 0:
        blocked_months = df_w.loc[gdp_pos_mask & unemp_pos_mask]
        log(f"  Months where GDP gap > 0 but unemployment gap ≥ 0")
        log(f"  (AND logic forces below-trend classification):")
        log()
        log(f"  {'Month':<8}  {'Regime assigned':<18}  {'GDP gap':>9}  "
            f"{'Unemp gap':>10}  {'CPI YoY':>8}")
        log("  " + "-" * 60)
        for date, row in blocked_months.iterrows():
            r = str(reg_w.loc[date])
            log(f"  {date.strftime('%Y-%m'):<8}  {r:<18}  "
                f"{row['GDP_GAP_SMOOTH']:>+8.2f}%  "
                f"{row['UNEMP_GAP']:>+9.2f}pp  "
                f"{row['CPI_YOY']:>7.2f}%")
        log()
        log(f"  Summary: {q_gdp_blocked} of {n_months} months ({q_gdp_blocked/n_months*100:.1f}%) "
            f"had a positive GDP gap that was overridden")
        log(f"  by an above-NAIRU unemployment gap under the AND logic.")
    else:
        log("  No months had GDP gap > 0 blocked by unemployment gap ≥ 0.")
    log()

    # -----------------------------------------------------------------------
    # Conclusion
    # -----------------------------------------------------------------------
    log(sep)
    log(f"  {'CONCLUSION':^78}")
    log(sep)
    log()

    db_months  = (reg_w == "Deflationary Bust").sum()
    gl_months  = (reg_w == "Goldilocks").sum()
    ov_months  = (reg_w == "Overheating").sum()
    sg_months  = (reg_w == "Stagflation").sum()
    db_pct     = db_months / n_months * 100
    ov_gl_pct  = (ov_months + gl_months) / n_months * 100

    log(f"  The 2003–2007 window is {db_pct:.0f}% Deflationary Bust and "
        f"{ov_gl_pct:.0f}% Overheating/Goldilocks.")
    log()

    if q_gdp_blocked >= 20:
        # Unemployment clearly blocking throughout
        log("  FINDING: The Deflationary Bust classification during 2003–2007 is")
        log("  principally explained by the unemployment gap remaining ABOVE NAIRU")
        log(f"  throughout most of the period, despite a positive output gap.")
        log()
        log(f"  In {q_gdp_blocked} of {n_months} months ({q_gdp_blocked/n_months*100:.0f}%), the GDP gap was positive")
        log("  but the AND logic correctly required BOTH signals to confirm above-trend")
        log("  growth. Unemployment did not fall below the natural rate until the")
        log("  late stages of the expansion.")
        log()
        log("  DEFENSIBILITY: HIGH.")
        log("  This is a legitimate AND-logic outcome, not a misclassification.")
        log("  The housing boom raised GDP above potential but did not immediately")
        log("  translate into labour market tightness — a pattern consistent with")
        log("  jobless-recovery dynamics from the 2001 recession.")
        log()
        log("  RECOMMENDED WHITE PAPER LANGUAGE:")
        log("    'The 2003–2007 housing expansion produced a positive output gap")
        log("    (average GDP gap: {:.2f}%) but unemployment remained above NAIRU")
        log("    (average unemployment gap: {:.2f}pp) through most of the period.")
        log("    The GMRI AND logic therefore classified the majority of this window")
        log("    as Deflationary Bust — reflecting an economy growing above potential")
        log("    output but not yet at full employment. This is consistent with the")
        log("    2001–2003 jobless recovery persisting longer than the GDP signal")
        log("    alone would suggest.'".format(avg_gdp, avg_unemp))
    elif q_gdp_blocked >= 10:
        log("  FINDING: A significant minority of 2003–2007 months show a positive")
        log("  GDP gap blocked by an above-NAIRU unemployment gap. The Deflationary")
        log("  Bust classification is partly explained by AND-logic conservatism and")
        log("  partly reflects months where both signals genuinely pointed below trend.")
        log()
        log("  DEFENSIBILITY: MEDIUM.")
        log("  Qualify in the white paper: acknowledge the GDP-unemployment signal")
        log("  conflict during this sub-period and present the sensitivity analysis")
        log("  showing how a GDP-gap-only growth signal would reclassify these months.")
    elif db_pct >= 30 and q_gdp_blocked < 5:
        log("  FINDING: Deflationary Bust classification during 2003–2007 is NOT")
        log("  primarily driven by the unemployment gap blocking a positive GDP signal.")
        log("  The GDP gap itself was negative or near-zero for much of this window,")
        log("  suggesting the economy was genuinely operating below potential despite")
        log("  house price appreciation — consistent with the 2001 recession leaving")
        log("  a persistent output gap that was only partially closed by 2007.")
        log()
        log("  DEFENSIBILITY: MEDIUM-HIGH.")
        log("  The Deflationary Bust is explained by output gap dynamics (GDP below")
        log("  potential), not by AND-logic stringency. However, reviewers may still")
        log("  contest this given contemporaneous perceptions of the boom.")
    else:
        log("  FINDING: The 2003–2007 window is not predominantly Deflationary Bust.")
        log("  Most months are classified as Overheating or Goldilocks, consistent with")
        log("  the conventional reading of the housing boom as an expansionary period.")
        log("  The paper's Post-GFC era Deflationary Bust percentage is driven by")
        log("  other sub-periods within 2001–2019 (likely 2001–2002 and 2008–2009).")
    log()
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  GMRI HOUSING BOOM AUDIT")
    print("=" * 60)

    print("\nFetching data from FRED...")
    fred = get_fred_client()
    df   = fetch_base_df(fred)
    print(f"  {len(df)} months: {df.index[0].strftime('%Y-%m')} – "
          f"{df.index[-1].strftime('%Y-%m')}")

    # Build signals and classify
    df     = build_signals(df)
    regime = classify_regime(df)
    regime = apply_min_duration_filter(regime)
    print(f"  Regime series: {len(regime)} months\n")

    # Save canonical regime CSV (used by reproducibility_check.py)
    csv_path = Path(__file__).parent / "regime_classification.csv"
    regime_df = pd.DataFrame({
        "date":          df.index.strftime("%Y-%m-%d"),
        "REGIME":        regime.astype(str),
        "CPI_YOY":       df["CPI_YOY"].round(4),
        "GDP_GAP_SMOOTH": df["GDP_GAP_SMOOTH"].round(4),
        "UNEMP_GAP":     df["UNEMP_GAP"].round(4),
    })
    regime_df.to_csv(csv_path, index=False)
    print(f"Saved production regime CSV: {csv_path}")
    print(f"  ({len(regime_df)} rows × {len(regime_df.columns)} columns)\n")

    # Build and save audit report
    lines = build_report(df, regime)
    txt_path = Path(__file__).parent / "housing_boom_audit.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {txt_path}")


if __name__ == "__main__":
    main()
