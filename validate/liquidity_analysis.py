"""
GMRI Liquidity Adoption Analysis
==================================
Compares the macro regime swap instrument against three historical precedents
by measuring the number of years from instrument introduction to widespread
quoting. Uses FRED proxy data to anchor the "widespread quoting" milestones
and illustrate market maturity over time.

Comparable instruments and FRED proxies:
  - Inflation Swap     (first quoted 1997)  → T10YIE  (breakeven rate)
  - Variance Swap      (standardized 2003)  → VIXCLS  (VIX index)
  - GDP-Linked Warrant (issued by Greece 2012) → DGS10 (10-yr Treasury, context)

"Widespread quoting" is anchored by:
  - T10YIE first available: 2003-01-02  → inflation swap adoption gap = 6 years
  - Variance swap gap hardcoded at 3 years (2003 → 2006, broad OTC standardization)
  - GDP warrant: still niche as of 2026 (gap = 14+ years, incomplete)

Outputs:
  validate/adoption_timeline.png  — timeline + FRED proxy chart
  validate/liquidity_analysis.txt — plain-text assessment

Run from project root:  python validate/liquidity_analysis.py
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

# ---------------------------------------------------------------------------
# Reference dates (hardcoded constants as specified)
# ---------------------------------------------------------------------------
INFL_SWAP_FIRST_QUOTED  = 1997   # inflation swaps first quoted OTC
VAR_SWAP_STANDARDIZED   = 2003   # ISDA variance swap standardization
GDP_WARRANT_ISSUED      = 2012   # Greece GDP-linked warrants issued

# "Widespread quoting" milestones — anchored to FRED data / market evidence
INFL_SWAP_WIDESPREAD    = 2003   # T10YIE series begins; ISDA CPI swap templates
VAR_SWAP_WIDESPREAD     = 2006   # ~3 yrs post-standardization; exchange vol products
# GDP warrants: still illiquid — no widespread date assigned

CURRENT_YEAR = 2026              # GMRI swap introduced today

# Adoption gaps (years intro → widespread)
INFL_SWAP_GAP  = INFL_SWAP_WIDESPREAD - INFL_SWAP_FIRST_QUOTED   # 6
VAR_SWAP_GAP   = VAR_SWAP_WIDESPREAD  - VAR_SWAP_STANDARDIZED     # 3
GDP_WARRANT_GAP_SO_FAR = CURRENT_YEAR - GDP_WARRANT_ISSUED        # 14+ (incomplete)

# GMRI projected range (based on comparable medians and range)
GMRI_OPTIMISTIC   = CURRENT_YEAR + VAR_SWAP_GAP   # 2029 — variance swap comp
GMRI_BASE         = CURRENT_YEAR + round((INFL_SWAP_GAP + VAR_SWAP_GAP) / 2)  # 2031
GMRI_PESSIMISTIC  = CURRENT_YEAR + INFL_SWAP_GAP  # 2032 — inflation swap comp
GMRI_TAIL         = CURRENT_YEAR + GDP_WARRANT_GAP_SO_FAR  # 2040 — GDP warrant tail

# Chart colour palette
COLORS = {
    "Inflation Swap":       "#E91E63",   # pink-red
    "Variance Swap":        "#FF9800",   # orange
    "GDP-Linked Warrant":   "#9C27B0",   # purple
    "GMRI Regime Swap":     "#2196F3",   # blue
    "T10YIE":               "#E91E63",
    "VIXCLS":               "#FF9800",
    "DGS10":                "#9C27B0",
}

FRED_START = "1995-01-01"   # fetch window for all series

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


def monthly_coverage(s: pd.Series, start_year: int) -> pd.Series:
    """
    Rolling 12-month data availability rate (fraction of business days with
    a valid observation), computed as a monthly series from start_year onward.
    Returns a pd.Series indexed by month-end dates.
    """
    # Resample to business daily, mark observed vs missing
    daily = s.resample("B").last()
    daily_present = daily.notna().astype(float)

    # Restrict to start_year onward
    clip_start = f"{start_year}-01-01"
    daily_present = daily_present.loc[clip_start:]

    # Rolling 252-day coverage → convert to monthly (take last value each month)
    rolling = daily_present.rolling(252, min_periods=1).mean()
    monthly = rolling.resample("ME").last()
    return monthly


# ---------------------------------------------------------------------------
# Adoption gap computation
# ---------------------------------------------------------------------------

def compute_gaps(t10yie: pd.Series) -> dict:
    """
    Compute years-to-widespread for each comparable instrument.
    T10YIE first-available date is used as the empirical anchor for
    inflation swap widespread quoting (FRED data starts when the
    market is deep enough for reliable daily publication).
    """
    # Inflation swap: FRED T10YIE first observation date
    t10yie_start     = t10yie.index[0]
    infl_widespread_actual = t10yie_start.year + t10yie_start.month / 12

    gaps = {
        "Inflation Swap": {
            "intro":       INFL_SWAP_FIRST_QUOTED,
            "widespread":  INFL_SWAP_WIDESPREAD,
            "gap_years":   INFL_SWAP_GAP,
            "note":        f"T10YIE first available {t10yie_start.date()} (market deep enough "
                           f"for daily publication after {INFL_SWAP_GAP} years)",
        },
        "Variance Swap": {
            "intro":       VAR_SWAP_STANDARDIZED,
            "widespread":  VAR_SWAP_WIDESPREAD,
            "gap_years":   VAR_SWAP_GAP,
            "note":        "ISDA standardization → exchange-listed vol products (~3 yrs)",
        },
        "GDP-Linked Warrant": {
            "intro":       GDP_WARRANT_ISSUED,
            "widespread":  None,
            "gap_years":   None,
            "note":        f"Still illiquid as of {CURRENT_YEAR} "
                           f"({GDP_WARRANT_GAP_SO_FAR}+ years and counting)",
        },
    }
    return gaps


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_adoption_timeline(
    gaps: dict,
    t10yie: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series,
    save_path: str,
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.35)
    ax_gantt = fig.add_subplot(gs[0])
    ax_ts    = fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # Panel 1 — Gantt-style adoption timeline
    # ------------------------------------------------------------------
    instruments = [
        {
            "name":        "Inflation Swap",
            "intro":       INFL_SWAP_FIRST_QUOTED,
            "widespread":  INFL_SWAP_WIDESPREAD,
            "today":       CURRENT_YEAR,
            "color":       COLORS["Inflation Swap"],
        },
        {
            "name":        "Variance Swap",
            "intro":       VAR_SWAP_STANDARDIZED,
            "widespread":  VAR_SWAP_WIDESPREAD,
            "today":       CURRENT_YEAR,
            "color":       COLORS["Variance Swap"],
        },
        {
            "name":        "GDP-Linked Warrant",
            "intro":       GDP_WARRANT_ISSUED,
            "widespread":  None,
            "today":       CURRENT_YEAR,
            "color":       COLORS["GDP-Linked Warrant"],
        },
        {
            "name":        "GMRI Regime Swap",
            "intro":       CURRENT_YEAR,
            "widespread":  None,    # projected
            "today":       CURRENT_YEAR,
            "color":       COLORS["GMRI Regime Swap"],
        },
    ]

    y_positions = list(range(len(instruments) - 1, -1, -1))   # top to bottom
    bar_h = 0.35

    for y, inst in zip(y_positions, instruments):
        color  = inst["color"]
        intro  = inst["intro"]
        today  = inst["today"]

        if inst["name"] == "GMRI Regime Swap":
            # "Introduced today" marker + projected range
            ax_gantt.barh(y, 0.3, left=intro, height=bar_h,
                          color=color, alpha=0.9, zorder=3)
            # Optimistic → pessimistic shaded band
            ax_gantt.barh(y, GMRI_PESSIMISTIC - intro, left=intro, height=bar_h,
                          color=color, alpha=0.25, zorder=2,
                          label="_nolegend_")
            ax_gantt.barh(y, GMRI_OPTIMISTIC - intro, left=intro, height=bar_h,
                          color=color, alpha=0.45, zorder=2,
                          label="_nolegend_")
            # Tail risk arrow
            ax_gantt.annotate(
                "", xy=(GMRI_TAIL + 0.5, y),
                xytext=(GMRI_PESSIMISTIC, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
            )
            # Annotation text
            ax_gantt.text(
                GMRI_OPTIMISTIC + 0.2, y + 0.22,
                f"Optimistic: ~{GMRI_OPTIMISTIC}",
                fontsize=8, color=color, va="bottom",
            )
            ax_gantt.text(
                GMRI_PESSIMISTIC + 0.2, y - 0.22,
                f"Pessimistic: ~{GMRI_PESSIMISTIC}",
                fontsize=8, color=color, va="top",
            )

        elif inst["widespread"] is not None:
            # Pre-widespread (dark) bar: intro → widespread
            ax_gantt.barh(y, inst["widespread"] - intro, left=intro, height=bar_h,
                          color=color, alpha=0.9, zorder=3)
            # Post-widespread (lighter) bar: widespread → today
            ax_gantt.barh(y, today - inst["widespread"], left=inst["widespread"],
                          height=bar_h, color=color, alpha=0.3, zorder=2)
            # Gap annotation
            gap = inst["widespread"] - intro
            mid = intro + gap / 2
            ax_gantt.text(mid, y + bar_h / 2 + 0.05, f"{gap} yr",
                          ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                          color=color)
            # Widespread marker
            ax_gantt.axvline(inst["widespread"], color=color, linewidth=0.8,
                             linestyle=":", alpha=0.6)

        else:
            # No widespread date — still niche
            ax_gantt.barh(y, today - intro, left=intro, height=bar_h,
                          color=color, alpha=0.5, zorder=3)
            ax_gantt.text(today + 0.3, y,
                          f"  {today - intro}+ yrs, no widespread",
                          va="center", fontsize=8, color=color, style="italic")

    # Today vertical line
    ax_gantt.axvline(CURRENT_YEAR, color="black", linewidth=1.5, linestyle="--",
                     zorder=5, label=f"Today ({CURRENT_YEAR})")

    ax_gantt.set_yticks(y_positions)
    ax_gantt.set_yticklabels([i["name"] for i in instruments], fontsize=10)
    ax_gantt.set_xlim(1994, GMRI_TAIL + 3)
    ax_gantt.set_xlabel("Year", fontsize=10)
    ax_gantt.set_title(
        "Macro Instrument Adoption Timeline — Years to Widespread Quoting",
        fontsize=12, fontweight="bold",
    )
    ax_gantt.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax_gantt.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax_gantt.grid(axis="x", which="major", alpha=0.3)
    ax_gantt.spines["top"].set_visible(False)
    ax_gantt.spines["right"].set_visible(False)

    # Legend patches
    dark_patch  = mpatches.Patch(color="grey", alpha=0.85, label="Pre-widespread (adoption period)")
    light_patch = mpatches.Patch(color="grey", alpha=0.3,  label="Post-widespread (active market)")
    proj_patch  = mpatches.Patch(color=COLORS["GMRI Regime Swap"], alpha=0.35,
                                 label="GMRI projected adoption range")
    ax_gantt.legend(handles=[dark_patch, light_patch, proj_patch],
                    loc="lower right", fontsize=8, framealpha=0.8)

    # ------------------------------------------------------------------
    # Panel 2 — FRED proxy time series
    # ------------------------------------------------------------------
    ax2 = ax_ts.twinx()

    # T10YIE — breakeven inflation rate (left axis)
    t10y_m = to_month_end(t10yie).loc["2003-01-01":]
    ax_ts.plot(t10y_m.index, t10y_m.values, color=COLORS["T10YIE"],
               linewidth=1.2, alpha=0.85, label="T10YIE — Breakeven Inflation Rate (%)")

    # VIX — right axis
    vix_m = to_month_end(vix).loc["2003-01-01":]
    ax2.plot(vix_m.index, vix_m.values, color=COLORS["VIXCLS"],
             linewidth=1.0, alpha=0.7, label="VIXCLS — VIX Index (rhs)")

    # DGS10 — 10-yr Treasury (left axis, dashed)
    dgs10_m = to_month_end(dgs10).loc["2003-01-01":]
    ax_ts.plot(dgs10_m.index, dgs10_m.values, color=COLORS["DGS10"],
               linewidth=1.0, alpha=0.65, linestyle="--",
               label="DGS10 — 10-yr Treasury Yield (%)")

    # Adoption milestone vertical lines
    milestones = [
        (f"{VAR_SWAP_STANDARDIZED}",       f"Var swap\nstandardized\n{VAR_SWAP_STANDARDIZED}",
         COLORS["Variance Swap"]),
        (f"{VAR_SWAP_WIDESPREAD}",         f"Var swap\nwidespread\n{VAR_SWAP_WIDESPREAD}",
         COLORS["Variance Swap"]),
        (f"{GDP_WARRANT_ISSUED}",          f"GDP warrants\nissued\n{GDP_WARRANT_ISSUED}",
         COLORS["GDP-Linked Warrant"]),
    ]
    ymax = ax_ts.get_ylim()[1] if ax_ts.get_ylim()[1] > 0 else 10
    for year_str, label, color in milestones:
        dt = pd.Timestamp(f"{year_str}-01-01")
        ax_ts.axvline(dt, color=color, linewidth=1.0, linestyle=":", alpha=0.7)
        ax_ts.text(dt, ax_ts.get_ylim()[1] * 0.95 if ax_ts.get_ylim()[1] > 0 else 9,
                   label, fontsize=7, color=color, ha="left", va="top",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

    # Today line
    ax_ts.axvline(pd.Timestamp(f"{CURRENT_YEAR}-01-01"), color="black",
                  linewidth=1.4, linestyle="--", alpha=0.9)

    ax_ts.set_ylabel("Rate / Yield (%)", fontsize=10)
    ax2.set_ylabel("VIX Level", fontsize=10, color=COLORS["VIXCLS"])
    ax2.tick_params(axis="y", colors=COLORS["VIXCLS"])
    ax_ts.set_xlabel("Date", fontsize=10)
    ax_ts.set_title(
        "FRED Proxy Series — Market Depth Indicators Since Var-Swap Standardization (2003)",
        fontsize=11, fontweight="bold",
    )
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(3))
    ax_ts.grid(axis="x", alpha=0.3)
    ax_ts.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax_ts.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
                 loc="upper right", framealpha=0.85)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart: {save_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def build_report(
    gaps: dict,
    t10yie: pd.Series,
    vix: pd.Series,
    dgs10: pd.Series,
) -> list[str]:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    sep  = "=" * 68
    dash = "-" * 68

    log(sep)
    log("  GMRI LIQUIDITY ADOPTION ANALYSIS")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(sep)
    log()

    # --- FRED series summary ---
    log("  FRED Proxy Series:")
    log(f"  {'Series':<10}  {'Description':<35}  {'First Obs':>12}  {'Last Obs':>12}")
    log("  " + "-" * 72)
    for s, desc, data in [
        ("T10YIE", "10-yr Breakeven Inflation Rate (%)",  t10yie),
        ("VIXCLS", "CBOE Volatility Index (VIX)",         vix),
        ("DGS10",  "10-yr Treasury Constant Maturity (%)", dgs10),
    ]:
        log(f"  {s:<10}  {desc:<35}  {data.index[0].date()!s:>12}  "
            f"{data.index[-1].date()!s:>12}")
    log()

    # --- Reference dates ---
    log("  Reference Dates (hardcoded constants):")
    log(f"    Inflation swaps first quoted:           {INFL_SWAP_FIRST_QUOTED}")
    log(f"    Variance swaps standardized (ISDA):     {VAR_SWAP_STANDARDIZED}")
    log(f"    GDP-linked warrants issued (Greece):    {GDP_WARRANT_ISSUED}")
    log(f"    GMRI Regime Swap introduced:            {CURRENT_YEAR}")
    log()

    # --- Adoption gaps ---
    log(sep)
    log(f"  {'ADOPTION GAP ANALYSIS':^66}")
    log(sep)
    log()
    log(f"  {'Instrument':<24}  {'Intro':>5}  {'Widespread':>10}  {'Gap (yrs)':>9}  Status")
    log("  " + "-" * 68)

    t10yie_start_date = t10yie.index[0].date()
    rows = [
        ("Inflation Swap",      INFL_SWAP_FIRST_QUOTED, INFL_SWAP_WIDESPREAD, INFL_SWAP_GAP,
         f"Widespread — T10YIE daily quoting from {t10yie_start_date}"),
        ("Variance Swap",       VAR_SWAP_STANDARDIZED,  VAR_SWAP_WIDESPREAD,  VAR_SWAP_GAP,
         "Widespread — exchange vol products, VIX options, VSTOXX"),
        ("GDP-Linked Warrant",  GDP_WARRANT_ISSUED,     None, None,
         f"Still niche — {GDP_WARRANT_GAP_SO_FAR}+ years, no secondary market"),
        ("GMRI Regime Swap",    CURRENT_YEAR,           None, None,
         "Introduced today — pre-liquidity"),
    ]
    for name, intro, wide, gap, status in rows:
        gap_str  = f"{gap}" if gap is not None else "TBD"
        wide_str = f"{wide}" if wide is not None else "N/A"
        log(f"  {name:<24}  {intro:>5}  {wide_str:>10}  {gap_str:>9}  {status}")
    log()

    # --- FRED data coverage ---
    log(dash)
    log("  FRED Data Coverage — Empirical Market Depth Indicator")
    log(dash)
    log()
    log("  T10YIE (Inflation Swap proxy):")
    log(f"    First observation:  {t10yie.index[0].date()}")
    t10yie_coverage_2003 = t10yie.loc["2003-01-01":"2003-12-31"]
    log(f"    2003 observations:  {len(t10yie_coverage_2003)} (daily quoting established "
        f"immediately upon index launch)")
    log(f"    Interpretation:     T10YIE publication from Jan 2003 confirms the inflation "
        f"swap market had achieved sufficient depth for reliable daily breakeven extraction "
        f"— exactly {INFL_SWAP_GAP} years after the first OTC quotes.")
    log()
    log("  VIXCLS (Variance Swap proxy):")
    vix_2003 = vix.loc["2003-01-01":"2006-12-31"]
    vix_mean_03_06 = vix_2003.mean()
    vix_std_03_06  = vix_2003.std()
    log(f"    2003–2006 mean VIX: {vix_mean_03_06:.1f}  (std: {vix_std_03_06:.1f})")
    log(f"    Interpretation:     VIX continuity from 1990 predates ISDA variance swap "
        f"standardization (2003). Post-standardization, broker-dealer vol desks rapidly "
        f"adopted consistent variance swap pricing, with exchange-listed products "
        f"(VIX futures 2004, VIX options 2006) confirming widespread status by 2006.")
    log()
    log("  DGS10 (GDP Warrant context):")
    dgs10_2012 = dgs10.loc["2012-01-01":"2026-01-01"]
    log(f"    2012–2026 observations: {len(dgs10_2012)}")
    log(f"    Interpretation:     Greece GDP-linked warrants were structured with payoffs "
        f"tied to real GDP growth above 2%. Despite 14+ years of maturity, the instruments "
        f"remain illiquid — no standardized secondary market has emerged. DGS10 shown as a "
        f"benchmark for a fully mature comparable rate market.")
    log()

    # --- Time-to-liquidity projection ---
    log(sep)
    log(f"  {'ESTIMATED TIME-TO-LIQUIDITY FOR GMRI REGIME SWAP':^66}")
    log(sep)
    log()
    log("  Comparable adoption gaps (years intro → widespread quoting):")
    log(f"    Variance Swap (fastest):   {VAR_SWAP_GAP} years  (2003 → 2006)")
    log(f"    Inflation Swap (mid):      {INFL_SWAP_GAP} years  (1997 → 2003)")
    log(f"    GDP Warrant (slowest):     {GDP_WARRANT_GAP_SO_FAR}+ years  (2012 → still niche)")
    log()
    median_comp = round((VAR_SWAP_GAP + INFL_SWAP_GAP) / 2)
    log(f"  Median of completed comps (excl. GDP warrant):  {median_comp} years")
    log(f"  Range of completed comps:                       {VAR_SWAP_GAP}–{INFL_SWAP_GAP} years")
    log()
    log("  GMRI Regime Swap projections (from 2026 introduction):")
    log(f"    Optimistic  (variance swap comp):   ~{GMRI_OPTIMISTIC}"
        f"  ({VAR_SWAP_GAP} years)")
    log(f"    Base case   (median comp):          ~{GMRI_BASE}"
        f"  ({GMRI_BASE - CURRENT_YEAR} years)")
    log(f"    Pessimistic (inflation swap comp):  ~{GMRI_PESSIMISTIC}"
        f"  ({INFL_SWAP_GAP} years)")
    log(f"    Tail risk   (GDP warrant comp):     {GMRI_TAIL}+"
        f"  ({GDP_WARRANT_GAP_SO_FAR}+ years, or indefinite)")
    log()

    # --- Assessment ---
    log(sep)
    log(f"  {'ASSESSMENT':^66}")
    log(sep)
    log()
    log("  The two completed analogues bracket a 3–6 year adoption window:")
    log()
    log("  Factors favouring the faster (variance swap) scenario:")
    log("    • Digital regime classification is binary — no continuous valuation")
    log("      curve to bootstrap, unlike inflation or credit curves.")
    log("    • The GMRI classification algorithm is fully transparent and")
    log("      reproducible from public FRED data, reducing barrier to price-")
    log("      checking and dealer adoption.")
    log("    • Institutional demand for macro regime hedges (risk parity,")
    log("      CTA, multi-asset) is well-established.")
    log()
    log("  Factors favouring the slower (inflation swap) scenario:")
    log("    • No existing exchange-listed product to anchor daily settlement.")
    log("    • Requires standardized ISDA confirmation language for the GMRI")
    log("      index definition, growth signal, and regime classification rules.")
    log("    • Limited natural two-way flow until both long-inflation-regime")
    log("      and short-inflation-regime constituencies are identified.")
    log()
    log("  GDP-warrant tail risk:")
    log("    • If the instrument attracts no natural hedging constituency")
    log("      (i.e., corporates/sovereigns with P&L materially linked to")
    log("      regime duration), liquidity may never consolidate, as with")
    log("      the Greece GDP warrants that remain orphaned 14+ years after")
    log("      issuance.")
    log()
    log("  Conclusion:")
    log(f"    Base estimate: widespread quoting achievable by ~{GMRI_BASE}")
    log(f"    Confidence interval: {GMRI_OPTIMISTIC}–{GMRI_PESSIMISTIC}")
    log(f"    Preconditions: ISDA template, at least 2 dealer market-makers,")
    log( "    and index governance body publishing monthly regime determinations.")
    log()
    log(sep)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  GMRI LIQUIDITY ADOPTION ANALYSIS")
    print("=" * 60)

    print("\nFetching data from FRED...")
    fred = get_fred_client()

    print("  T10YIE...", end=" ", flush=True)
    t10yie = fetch(fred, "T10YIE", start=FRED_START)
    print("VIXCLS...", end=" ", flush=True)
    vix    = fetch(fred, "VIXCLS", start=FRED_START)
    print("DGS10...")
    dgs10  = fetch(fred, "DGS10",  start=FRED_START)

    print(f"  T10YIE: {t10yie.index[0].date()} → {t10yie.index[-1].date()} "
          f"({len(t10yie)} obs)")
    print(f"  VIXCLS: {vix.index[0].date()} → {vix.index[-1].date()} "
          f"({len(vix)} obs)")
    print(f"  DGS10:  {dgs10.index[0].date()} → {dgs10.index[-1].date()} "
          f"({len(dgs10)} obs)")

    # Compute adoption gaps
    gaps = compute_gaps(t10yie)

    print(f"\n  Adoption gaps:")
    for name, g in gaps.items():
        gap_str = f"{g['gap_years']} years" if g["gap_years"] is not None else "TBD (still niche)"
        print(f"    {name:<24}  intro={g['intro']}  widespread={g['widespread']}  gap={gap_str}")

    # Text report
    lines = build_report(gaps, t10yie, vix, dgs10)
    txt_path = Path(__file__).parent / "liquidity_analysis.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved report: {txt_path}")

    # Chart
    png_path = str(Path(__file__).parent / "adoption_timeline.png")
    plot_adoption_timeline(gaps, t10yie, vix, dgs10, png_path)


if __name__ == "__main__":
    main()
