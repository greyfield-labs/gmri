import os
from pathlib import Path

from .data import get_fred_client, build_master_dataframe
from .classifier import (
    classify,
    compute_regime_stats,
    compute_transition_matrix,
    print_signal_conflict_report,
)
from .charts import (
    plot_regime_timeline,
    plot_regime_timeline_v2,
    plot_transition_matrix,
    plot_signal_chart,
    print_stats_table,
)
from .robustness import (
    threshold_sensitivity_analysis,
    plot_threshold_sensitivity,
    print_threshold_sensitivity,
    era_breakdown,
    print_era_breakdown,
    plot_era_breakdown,
    print_filtered_vs_unfiltered,
)


OUTPUTS = Path(__file__).parent.parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def _out(filename: str) -> str:
    return str(OUTPUTS / filename)


def main() -> None:
    print("Connecting to FRED...")
    fred = get_fred_client()

    print("Fetching data from FRED...")
    df_raw = build_master_dataframe(fred)
    print(f"  Data range: {df_raw.index[0].strftime('%Y-%m')} to {df_raw.index[-1].strftime('%Y-%m')} ({len(df_raw)} months)")

    print("Classifying regimes (AND growth logic, 2.5% CPI threshold)...")
    df = classify(df_raw, cpi_threshold=2.5, require_both_growth=True, min_duration=3)

    # --- Core statistics ---
    print("Computing statistics...")
    stats = compute_regime_stats(df)
    transition_matrix = compute_transition_matrix(df["REGIME"])
    print_stats_table(stats)

    print("Transition matrix:")
    print(transition_matrix.to_string(float_format=lambda x: f"{x:.2f}"))

    # --- Signal conflict report ---
    print_signal_conflict_report(df)

    # --- Core charts ---
    print("Generating core charts...")
    plot_regime_timeline(df, save_path=_out("regime_timeline.png"))
    plot_transition_matrix(transition_matrix, save_path=_out("transition_matrix.png"))
    plot_signal_chart(df, save_path=_out("signal_chart.png"))

    # --- Robustness checks ---
    print("\nRunning robustness checks...")

    # 1. Threshold sensitivity
    print("  Threshold sensitivity analysis...")
    sensitivity_results = threshold_sensitivity_analysis(df_raw, thresholds=[2.0, 2.5, 3.0])
    print_threshold_sensitivity(sensitivity_results)
    plot_threshold_sensitivity(sensitivity_results, save_path=_out("threshold_sensitivity.png"))

    # 2. Era breakdown
    print("  Era breakdown...")
    era_results = era_breakdown(df)
    print_era_breakdown(era_results)
    plot_era_breakdown(era_results, save_path=_out("era_breakdown.png"))

    # 3. Filtered vs unfiltered comparison
    print_filtered_vs_unfiltered(df, min_duration=3)

    # 4. Updated timeline with era annotations
    print("  Generating regime_timeline_v2...")
    plot_regime_timeline_v2(df, era_results=era_results, save_path=_out("regime_timeline_v2.png"))

    print("\nDone. Output files:")
    for f in [
        "regime_timeline.png",
        "transition_matrix.png",
        "signal_chart.png",
        "threshold_sensitivity.png",
        "era_breakdown.png",
        "regime_timeline_v2.png",
    ]:
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
