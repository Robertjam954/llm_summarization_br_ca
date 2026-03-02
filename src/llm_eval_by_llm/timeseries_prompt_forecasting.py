"""
Time-Series Forecasting of Accuracy & Fabrication Rate Over Prompt Iterations
==============================================================================
Evaluates whether changes in model accuracy and fabrication rate across
sequential prompt iterations can be modeled with time-series methods.

Approaches:
  1. Simple trend analysis (linear regression on iteration index)
  2. Exponential smoothing (Holt's method for trend)
  3. ARIMA (if enough data points)
  4. Change-point detection (to identify when a prompt change had impact)

Reads from: data/processed/prompt_iteration_tracking/tracking_table.csv
Outputs to: data/processed/prompt_iteration_tracking/forecasting/

Usage:
    python timeseries_prompt_forecasting.py
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKER_DIR = PROJECT_ROOT / "data" / "processed" / "prompt_iteration_tracking"
FORECAST_DIR = TRACKER_DIR / "forecasting"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ===================================================================
# Load tracking data
# ===================================================================
def load_tracking_data() -> pd.DataFrame:
    """Load the tracking table from prompt_iteration_tracker output."""
    path = TRACKER_DIR / "tracking_table.csv"
    if not path.exists():
        log.error("Tracking table not found at %s. Run prompt_iteration_tracker.py first.", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    log.info("Loaded tracking table: %d rows", len(df))
    return df


# ===================================================================
# Prepare time-series data
# ===================================================================
def prepare_timeseries(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a time-ordered series of (iteration, model, approach, accuracy, fabrication_rate).
    Iterations are ordered by run_id timestamp suffix.
    """
    if tracking_df.empty:
        return pd.DataFrame()

    # Get aggregate rows only
    agg_df = tracking_df[tracking_df["element"] == "__AGGREGATE__"].copy()
    if agg_df.empty:
        # Fall back to computing aggregates from element rows
        elem_df = tracking_df[tracking_df["element"] != "__AGGREGATE__"].copy()
        agg_df = elem_df.groupby(["run_id", "model", "prompt_approach"]).agg(
            accuracy=("accuracy", "mean"),
            fabrication_rate=("fabrication_rate", "mean"),
        ).reset_index()

    # Sort by run_id (which contains timestamp)
    agg_df = agg_df.sort_values("run_id").reset_index(drop=True)
    agg_df["iteration"] = range(len(agg_df))

    return agg_df


# ===================================================================
# Linear trend analysis
# ===================================================================
def linear_trend_analysis(ts_df: pd.DataFrame) -> dict:
    """
    Fit a simple linear regression: metric ~ iteration.
    Returns slope, intercept, R², p-value for both accuracy and fabrication_rate.
    """
    if len(ts_df) < 3:
        return {
            "sufficient_data": False,
            "message": f"Only {len(ts_df)} data points; need ≥3 for trend analysis.",
        }

    from scipy import stats

    results = {}
    for metric in ["accuracy", "fabrication_rate"]:
        y = ts_df[metric].dropna().values
        x = np.arange(len(y))

        if len(y) < 3:
            results[metric] = {"sufficient_data": False}
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        results[metric] = {
            "sufficient_data": True,
            "n_points": len(y),
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 6),
            "std_err": round(float(std_err), 6),
            "trend_direction": "improving" if (
                (metric == "accuracy" and slope > 0) or
                (metric == "fabrication_rate" and slope < 0)
            ) else "worsening" if (
                (metric == "accuracy" and slope < 0) or
                (metric == "fabrication_rate" and slope > 0)
            ) else "flat",
            "significant": bool(p_value < 0.05),
        }

    return {"method": "linear_regression", **results}


# ===================================================================
# Exponential smoothing (Holt's linear trend)
# ===================================================================
def exponential_smoothing_analysis(ts_df: pd.DataFrame) -> dict:
    """
    Apply Holt's linear trend method to forecast next iteration's metrics.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return {"method": "exponential_smoothing", "error": "statsmodels not installed"}

    if len(ts_df) < 4:
        return {
            "method": "exponential_smoothing",
            "sufficient_data": False,
            "message": f"Only {len(ts_df)} data points; need ≥4 for Holt's method.",
        }

    results = {}
    for metric in ["accuracy", "fabrication_rate"]:
        y = ts_df[metric].dropna().values
        if len(y) < 4:
            results[metric] = {"sufficient_data": False}
            continue

        try:
            model = ExponentialSmoothing(
                y, trend="add", seasonal=None, initialization_method="estimated"
            ).fit(optimized=True)

            forecast = model.forecast(steps=3)
            fitted = model.fittedvalues

            # Residual analysis
            residuals = y - fitted
            mae = float(np.mean(np.abs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))

            results[metric] = {
                "sufficient_data": True,
                "n_points": len(y),
                "forecast_next_3": [round(float(f), 4) for f in forecast],
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "smoothing_level": round(float(model.params.get("smoothing_level", 0)), 4),
                "smoothing_trend": round(float(model.params.get("smoothing_trend", 0)), 4),
            }
        except Exception as exc:
            results[metric] = {"error": str(exc)}

    return {"method": "exponential_smoothing", **results}


# ===================================================================
# ARIMA analysis
# ===================================================================
def arima_analysis(ts_df: pd.DataFrame) -> dict:
    """
    Fit ARIMA model if enough data points exist.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"method": "arima", "error": "statsmodels not installed"}

    if len(ts_df) < 8:
        return {
            "method": "arima",
            "sufficient_data": False,
            "message": f"Only {len(ts_df)} data points; need ≥8 for ARIMA.",
        }

    results = {}
    for metric in ["accuracy", "fabrication_rate"]:
        y = ts_df[metric].dropna().values
        if len(y) < 8:
            results[metric] = {"sufficient_data": False}
            continue

        try:
            # Try simple ARIMA(1,1,1)
            model = ARIMA(y, order=(1, 1, 1)).fit()
            forecast = model.forecast(steps=3)

            results[metric] = {
                "sufficient_data": True,
                "n_points": len(y),
                "order": [1, 1, 1],
                "aic": round(float(model.aic), 2),
                "bic": round(float(model.bic), 2),
                "forecast_next_3": [round(float(f), 4) for f in forecast],
            }
        except Exception as exc:
            results[metric] = {"error": str(exc)}

    return {"method": "arima", **results}


# ===================================================================
# Change-point detection
# ===================================================================
def change_point_detection(ts_df: pd.DataFrame) -> dict:
    """
    Simple change-point detection using cumulative sum (CUSUM) approach.
    Identifies iterations where accuracy or fabrication rate shifted significantly.
    """
    if len(ts_df) < 5:
        return {
            "method": "cusum_change_point",
            "sufficient_data": False,
            "message": f"Only {len(ts_df)} data points; need ≥5 for change-point detection.",
        }

    results = {}
    for metric in ["accuracy", "fabrication_rate"]:
        y = ts_df[metric].dropna().values
        if len(y) < 5:
            results[metric] = {"sufficient_data": False}
            continue

        mean_val = np.mean(y)
        cusum = np.cumsum(y - mean_val)

        # Find the point of maximum deviation from the mean
        max_idx = int(np.argmax(np.abs(cusum)))
        max_deviation = float(cusum[max_idx])

        # Check if the change is meaningful (> 1 std dev)
        std_val = np.std(y)
        is_significant = abs(max_deviation) > std_val * np.sqrt(len(y)) if std_val > 0 else False

        # Before/after comparison
        if max_idx > 0 and max_idx < len(y) - 1:
            before_mean = float(np.mean(y[:max_idx + 1]))
            after_mean = float(np.mean(y[max_idx + 1:]))
            shift = after_mean - before_mean
        else:
            before_mean = after_mean = shift = None

        # Map change point back to run info
        change_run = ts_df.iloc[max_idx] if max_idx < len(ts_df) else None

        results[metric] = {
            "sufficient_data": True,
            "change_point_iteration": max_idx,
            "change_point_run": str(change_run["run_id"]) if change_run is not None else None,
            "change_point_approach": str(change_run["prompt_approach"]) if change_run is not None else None,
            "cusum_max_deviation": round(max_deviation, 4),
            "is_significant": bool(is_significant),
            "before_mean": round(before_mean, 4) if before_mean is not None else None,
            "after_mean": round(after_mean, 4) if after_mean is not None else None,
            "shift": round(shift, 4) if shift is not None else None,
        }

    return {"method": "cusum_change_point", **results}


# ===================================================================
# Per-model analysis
# ===================================================================
def per_model_trend(tracking_df: pd.DataFrame) -> dict:
    """Analyze trends separately for each model."""
    models = tracking_df["model"].unique()
    model_results = {}

    for model in models:
        model_df = tracking_df[tracking_df["model"] == model]
        ts = prepare_timeseries(model_df)
        if ts.empty or len(ts) < 2:
            model_results[model] = {"sufficient_data": False, "n_runs": len(ts)}
            continue

        model_results[model] = {
            "n_runs": len(ts),
            "linear_trend": linear_trend_analysis(ts),
        }

    return model_results


# ===================================================================
# Feasibility assessment
# ===================================================================
def assess_forecasting_feasibility(ts_df: pd.DataFrame, all_results: dict) -> dict:
    """
    Provide a plain-language assessment of whether time-series forecasting
    makes sense for this data.
    """
    n_points = len(ts_df)

    assessment = {
        "n_data_points": n_points,
        "recommendations": [],
    }

    if n_points < 3:
        assessment["verdict"] = "NOT_FEASIBLE"
        assessment["recommendations"].append(
            "Fewer than 3 prompt iterations completed. Run more experiments first."
        )
        return assessment

    if n_points < 8:
        assessment["verdict"] = "LIMITED"
        assessment["recommendations"].append(
            f"Only {n_points} iterations. Linear trend analysis is possible but "
            "ARIMA and advanced methods need ≥8 data points."
        )
    else:
        assessment["verdict"] = "FEASIBLE"
        assessment["recommendations"].append(
            f"{n_points} iterations available. All forecasting methods can be applied."
        )

    # Check if there's actually a trend
    linear = all_results.get("linear_trend", {})
    for metric in ["accuracy", "fabrication_rate"]:
        m = linear.get(metric, {})
        if m.get("significant"):
            assessment["recommendations"].append(
                f"Significant linear trend detected in {metric} "
                f"(slope={m.get('slope')}, p={m.get('p_value')}). "
                f"Direction: {m.get('trend_direction')}."
            )
        elif m.get("sufficient_data"):
            assessment["recommendations"].append(
                f"No significant linear trend in {metric} (p={m.get('p_value')}). "
                "Prompt changes may have inconsistent effects, or more data is needed."
            )

    # Check variance
    for metric in ["accuracy", "fabrication_rate"]:
        vals = ts_df[metric].dropna().values
        if len(vals) >= 3:
            cv = np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else 0
            if cv > 0.2:
                assessment["recommendations"].append(
                    f"High variability in {metric} (CV={cv:.2f}). "
                    "Consider grouping by model or approach for cleaner trends."
                )

    return assessment


# ===================================================================
# Visualization
# ===================================================================
def plot_forecasting(ts_df: pd.DataFrame, results: dict, output_dir: Path):
    """Generate time-series plots with trend lines and forecasts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return

    if ts_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, metric, title in zip(
        axes,
        ["accuracy", "fabrication_rate"],
        ["Accuracy Over Prompt Iterations", "Fabrication Rate Over Prompt Iterations"],
    ):
        y = ts_df[metric].dropna().values
        x = np.arange(len(y))

        # Scatter + line
        ax.plot(x, y, "o-", color="steelblue", markersize=6, label="Observed")

        # Linear trend
        linear = results.get("linear_trend", {}).get(metric, {})
        if linear.get("sufficient_data"):
            slope = linear["slope"]
            intercept = linear["intercept"]
            trend_line = slope * x + intercept
            ax.plot(x, trend_line, "--", color="red", alpha=0.7,
                    label=f"Linear trend (R²={linear['r_squared']:.3f})")

        # Exponential smoothing forecast
        es = results.get("exponential_smoothing", {}).get(metric, {})
        if es.get("forecast_next_3"):
            forecast_x = np.arange(len(y), len(y) + 3)
            ax.plot(forecast_x, es["forecast_next_3"], "s--", color="green",
                    alpha=0.7, label="Holt forecast (next 3)")

        # Change point
        cp = results.get("change_point", {}).get(metric, {})
        if cp.get("is_significant") and cp.get("change_point_iteration") is not None:
            cp_idx = cp["change_point_iteration"]
            ax.axvline(x=cp_idx, color="orange", linestyle=":", alpha=0.8,
                       label=f"Change point (iter {cp_idx})")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Annotate with approach names
        if "prompt_approach" in ts_df.columns:
            for i, approach in enumerate(ts_df["prompt_approach"].values[:len(y)]):
                ax.annotate(approach, (i, y[i]), fontsize=6, rotation=45,
                            ha="left", va="bottom", alpha=0.6)

    axes[-1].set_xlabel("Prompt Iteration")
    plt.tight_layout()
    plt.savefig(output_dir / "timeseries_forecast.png", dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved forecast plot to %s", output_dir / "timeseries_forecast.png")


# ===================================================================
# Main
# ===================================================================
def main():
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    tracking_df = load_tracking_data()
    if tracking_df.empty:
        log.warning("No tracking data available. Run the pipeline and tracker first.")
        # Write feasibility report
        report = {
            "status": "NO_DATA",
            "message": "No prompt iteration data available yet. "
                       "Run deepeval_multi_model_pipeline.py followed by prompt_iteration_tracker.py first.",
            "recommendations": [
                "1. Run: python deepeval_multi_model_pipeline.py --dry-run  (to test the pipeline)",
                "2. Run: python deepeval_multi_model_pipeline.py  (with real LLM calls)",
                "3. Run: python prompt_iteration_tracker.py  (to build tracking table)",
                "4. Run: python timeseries_prompt_forecasting.py  (this script, for forecasting)",
            ],
        }
        with open(FORECAST_DIR / "forecasting_report.json", "w") as f:
            json.dump(report, f, indent=2)
        log.info("Wrote empty report to %s", FORECAST_DIR / "forecasting_report.json")
        return

    # Prepare time series
    ts_df = prepare_timeseries(tracking_df)
    log.info("Time series has %d data points", len(ts_df))

    # Run analyses
    all_results = {}

    log.info("Running linear trend analysis...")
    all_results["linear_trend"] = linear_trend_analysis(ts_df)

    log.info("Running exponential smoothing...")
    all_results["exponential_smoothing"] = exponential_smoothing_analysis(ts_df)

    log.info("Running ARIMA analysis...")
    all_results["arima"] = arima_analysis(ts_df)

    log.info("Running change-point detection...")
    all_results["change_point"] = change_point_detection(ts_df)

    log.info("Running per-model trend analysis...")
    all_results["per_model"] = per_model_trend(tracking_df)

    # Feasibility assessment
    log.info("Assessing forecasting feasibility...")
    all_results["feasibility"] = assess_forecasting_feasibility(ts_df, all_results)

    # Save results
    report_path = FORECAST_DIR / "forecasting_report.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved forecasting report to %s", report_path)

    # Save time series data
    ts_df.to_csv(FORECAST_DIR / "timeseries_data.csv", index=False)

    # Plot
    plot_forecasting(ts_df, all_results, FORECAST_DIR)

    # Print summary
    feasibility = all_results["feasibility"]
    log.info("=" * 60)
    log.info("FORECASTING FEASIBILITY: %s", feasibility["verdict"])
    for rec in feasibility["recommendations"]:
        log.info("  → %s", rec)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
