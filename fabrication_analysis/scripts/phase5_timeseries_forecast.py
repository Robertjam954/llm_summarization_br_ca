"""
Phase 5: Time-Series Prompt Iteration Forecasting
==================================================
Loads iteration tracking data from Phase 4 (phase4_tracking_table.csv +
run_summary.json) and forecasts accuracy / fabrication-detection rates over
future prompt iterations using linear regression, exponential smoothing,
and (if sufficient data) ARIMA.

Usage:
    python fabrication_analysis/scripts/phase5_timeseries_forecast.py

Outputs (all under experiments/runs/v1_v2_comparison/forecast/):
    forecast_metrics.json
    forecast_plots.png
    feasibility_report.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT",
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center"
    r"\Documents\GitHub\llm_summarization_br_ca"))
DATA_PRIVATE = Path(os.getenv("DATA_PRIVATE_DIR",
    r"C:\Users\jamesr4\loc\data_private"))

RUN_DIR     = PROJECT_ROOT / "experiments" / "runs" / "v1_v2_comparison"
OUT_DIR     = RUN_DIR / "forecast"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

MIN_POINTS_FOR_ARIMA = 5   # need more iterations before ARIMA is reliable
MIN_POINTS_FOR_ES    = 3   # exponential smoothing minimum


# ── Load iteration data ────────────────────────────────────────────────────────
def load_iterations() -> pd.DataFrame:
    """
    Load iteration summary from run_summary.json.
    Returns DataFrame with columns: iteration, precision, recall, f1,
    issue_match_rate, version.
    """
    summary_path = RUN_DIR / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"run_summary.json not found at {summary_path}. "
            "Run notebook 06 first."
        )

    with open(summary_path) as f:
        summary = json.load(f)

    rows = []
    for it in summary.get("iterations", []):
        rows.append({
            "iteration":        it["iteration"],
            "version":          it.get("version", ""),
            "prompt_id":        it.get("prompt_id", ""),
            "precision":        it.get("precision"),
            "recall":           it.get("recall"),
            "f1":               it.get("f1"),
            "issue_match_rate": it.get("issue_match_rate"),
        })

    if not rows:
        raise ValueError("No iteration data found in run_summary.json")

    return pd.DataFrame(rows).sort_values("iteration").reset_index(drop=True)


# ── Linear regression trend ────────────────────────────────────────────────────
def linear_forecast(x: np.ndarray, y: np.ndarray,
                    n_future: int = 3) -> dict:
    coeffs = np.polyfit(x, y, 1)  # slope, intercept
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    x_future = np.arange(x[-1] + 1, x[-1] + n_future + 1)
    y_future = slope * x_future + intercept
    y_future = np.clip(y_future, 0, 1)

    return {
        "method":    "linear_regression",
        "slope":     round(slope, 4),
        "intercept": round(intercept, 4),
        "x_future":  x_future.tolist(),
        "y_future":  y_future.tolist(),
        "r_squared": round(float(np.corrcoef(x, y)[0, 1] ** 2), 4)
                     if len(x) > 1 else None,
    }


# ── Exponential smoothing ──────────────────────────────────────────────────────
def exp_smoothing_forecast(y: np.ndarray, n_future: int = 3,
                           alpha: float = 0.6) -> dict:
    """Simple exponential smoothing."""
    smoothed = [float(y[0])]
    for v in y[1:]:
        smoothed.append(alpha * float(v) + (1 - alpha) * smoothed[-1])
    last = smoothed[-1]
    y_future = np.full(n_future, last)

    return {
        "method":    "exponential_smoothing",
        "alpha":     alpha,
        "smoothed":  [round(s, 4) for s in smoothed],
        "y_future":  np.clip(y_future, 0, 1).tolist(),
    }


# ── ARIMA (optional) ──────────────────────────────────────────────────────────
def arima_forecast(y: np.ndarray, n_future: int = 3) -> dict | None:
    if len(y) < MIN_POINTS_FOR_ARIMA:
        return {
            "method":  "arima",
            "status":  f"skipped — need >= {MIN_POINTS_FOR_ARIMA} iterations, "
                       f"have {len(y)}",
            "y_future": None,
        }
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(y, order=(1, 0, 0))
        fit   = model.fit()
        fc    = fit.forecast(steps=n_future)
        return {
            "method":   "arima",
            "order":    (1, 0, 0),
            "aic":      round(float(fit.aic), 3),
            "y_future": np.clip(fc, 0, 1).tolist(),
        }
    except Exception as exc:
        return {"method": "arima", "status": f"failed: {exc}", "y_future": None}


# ── Change-point detection ────────────────────────────────────────────────────
def detect_change_points(y: np.ndarray) -> list[int]:
    """Simple L1-norm change-point detection for short series."""
    if len(y) < 3:
        return []
    diffs   = np.abs(np.diff(y))
    thresh  = np.mean(diffs) + np.std(diffs)
    changes = [int(i + 1) for i, d in enumerate(diffs) if d > thresh]
    return changes


# ── Feasibility assessment ────────────────────────────────────────────────────
def assess_feasibility(df_iter: pd.DataFrame) -> dict:
    n = len(df_iter)
    metrics_available = df_iter["f1"].notna().sum()

    if n < 2:
        status = "insufficient_data"
        recommendation = (
            f"Need at least 2 iterations for trend analysis; have {n}. "
            "Run more prompt iterations."
        )
    elif n < MIN_POINTS_FOR_ARIMA:
        status = "limited_data"
        recommendation = (
            f"Have {n} iterations — linear regression and exponential smoothing "
            f"available. Run {MIN_POINTS_FOR_ARIMA - n} more for ARIMA."
        )
    else:
        status = "sufficient_data"
        recommendation = "Sufficient data for all forecast methods."

    f1_values = df_iter["f1"].dropna().values
    trend_dir = None
    if len(f1_values) >= 2:
        slope = float(np.polyfit(range(len(f1_values)), f1_values, 1)[0])
        trend_dir = "improving" if slope > 0.01 else (
            "degrading" if slope < -0.01 else "stable"
        )

    return {
        "n_iterations":       n,
        "metrics_available":  int(metrics_available),
        "status":             status,
        "recommendation":     recommendation,
        "f1_trend_direction": trend_dir,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def make_forecast_plots(df_iter: pd.DataFrame,
                        forecasts: dict,
                        n_future: int,
                        save_path: Path) -> None:
    metrics_to_plot = ["precision", "recall", "f1"]
    n_cols = len(metrics_to_plot)

    fig = plt.figure(figsize=(6 * n_cols, 5))
    gs  = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.35)

    x_obs = df_iter["iteration"].values.astype(float)
    x_fut = np.arange(x_obs[-1] + 1, x_obs[-1] + n_future + 1)

    for col_idx, metric in enumerate(metrics_to_plot):
        ax  = fig.add_subplot(gs[0, col_idx])
        y   = df_iter[metric].values.astype(float)
        fc  = forecasts.get(metric, {})

        # Observed
        ax.plot(x_obs, y, "o-", color="#2c3e50", lw=2, ms=8,
                label="Observed", zorder=4)
        for xi, yi, ver in zip(x_obs, y, df_iter["version"].values):
            ax.annotate(f"{ver}\n{yi:.2f}", (xi, yi),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        # Linear regression
        lr = fc.get("linear")
        if lr and lr.get("y_future"):
            ax.plot(x_fut, lr["y_future"], "--", color="#3498db",
                    lw=1.5, label=f"Linear (R²={lr.get('r_squared','?')})")

        # Exponential smoothing
        es = fc.get("exp_smoothing")
        if es and es.get("y_future"):
            ax.plot(x_fut, es["y_future"], "--", color="#e67e22",
                    lw=1.5, label="Exp. Smoothing")

        # ARIMA
        ar = fc.get("arima")
        if ar and ar.get("y_future"):
            ax.plot(x_fut, ar["y_future"], "--", color="#9b59b6",
                    lw=1.5, label="ARIMA(1,0,0)")

        ax.set_xlim(x_obs[0] - 0.3, x_fut[-1] + 0.3)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xticks(list(x_obs) + list(x_fut))
        ax.axvspan(x_obs[-1] + 0.5, x_fut[-1] + 0.5,
                   alpha=0.06, color="gray", label="Forecast zone")
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xlabel("Prompt Iteration")
        ax.set_ylabel("Score")
        if col_idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Phase 5 — Prompt Iteration Forecasting\n"
        "v1 (narrative) → v2 (JSON) → future iterations",
        fontsize=12, fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    N_FUTURE = 3

    print("Loading iteration data...")
    df_iter = load_iterations()
    print(df_iter[["iteration", "version", "precision", "recall",
                   "f1", "issue_match_rate"]].to_string(index=False))

    feasibility = assess_feasibility(df_iter)
    print(f"\nFeasibility: {feasibility['status']}")
    print(f"  {feasibility['recommendation']}")
    print(f"  F1 trend: {feasibility['f1_trend_direction']}")

    # ── Run forecasts per metric ───────────────────────────────────────────────
    forecasts_out = {}
    metrics = ["precision", "recall", "f1", "issue_match_rate"]

    for metric in metrics:
        y_vals = df_iter[metric].dropna().values
        x_vals = np.arange(1, len(y_vals) + 1, dtype=float)

        if len(y_vals) < 1:
            continue

        fc = {
            "linear":       linear_forecast(x_vals, y_vals, N_FUTURE),
            "exp_smoothing": exp_smoothing_forecast(y_vals, N_FUTURE),
            "arima":        arima_forecast(y_vals, N_FUTURE),
            "change_points": detect_change_points(y_vals),
        }
        forecasts_out[metric] = fc

        lr = fc["linear"]
        print(f"\n  {metric}:")
        print(f"    Linear slope: {lr['slope']:+.4f} per iteration "
              f"(R²={lr.get('r_squared', 'N/A')})")
        if fc["arima"].get("y_future"):
            print(f"    ARIMA forecast (next {N_FUTURE}): "
                  f"{[round(v,3) for v in fc['arima']['y_future']]}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    make_forecast_plots(
        df_iter, forecasts_out, N_FUTURE,
        save_path=REPORTS_DIR / "phase5_forecast_plots.png",
    )

    # ── Save outputs ───────────────────────────────────────────────────────────
    json.dump(
        {k: {m: {kk: vv for kk, vv in v.items() if kk != "smoothed"}
             for m, v in fc_data.items()}
         for k, fc_data in {"forecasts": forecasts_out}.items()},
        open(OUT_DIR / "forecast_metrics.json", "w"), indent=2, default=str,
    )

    feasibility["iterations_data"] = df_iter.to_dict(orient="records")
    json.dump(
        feasibility,
        open(OUT_DIR / "feasibility_report.json", "w"),
        indent=2, default=str,
    )

    print(f"\nPhase 5 complete. Outputs in: {OUT_DIR}")
    print(f"Figures in: {REPORTS_DIR}")

    # ── Interpretation note ────────────────────────────────────────────────────
    print("\nNOTE: With only 2 iterations, linear regression gives directional")
    print("insight only. Collect 5+ iterations for statistically reliable forecasts.")


if __name__ == "__main__":
    main()
