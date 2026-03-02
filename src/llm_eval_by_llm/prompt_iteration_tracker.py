"""
Prompt Iteration Tracker & Auto-Refinement
============================================
Tracks total correct and fabrication count per element across prompt versions
and prompting approaches. Models auto-generate new prompt versions with
reasoning about what problem they are trying to address.

Reads results from deepeval_runs/ and produces:
  1. A tracking table (CSV/JSON) of metrics per (model, prompt_approach, element)
  2. Auto-generated new prompt versions with reasoning
  3. Visualization of metric trends across prompt iterations

Usage:
    python prompt_iteration_tracker.py [--auto-refine] [--max-iterations 5]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEEPEVAL_RUNS_DIR = PROCESSED_DIR / "deepeval_runs"
TRACKER_DIR = PROCESSED_DIR / "prompt_iteration_tracking"
PROMPT_LIBRARY_CSV = PROJECT_ROOT / "references" / "Prompts" / "prompt_library (1).csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element list
# ---------------------------------------------------------------------------
ELEMENTS = [
    "Lesion Size", "Lesion Laterality", "Lesion Location",
    "Calcifications / Asymmetry", "Additional Enhancement (MRI)",
    "Extent", "Accurate Clip Placement", "Workup Recommendation",
    "Lymph Node", "Chronology Preserved", "Biopsy Method",
    "Invasive Component Size (Pathology)", "Histologic Diagnosis",
    "Receptor Status",
]


# ===================================================================
# Load all run results
# ===================================================================
def load_all_runs(runs_dir: Path) -> list:
    """Load metrics from all completed runs."""
    runs = []
    if not runs_dir.exists():
        log.warning("No runs directory found at %s", runs_dir)
        return runs

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Parse run_id to extract model and approach
        run_id = run_dir.name
        parts = run_id.rsplit("_", 2)  # model_approach_timestamp
        # Heuristic: find the approach name
        model = None
        approach = None
        for known_approach in [
            "zero_shot", "zero_shot2", "few_shot", "few_shot2",
            "chain_of_thought", "chain_of_thought2",
            "program_aided", "program_aided2",
            "rag", "rag2", "react", "react2",
            "bfop", "2pop",
        ]:
            if f"_{known_approach}_" in run_id:
                idx = run_id.index(f"_{known_approach}_")
                model = run_id[:idx]
                approach = known_approach
                break

        if model is None:
            model = run_id
            approach = "unknown"

        runs.append({
            "run_id": run_id,
            "run_dir": str(run_dir),
            "model": model,
            "prompt_approach": approach,
            "metrics": metrics,
        })

    log.info("Loaded %d runs", len(runs))
    return runs


# ===================================================================
# Build tracking table
# ===================================================================
def build_tracking_table(runs: list) -> pd.DataFrame:
    """
    Build a flat table with one row per (run, element) showing:
    - model, prompt_approach, element
    - TP, FP, FN, TN, accuracy, precision, recall, f1, fabrication_rate
    """
    rows = []
    for run in runs:
        model = run["model"]
        approach = run["prompt_approach"]
        run_id = run["run_id"]
        metrics = run["metrics"]

        for element in ELEMENTS:
            elem_metrics = metrics.get(element, {})
            if not elem_metrics:
                continue
            rows.append({
                "run_id": run_id,
                "model": model,
                "prompt_approach": approach,
                "element": element,
                "TP": elem_metrics.get("TP"),
                "FP": elem_metrics.get("FP"),
                "FN": elem_metrics.get("FN"),
                "TN": elem_metrics.get("TN"),
                "total_evaluated": elem_metrics.get("total_evaluated"),
                "accuracy": elem_metrics.get("accuracy"),
                "precision": elem_metrics.get("precision"),
                "recall": elem_metrics.get("recall"),
                "f1": elem_metrics.get("f1"),
                "fabrication_rate": elem_metrics.get("fabrication_rate"),
            })

        # Add aggregate row
        agg = metrics.get("__aggregate__", {})
        if agg:
            rows.append({
                "run_id": run_id,
                "model": model,
                "prompt_approach": approach,
                "element": "__AGGREGATE__",
                "TP": None,
                "FP": agg.get("total_fabrications"),
                "FN": agg.get("total_misses"),
                "TN": None,
                "total_evaluated": agg.get("total_evaluated"),
                "accuracy": agg.get("overall_accuracy"),
                "precision": None,
                "recall": None,
                "f1": None,
                "fabrication_rate": agg.get("overall_fabrication_rate"),
            })

    return pd.DataFrame(rows)


# ===================================================================
# Identify worst-performing elements
# ===================================================================
def identify_problem_elements(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each element, find the best and worst performing (model, approach)
    and compute the gap. Elements with low accuracy or high fabrication
    are flagged for prompt refinement.
    """
    elem_df = tracking_df[tracking_df["element"] != "__AGGREGATE__"].copy()
    if elem_df.empty:
        return pd.DataFrame()

    summary = elem_df.groupby("element").agg(
        mean_accuracy=("accuracy", "mean"),
        max_accuracy=("accuracy", "max"),
        min_accuracy=("accuracy", "min"),
        mean_fabrication=("fabrication_rate", "mean"),
        max_fabrication=("fabrication_rate", "max"),
        n_runs=("run_id", "count"),
    ).reset_index()

    summary["accuracy_gap"] = summary["max_accuracy"] - summary["min_accuracy"]
    summary["needs_refinement"] = (
        (summary["mean_accuracy"] < 0.90) |
        (summary["mean_fabrication"] > 0.02) |
        (summary["accuracy_gap"] > 0.10)
    )

    return summary.sort_values("mean_accuracy")


# ===================================================================
# Auto-generate refined prompts using an LLM
# ===================================================================
REFINEMENT_SYSTEM_PROMPT = """You are a prompt engineering expert specializing in clinical NLP for oncology.

Your task: Given performance metrics for a clinical data extraction prompt, generate an IMPROVED prompt version.

For each element, you will receive:
- The current prompt text
- Performance metrics (accuracy, precision, recall, fabrication rate)
- Common error patterns

You must:
1. Analyze WHY the current prompt is failing (miss vs fabrication)
2. Propose a specific fix addressing the root cause
3. Write the improved prompt
4. Explain your reasoning

Output JSON with keys: "reasoning", "problem_addressed", "improved_prompt"
"""


def auto_refine_prompt(
    element_key: str,
    element_display: str,
    current_prompt: str,
    metrics: dict,
    model_name: str = "gpt-4o",
) -> dict:
    """
    Use an LLM to generate a refined prompt for an underperforming element.
    """
    # Import LLM caller from the pipeline
    try:
        from deepeval_multi_model_pipeline import call_llm
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from deepeval_multi_model_pipeline import call_llm

    user_prompt = f"""Element: {element_display}

Current prompt:
{current_prompt}

Performance metrics across runs:
- Mean accuracy: {metrics.get('mean_accuracy', 'N/A')}
- Max accuracy: {metrics.get('max_accuracy', 'N/A')}
- Min accuracy: {metrics.get('min_accuracy', 'N/A')}
- Mean fabrication rate: {metrics.get('mean_fabrication', 'N/A')}
- Max fabrication rate: {metrics.get('max_fabrication', 'N/A')}

The element has {'high fabrication rate' if metrics.get('mean_fabrication', 0) > 0.02 else 'low accuracy (misses)'}.

Generate an improved prompt. Output JSON:
{{"reasoning": "...", "problem_addressed": "...", "improved_prompt": "..."}}
"""

    try:
        response = call_llm(model_name, REFINEMENT_SYSTEM_PROMPT, user_prompt)
        # Parse JSON from response
        import re
        json_match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "element": element_display,
                "original_prompt": current_prompt,
                "refined_prompt": parsed.get("improved_prompt", ""),
                "reasoning": parsed.get("reasoning", ""),
                "problem_addressed": parsed.get("problem_addressed", ""),
                "generated_by": model_name,
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as exc:
        log.error("Prompt refinement failed for %s: %s", element_display, exc)

    return {
        "element": element_display,
        "original_prompt": current_prompt,
        "refined_prompt": None,
        "reasoning": "Auto-refinement failed",
        "problem_addressed": None,
        "generated_by": model_name,
        "timestamp": datetime.now().isoformat(),
    }


def run_auto_refinement(
    problem_elements: pd.DataFrame,
    prompt_df: pd.DataFrame,
    approach: str = "zero_shot",
    refine_model: str = "gpt-4o",
    dry_run: bool = False,
) -> list:
    """
    Auto-refine prompts for elements flagged as needing refinement.
    """
    # Map element display names to CSV element names (reverse of CSV_ELEMENT_MAP)
    from deepeval_multi_model_pipeline import CSV_ELEMENT_MAP
    display_to_csv = {}
    for csv_name, our_key in CSV_ELEMENT_MAP.items():
        for elem_def in [
            {"key": "lesion_size", "display": "Lesion Size"},
            {"key": "lesion_laterality", "display": "Lesion Laterality"},
            {"key": "lesion_location", "display": "Lesion Location"},
            {"key": "calcifications_asymmetry", "display": "Calcifications / Asymmetry"},
            {"key": "additional_enhancement_mri", "display": "Additional Enhancement (MRI)"},
            {"key": "extent", "display": "Extent"},
            {"key": "accurate_clip_placement", "display": "Accurate Clip Placement"},
            {"key": "workup_recommendation", "display": "Workup Recommendation"},
            {"key": "lymph_node", "display": "Lymph Node"},
            {"key": "chronology_preserved", "display": "Chronology Preserved"},
            {"key": "biopsy_method", "display": "Biopsy Method"},
            {"key": "invasive_component_size", "display": "Invasive Component Size (Pathology)"},
            {"key": "histologic_diagnosis", "display": "Histologic Diagnosis"},
            {"key": "receptor_status", "display": "Receptor Status"},
        ]:
            if elem_def["key"] == our_key:
                display_to_csv[elem_def["display"]] = csv_name

    refinements = []
    flagged = problem_elements[problem_elements["needs_refinement"] == True]

    for _, row in flagged.iterrows():
        element_display = row["element"]
        csv_element = display_to_csv.get(element_display)

        # Get current prompt
        current_prompt = ""
        if csv_element and approach in prompt_df.columns:
            match = prompt_df[prompt_df["element"] == csv_element]
            if not match.empty:
                current_prompt = str(match.iloc[0][approach])

        if not current_prompt:
            log.warning("No current prompt found for %s / %s", element_display, approach)
            continue

        metrics_dict = row.to_dict()

        if dry_run:
            refinement = {
                "element": element_display,
                "original_prompt": current_prompt,
                "refined_prompt": "[DRY RUN — would generate refined prompt here]",
                "reasoning": "Dry run mode",
                "problem_addressed": f"Mean accuracy {metrics_dict.get('mean_accuracy', 'N/A')}",
                "generated_by": refine_model,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            refinement = auto_refine_prompt(
                element_key=csv_element or element_display,
                element_display=element_display,
                current_prompt=current_prompt,
                metrics=metrics_dict,
                model_name=refine_model,
            )

        refinements.append(refinement)
        log.info("  Refined prompt for: %s", element_display)

    return refinements


# ===================================================================
# Visualization
# ===================================================================
def plot_metric_trends(tracking_df: pd.DataFrame, output_dir: Path):
    """Generate plots of accuracy and fabrication rate across prompt approaches."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots")
        return

    elem_df = tracking_df[tracking_df["element"] != "__AGGREGATE__"].copy()
    if elem_df.empty:
        return

    # Plot 1: Accuracy by approach (aggregated across elements)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    approach_stats = elem_df.groupby("prompt_approach").agg(
        mean_accuracy=("accuracy", "mean"),
        mean_fabrication=("fabrication_rate", "mean"),
    ).reset_index().sort_values("mean_accuracy", ascending=False)

    axes[0].barh(approach_stats["prompt_approach"], approach_stats["mean_accuracy"] * 100)
    axes[0].set_xlabel("Mean Accuracy (%)")
    axes[0].set_title("Accuracy by Prompt Approach")
    axes[0].set_xlim(0, 100)

    axes[1].barh(approach_stats["prompt_approach"], approach_stats["mean_fabrication"] * 100, color="salmon")
    axes[1].set_xlabel("Mean Fabrication Rate (%)")
    axes[1].set_title("Fabrication Rate by Prompt Approach")

    plt.tight_layout()
    plt.savefig(output_dir / "approach_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Per-element accuracy heatmap across approaches
    pivot = elem_df.pivot_table(
        index="element", columns="prompt_approach", values="accuracy", aggfunc="mean"
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, label="Accuracy (%)")
        ax.set_title("Element × Approach Accuracy Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "element_approach_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    log.info("Saved plots to %s", output_dir)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Prompt Iteration Tracker")
    parser.add_argument("--auto-refine", action="store_true",
                        help="Auto-generate refined prompts for underperforming elements")
    parser.add_argument("--refine-model", type=str, default="gpt-4o",
                        help="Model to use for prompt refinement")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max auto-refinement iterations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip actual LLM calls")
    args = parser.parse_args()

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = load_all_runs(DEEPEVAL_RUNS_DIR)
    if not runs:
        log.warning("No runs found. Run deepeval_multi_model_pipeline.py first.")
        # Create empty tracking structure
        empty_tracking = pd.DataFrame(columns=[
            "run_id", "model", "prompt_approach", "element",
            "TP", "FP", "FN", "TN", "total_evaluated",
            "accuracy", "precision", "recall", "f1", "fabrication_rate",
        ])
        empty_tracking.to_csv(TRACKER_DIR / "tracking_table.csv", index=False)
        log.info("Created empty tracking table at %s", TRACKER_DIR / "tracking_table.csv")
        return

    # Build tracking table
    tracking_df = build_tracking_table(runs)
    tracking_df.to_csv(TRACKER_DIR / "tracking_table.csv", index=False)
    log.info("Saved tracking table: %d rows", len(tracking_df))

    # Identify problem elements
    problems = identify_problem_elements(tracking_df)
    problems.to_csv(TRACKER_DIR / "problem_elements.csv", index=False)
    log.info("Problem elements:\n%s", problems[["element", "mean_accuracy", "mean_fabrication", "needs_refinement"]].to_string())

    # Generate plots
    plot_metric_trends(tracking_df, TRACKER_DIR)

    # Auto-refinement
    if args.auto_refine:
        log.info("Running auto-refinement...")
        prompt_df = pd.read_csv(PROMPT_LIBRARY_CSV) if PROMPT_LIBRARY_CSV.exists() else pd.DataFrame()

        all_refinements = []
        for iteration in range(args.max_iterations):
            log.info("--- Refinement iteration %d ---", iteration + 1)
            refinements = run_auto_refinement(
                problems, prompt_df,
                approach="zero_shot",
                refine_model=args.refine_model,
                dry_run=args.dry_run,
            )
            if not refinements:
                log.info("No elements need refinement. Stopping.")
                break
            all_refinements.extend(refinements)

            # Save iteration results
            iter_path = TRACKER_DIR / f"refinements_iter_{iteration + 1}.json"
            with open(iter_path, "w") as f:
                json.dump(refinements, f, indent=2, default=str)

        # Save all refinements
        if all_refinements:
            all_ref_path = TRACKER_DIR / "all_refinements.json"
            with open(all_ref_path, "w") as f:
                json.dump(all_refinements, f, indent=2, default=str)
            log.info("Saved %d refinements to %s", len(all_refinements), all_ref_path)

    log.info("Done. Results in %s", TRACKER_DIR)


if __name__ == "__main__":
    main()
