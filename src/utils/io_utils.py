"""
io_utils.py
File I/O helpers: JSON round-trip, parquet, run artifact saving.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def generate_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    short = str(uuid.uuid4())[:8]
    return f"{ts}_{short}"


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_run_dir(
    experiments_dir: Path,
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir = experiments_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        save_json(config, run_dir / "config.json")
    return run_dir


def save_case_result(
    result: Dict[str, Any],
    run_dir: Path,
    case_id: str,
) -> Path:
    out_path = run_dir / "case_results" / f"{case_id}.json"
    save_json(result, out_path)
    return out_path


def flatten_case_results_to_df(
    results: list[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for r in results:
        case_id = r.get("case_id", "")
        prompt_id = r.get("prompt_id", "")
        model_id = r.get("model_id", "")
        for feat_name, feat_data in r.get("features", {}).items():
            row = {
                "case_id": case_id,
                "prompt_id": prompt_id,
                "model_id": model_id,
                "feature_name": feat_name,
            }
            row.update(feat_data)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/utils/io_utils.py
# Purpose: File I/O helpers for run directory management, JSON/parquet
#          saving, and flattening nested case results into tidy DataFrames.
#
# Functions:
#   generate_run_id() -> str
#     Returns a timestamped run ID string: "run_YYYYMMDD_HHMMSS".
#
#   make_run_dir(base_dir, run_id) -> Path
#     Creates experiments/runs/{run_id}/ and returns the Path.
#
#   save_json(data, path) -> None
#     Writes a dict to a JSON file with indent=2, default=str fallback.
#
#   load_json(path) -> dict
#     Reads and returns a JSON file as a dict.
#
#   save_parquet(df, path) -> None
#     Saves a pandas DataFrame to parquet via pyarrow.
#
#   save_case_result(result, run_dir) -> Path
#     Saves one case result dict to {run_dir}/{case_id}.json.
#     Returns the saved file path.
#
#   flatten_case_results_to_df(results) -> pd.DataFrame
#     Converts a list of case result dicts into a flat wide DataFrame
#     with one row per (case, feature). Includes run_id, prompt_id,
#     model_id, feature_name, and all FeatureResult fields as columns.
#
# Consumed by:
#   src/workflows/orchestration.py
#   fabrication_analysis/03_fabrication_omission_pipeline.ipynb
# =============================================================================
