"""
orchestration.py
High-level orchestration: runs the extraction graph for one or many cases.
Saves per-case JSON results and aggregated parquet to the run directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.utils.io_utils import (
    flatten_case_results_to_df,
    generate_run_id,
    make_run_dir,
    save_case_result,
    save_parquet,
)
from src.utils.logging_utils import get_logger
from src.workflows.extraction_state import make_initial_state

load_dotenv()
logger = get_logger(__name__)

PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center"
        r"\Documents\GitHub\llm_summarization_br_ca",
    )
)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def run_single_case(
    case_id: str,
    ocr_text: str,
    prompt_id: str = "rag_verify_v1",
    model_id: str = "claude-3-5-sonnet-20241022",
    feature_queue: Optional[List[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    from src.workflows.extraction_graph import compile_graph

    if run_id is None:
        run_id = generate_run_id()

    state = make_initial_state(
        run_id=run_id,
        case_id=case_id,
        ocr_text=ocr_text,
        prompt_id=prompt_id,
        model_id=model_id,
        feature_queue=feature_queue,
    )

    app = compile_graph()
    final_state = app.invoke(state)

    result = {
        "case_id": case_id,
        "run_id": run_id,
        "prompt_id": prompt_id,
        "model_id": model_id,
        "features": final_state["extracted_elements"],
        "fabrication_flags": final_state["fabrication_flags"],
        "omission_flags": final_state["omission_flags"],
    }
    return result


def run_batch(
    cases: List[Dict[str, str]],
    prompt_id: str = "rag_verify_v1",
    model_id: str = "claude-3-5-sonnet-20241022",
    run_id: Optional[str] = None,
    save_results: bool = True,
) -> List[Dict[str, Any]]:
    if run_id is None:
        run_id = generate_run_id()

    config = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "model_id": model_id,
        "n_cases": len(cases),
    }
    run_dir = make_run_dir(EXPERIMENTS_DIR, run_id, config)
    logger.info(f"[batch] run_id={run_id} n_cases={len(cases)}")

    all_results = []
    for case in cases:
        case_id = case["case_id"]
        ocr_text = case["ocr_text"]
        logger.info(f"[batch] processing case={case_id}")
        try:
            result = run_single_case(
                case_id=case_id,
                ocr_text=ocr_text,
                prompt_id=prompt_id,
                model_id=model_id,
                run_id=run_id,
            )
            if save_results:
                save_case_result(result, run_dir, case_id)
            all_results.append(result)
        except Exception as e:
            logger.error(f"[batch] FAILED case={case_id}: {e}")
            all_results.append({
                "case_id": case_id,
                "run_id": run_id,
                "error": str(e),
            })

    if save_results and all_results:
        df = flatten_case_results_to_df(all_results)
        save_parquet(df, run_dir / "feature_outputs.parquet")
        logger.info(
            f"[batch] saved {len(df)} feature rows to {run_dir}"
        )

    return all_results


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/workflows/orchestration.py
# Purpose: High-level runner that executes the compiled LangGraph extraction
#          graph for single cases or entire batches with run tracking and
#          incremental output saving.
#
# Functions:
#   run_single_case(case_id, ocr_text, prompt_id, model_id, feature_queue,
#                  run_id, save_result) -> dict
#     Runs the full LangGraph pipeline on one OCR document. Saves result to
#     experiments/runs/{run_id}/{case_id}.json. Returns ExtractionState dict.
#
#   run_batch(cases, prompt_id, model_id, run_id, save_results,
#             max_workers) -> list[dict]
#     Runs run_single_case over a list of {case_id, ocr_text} dicts in
#     parallel threads. Returns list of case result dicts.
#
# Outputs:
#   experiments/runs/{run_id}/{case_id}.json   - Per-case auditable JSON
#   experiments/runs/{run_id}/feature_outputs.parquet  - Flat feature table
#   experiments/runs/{run_id}/run_manifest.json        - Run metadata
#
# Consumed by:
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
#   fabrication_analysis/03_fabrication_omission_pipeline.ipynb
# =============================================================================
