"""
logging_utils.py
Structured logging for the extraction pipeline.
Logs run_id, case_id, feature, node, and verdict at every step.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_node_entry(
    logger: logging.Logger,
    node: str,
    run_id: str,
    case_id: str,
    feature: Optional[str] = None,
) -> None:
    msg = f"[{node}] run={run_id} case={case_id}"
    if feature:
        msg += f" feature={feature}"
    logger.info(msg)


def log_verdict(
    logger: logging.Logger,
    run_id: str,
    case_id: str,
    feature: str,
    verdict: str,
    confidence: float,
    retrieval_attempts: int,
) -> None:
    logger.info(
        f"[VERDICT] run={run_id} case={case_id} feature={feature} "
        f"verdict={verdict} confidence={confidence:.2f} "
        f"retrieval_attempts={retrieval_attempts}"
    )


def log_fabrication_flag(
    logger: logging.Logger,
    run_id: str,
    case_id: str,
    feature: str,
    value: str,
) -> None:
    logger.warning(
        f"[FABRICATION_FLAG] run={run_id} case={case_id} "
        f"feature={feature} value='{value}'"
    )


def log_retrieval(
    logger: logging.Logger,
    run_id: str,
    case_id: str,
    feature: str,
    k: int,
    n_retrieved: int,
    query: str,
) -> None:
    logger.debug(
        f"[RETRIEVAL] run={run_id} case={case_id} feature={feature} "
        f"k={k} retrieved={n_retrieved} query='{query[:80]}...'"
    )


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/utils/logging_utils.py
# Purpose: Structured pipeline logging with per-node, per-feature, and
#          per-verdict context for traceability and debugging.
#
# Functions:
#   get_logger(name) -> logging.Logger
#     Returns a logger with a consistent format: timestamp, level, name.
#
#   log_node_entry(logger, node_name, run_id, case_id, feature) -> None
#     Logs INFO entry into a LangGraph node with run/case/feature context.
#
#   log_verdict(logger, run_id, case_id, feature, verdict,
#               confidence) -> None
#     Logs INFO verdict assignment with confidence score.
#
#   log_fabrication_flag(logger, run_id, case_id, feature, value) -> None
#     Logs WARNING for fabrication flag; intended for audit trails.
#
#   log_retrieval(logger, run_id, case_id, feature, k,
#                 n_retrieved, query) -> None
#     Logs DEBUG retrieval stats: k requested vs n_retrieved, query prefix.
#
# Consumed by: all src/agents/*.py, src/workflows/extraction_graph.py,
#              src/graph/build_graph.py, src/graph/neo4j_io.py
# =============================================================================
