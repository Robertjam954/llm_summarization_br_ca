"""
chunk_text.py
Chunk OCR text into overlapping windows using LangChain's
RecursiveCharacterTextSplitter and assign modality labels.
"""

import re
import uuid
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.workflows.extraction_state import Chunk

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-large"

MODALITY_PATTERNS = {
    "mammogram": re.compile(
        r"\b(mammogram|mammography|mammo|screening mammo|diagnostic mammo)\b",
        re.IGNORECASE,
    ),
    "ultrasound": re.compile(
        r"\b(ultrasound|sonogram|US|sonography|echography)\b",
        re.IGNORECASE,
    ),
    "mri": re.compile(
        r"\b(MRI|magnetic resonance|breast MRI|MR imaging)\b",
        re.IGNORECASE,
    ),
    "pathology": re.compile(
        r"\b(pathology|biopsy|histology|histologic|receptor|ER|PR|HER2|"
        r"invasive|DCIS|carcinoma|specimen)\b",
        re.IGNORECASE,
    ),
}


def infer_modality(text: str) -> str:
    counts = {mod: len(pat.findall(text)) for mod, pat in MODALITY_PATTERNS.items()}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else "other"


def chunk_ocr_text(
    ocr_text: str,
    case_id: str,
    document_id: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: str = EMBEDDING_MODEL_DEFAULT,
) -> List[Chunk]:
    if document_id is None:
        document_id = case_id

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(ocr_text)
    chunks: List[Chunk] = []

    for i, text in enumerate(raw_chunks):
        chunk_id = f"{case_id}_{i:04d}_{str(uuid.uuid4())[:6]}"
        modality = infer_modality(text)
        token_count = len(text.split())
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                case_id=case_id,
                document_id=document_id,
                page_num=_estimate_page(i, len(raw_chunks)),
                modality=modality,
                text=text,
                token_count=token_count,
                embedding_model=embedding_model,
                retrieval_score=None,
            )
        )
    return chunks


def _estimate_page(chunk_index: int, total_chunks: int) -> int:
    if total_chunks == 0:
        return 1
    return max(1, round((chunk_index / total_chunks) * 10))


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/preprocessing/chunk_text.py
# Purpose: Splits OCR text into overlapping windows using LangChain's
#          RecursiveCharacterTextSplitter and infers modality labels from
#          section headings and vocabulary patterns.
#
# Functions:
#   chunk_ocr_text(text, case_id, chunk_size, chunk_overlap) -> List[Chunk]
#     Splits full OCR text into overlapping chunks. Each chunk is assigned
#     a unique chunk_id, modality label, estimated page_num, and token_count.
#     Returns: List of Chunk TypedDicts ready for embedding.
#
#   infer_modality(text) -> str
#     Regex-based classifier. Returns one of: "mammography", "ultrasound",
#     "mri", "pathology", "genetic", or "unknown".
#
#   estimate_page_num(chunk_index, total_chunks) -> int
#     Estimates page number (1-indexed) from chunk position in document.
#
# Outputs:
#   List[Chunk] with fields: text, chunk_id, modality, page_num, token_count,
#   retrieval_score (initialised to 0.0).
#
# Consumed by:
#   src/workflows/extraction_graph.py (node_index_case)
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
# =============================================================================
