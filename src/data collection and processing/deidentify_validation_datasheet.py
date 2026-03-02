"""Utility script to remove direct identifiers from the LLM validation datasheet.

The script drops explicitly sensitive columns (MRN, patient initials, free text
comments, etc.) and replaces the named surgeon column with deterministic
surrogate identifiers so that analysts can still stratify by surgeon without
having access to the original names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Root of the repository, inferred from the script location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "merged_llm_summary_validation_datasheet.xlsx"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "merged_llm_summary_validation_datasheet_deidentified.xlsx"
)
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_PATH.with_suffix(".metadata.json")

# Explicit identifiers that must be removed entirely.
DIRECT_IDENTIFIER_COLUMNS = [
    "mrn",
    "patient_initials",
    "comments",
    "Unnamed: 0",
]

# Columns that should be pseudonymized so downstream analysis retains group info.
PSEUDONYMIZE_COLUMNS: Dict[str, str] = {
    "surgeon": "surgeon_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a de-identified copy of merged_llm_summary_validation_datasheet"
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the raw Excel file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination for the de-identified file (extension controls format)",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to write a JSON audit log describing the transformations",
    )
    return parser.parse_args()


def sequential_surrogate_ids(series: pd.Series, prefix: str) -> pd.Series:
    """Create deterministic surrogate IDs while avoiding leakage of original values."""

    mapping: Dict[str, str] = {}
    counter = 1

    def _transform(value):
        nonlocal counter
        if pd.isna(value):
            return pd.NA
        key = str(value).strip()
        if not key:
            return pd.NA
        if key not in mapping:
            mapping[key] = f"{prefix}{counter:03d}"
            counter += 1
        return mapping[key]

    return series.map(_transform)


def drop_direct_identifiers(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Drop identifier columns and report which ones were present."""

    present = [col for col in columns if col in df.columns]
    if present:
        df.drop(columns=present, inplace=True)
    return present


def pseudonymize_columns(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Replace sensitive columns with surrogate identifiers."""

    details: List[Dict[str, str]] = []
    for column, prefix in PSEUDONYMIZE_COLUMNS.items():
        if column not in df.columns:
            continue
        surrogate_column = f"{column}_id"
        df[surrogate_column] = sequential_surrogate_ids(df[column], prefix)
        df.drop(columns=[column], inplace=True)
        details.append(
            {
                "source_column": column,
                "new_column": surrogate_column,
                "prefix": prefix,
                "non_null_surrogates": int(df[surrogate_column].notna().sum()),
                "unique_surrogates": int(df[surrogate_column].nunique(dropna=True)),
            }
        )
    return details


def write_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        # Default to Excel when extension is .xlsx or unknown.
        df.to_excel(output_path, index=False)


def write_metadata(metadata: Dict[str, object], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()

    df = pd.read_excel(args.input_path)

    dropped_columns = drop_direct_identifiers(df, DIRECT_IDENTIFIER_COLUMNS)
    pseudonymized_details = pseudonymize_columns(df)

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "record_count": int(len(df)),
        "dropped_columns": dropped_columns,
        "pseudonymized_columns": pseudonymized_details,
    }

    write_dataframe(df, args.output_path)
    write_metadata(metadata, args.metadata_path)


if __name__ == "__main__":
    main()
