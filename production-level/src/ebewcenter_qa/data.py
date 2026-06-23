"""Data loading and preprocessing utilities."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from .constants import REQUIRED_COLUMNS
from .exceptions import DataValidationError


def _find_answer_start(answer: str, context: str) -> int:
    idx = context.find(answer)
    if idx != -1:
        return idx
    match = SequenceMatcher(None, context, answer).find_longest_match(0, len(context), 0, len(answer))
    return match.a if match.size > 0 else -1


def load_qa_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load and validate the expected QA CSV schema."""
    if not csv_path.exists():
        raise DataValidationError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).dropna()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    records = []
    for idx, row in df.reset_index(drop=True).iterrows():
        answer_start = _find_answer_start(str(row["answers"]), str(row["context"]))
        records.append(
            {
                "id": f"sample-{idx}",
                "question": str(row["question"]).strip(),
                "context": str(row["context"]),
                "answer_idx": {"text": [str(row["answers"])], "answer_start": [answer_start]},
            }
        )
    return pd.DataFrame(records)


def split_dataframe(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible train/validation split."""
    train = df.sample(frac=train_ratio, random_state=seed)
    validation = df.drop(train.index)
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def squad_references(df: Iterable[Dict]) -> list[Dict]:
    """Build references in SQuAD metric shape."""
    return [{"id": row["id"], "answers": row["answer_idx"]} for row in df]
