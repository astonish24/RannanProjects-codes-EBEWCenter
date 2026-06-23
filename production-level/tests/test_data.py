from pathlib import Path

import pandas as pd
import pytest

from ebewcenter_qa.data import load_qa_dataframe, split_dataframe
from ebewcenter_qa.exceptions import DataValidationError


def test_load_qa_dataframe_happy_path(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "question": ["What is PPE?"],
            "answers": ["personal protective equipment"],
            "context": ["PPE means personal protective equipment in construction."],
        }
    ).to_csv(csv_file, index=False)

    df = load_qa_dataframe(csv_file)

    assert set(df.columns) == {"id", "question", "context", "answer_idx"}
    assert df.iloc[0]["answer_idx"]["answer_start"][0] >= 0


def test_load_qa_dataframe_missing_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample.csv"
    pd.DataFrame({"question": ["Q"], "context": ["C"]}).to_csv(csv_file, index=False)

    with pytest.raises(DataValidationError):
        load_qa_dataframe(csv_file)


def test_split_dataframe_reproducible() -> None:
    df = pd.DataFrame({"id": [f"id-{i}" for i in range(10)]})
    train_a, val_a = split_dataframe(df, seed=1)
    train_b, val_b = split_dataframe(df, seed=1)

    assert train_a.equals(train_b)
    assert val_a.equals(val_b)
