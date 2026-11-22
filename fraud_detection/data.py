"""Load and validate the benchmark before separating its target."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EXPECTED_COLUMNS, FEATURES, RANDOM_STATE, TEST_SIZE


def validate_data(frame: pd.DataFrame, benchmark: bool = True) -> None:
    """Reject malformed data; benchmark mode also checks expected population."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}; got {list(frame.columns)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if not all(pd.api.types.is_numeric_dtype(frame[c]) for c in EXPECTED_COLUMNS):
        raise ValueError("All features and Class must be numeric")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or infinite values")
    if not set(frame['Class'].unique()).issubset({0, 1}) or frame['Class'].nunique() != 2:
        raise ValueError("Class must contain both 0 and 1 and no other values")
    if benchmark:
        counts = frame['Class'].value_counts()
        if len(frame) != 284807 or counts.get(0, 0) != 284315 or counts.get(1, 0) != 492:
            raise ValueError("Dataset does not match expected benchmark counts: "
                             f"rows={len(frame)}, class counts={counts.to_dict()}")


def load_data(path: Path) -> pd.DataFrame:
    """Read and validate the full original CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Place creditcard.csv in data/.")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read dataset {path}: {exc}") from exc
    validate_data(frame)
    return frame


def split_data(frame: pd.DataFrame):
    """Use labels only to stratify the benchmark split, never as features."""
    return train_test_split(frame[FEATURES], frame['Class'], test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=frame['Class'])
