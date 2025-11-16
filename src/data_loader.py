"""Utilities for downloading and loading the cardiovascular dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

DATASET_ID = "sulianova/cardiovascular-disease-dataset"
DATASET_FILENAME = "cardio_train.csv"
DATA_DIRECTORY = Path("data")


def load_cardio_dataframe(local_file: Optional[Path | str] = None) -> pd.DataFrame:
    """Return the cardiovascular disease dataframe.

    Parameters
    ----------
    local_file:
        Optional explicit path to a CSV file. If omitted, KaggleHub is used to
        fetch ``cardio_train.csv`` automatically. The CSV uses ``;`` as a
        delimiter, so we read it accordingly.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be downloaded automatically and ``local_file`` is
        not provided.
    """

    if local_file is not None:
        csv_path = Path(local_file)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        return pd.read_csv(csv_path, sep=";")

    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET_ID,
            DATASET_FILENAME,
        )
        return df
    except Exception as exc:  # pragma: no cover - depends on Kaggle creds
        raise FileNotFoundError(
            "Dataset not found. Provide `cardio_train.csv` via the `local_file` "
            "argument or place it under the `data/` directory."
        ) from exc


def ensure_local_copy(destination_dir: Path | None = None) -> Path:
    """Download ``cardio_train.csv`` into ``destination_dir`` and return its path."""

    destination_dir = destination_dir or DATA_DIRECTORY
    destination_dir.mkdir(parents=True, exist_ok=True)
    local_path = destination_dir / DATASET_FILENAME

    if local_path.exists():
        return local_path

    df = load_cardio_dataframe()
    df.to_csv(local_path, sep=";", index=False)
    return local_path
