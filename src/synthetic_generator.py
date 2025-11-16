"""Synthetic data generator for cardiovascular features."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from . import preprocessing as prep
from .data_loader import ensure_local_copy, load_cardio_dataframe

DEFAULT_SYNTHETIC_PATH = Path("artifacts/synthetic_cardio.csv")


@dataclass
class FeatureConstraint:
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    categories: Optional[Iterable[int]] = None


class SyntheticCardioGenerator:
    def __init__(self, dataframe: pd.DataFrame):
        engineered = prep.engineer_features(dataframe)
        self.features = prep.feature_columns()
        self.numeric_stats = {
            col: {
                "mean": engineered[col].mean(),
                "std": engineered[col].std(ddof=0) or 1.0,
                "min": engineered[col].min(),
                "max": engineered[col].max(),
            }
            for col in prep.numeric_features()
        }
        self.categorical_probs = {
            col: (engineered[col].value_counts(normalize=True).to_dict())
            for col in prep.categorical_features()
        }
        self.target_prob = engineered["cardio"].mean()

    def sample(
        self,
        n_rows: int,
        constraints: Optional[Mapping[str, FeatureConstraint]] = None,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        constraints = constraints or {}
        numeric_df = {
            col: self._sample_numeric(col, n_rows, constraints.get(col), rng)
            for col in prep.numeric_features()
        }
        categorical_df = {
            col: self._sample_categorical(col, n_rows, constraints.get(col), rng)
            for col in prep.categorical_features()
        }
        target = (rng.random(n_rows) < self.target_prob).astype(int)

        df = pd.DataFrame({**numeric_df, **categorical_df})
        df["cardio"] = target
        df["age"] = (df["age_years"] * 365.25).astype(int)
        df["ap_hi"] = df["systolic_bp"].round().astype(int)
        df["ap_lo"] = df["diastolic_bp"].round().astype(int)
        df["height"] = df["height"].round().astype(int)
        df["weight"] = df["weight"].round(1)
        ordered_cols = [
            "age",
            "gender",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "cardio",
        ]
        return df[ordered_cols]

    def _sample_numeric(
        self,
        column: str,
        n_rows: int,
        constraint: Optional[FeatureConstraint],
        rng: np.random.Generator,
    ) -> np.ndarray:
        stats = self.numeric_stats[column]
        samples = rng.normal(stats["mean"], stats["std"], size=n_rows)
        samples = np.clip(samples, stats["min"], stats["max"])
        if constraint:
            if constraint.minimum is not None:
                samples = np.maximum(samples, constraint.minimum)
            if constraint.maximum is not None:
                samples = np.minimum(samples, constraint.maximum)
        return samples

    def _sample_categorical(
        self,
        column: str,
        n_rows: int,
        constraint: Optional[FeatureConstraint],
        rng: np.random.Generator,
    ) -> np.ndarray:
        probs = self.categorical_probs[column]
        categories = list(probs.keys())
        p = np.array([probs[c] for c in categories])
        p = p / p.sum()
        if constraint and constraint.categories:
            allowed = [c for c in categories if c in set(constraint.categories)]
            if not allowed:
                raise ValueError(f"No allowed categories left for {column}")
            mask = [c in allowed for c in categories]
            p = p[mask]
            categories = allowed
            p = p / p.sum()
        return rng.choice(categories, size=n_rows, p=p)


def generate_to_csv(
    dataframe: pd.DataFrame,
    output_path: Path,
    n_rows: int,
    random_state: Optional[int] = None,
) -> Path:
    generator = SyntheticCardioGenerator(dataframe)
    fake_df = generator.sample(n_rows=n_rows, random_state=random_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic cardio data")
    parser.add_argument("-n", "--rows", type=int, default=1000, help="Number of rows to generate")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("-s", "--seed", type=int, default=13)
    args = parser.parse_args()

    ensure_local_copy()
    df = load_cardio_dataframe(Path("data") / "cardio_train.csv")
    output = generate_to_csv(df, args.output, args.rows, args.seed)
    print(f"Synthetic data saved to {output}")


if __name__ == "__main__":
    main()
