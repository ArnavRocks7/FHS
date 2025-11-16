"""Feature engineering helpers for the cardiovascular dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_TARGET_COLUMN = "cardio"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered features for modeling."""

    engineered = df.copy()
    engineered["age_years"] = (engineered["age"] / 365.25).clip(0, None)
    engineered["bmi"] = engineered["weight"] / ((engineered["height"] / 100) ** 2)
    engineered = engineered.rename(
        columns={
            "ap_hi": "systolic_bp",
            "ap_lo": "diastolic_bp",
        }
    )
    return engineered


def feature_columns() -> List[str]:
    return [
        "age_years",
        "height",
        "weight",
        "systolic_bp",
        "diastolic_bp",
        "bmi",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "gender",
    ]


def numeric_features() -> List[str]:
    return ["age_years", "height", "weight", "systolic_bp", "diastolic_bp", "bmi"]


def categorical_features() -> List[str]:
    return ["cholesterol", "gluc", "smoke", "alco", "active", "gender"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features()),
            ("cat", categorical_pipeline, categorical_features()),
        ]
    )


def get_target_series(df: pd.DataFrame) -> pd.Series:
    return df[RAW_TARGET_COLUMN].astype(int)


def get_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(df)[feature_columns()]
