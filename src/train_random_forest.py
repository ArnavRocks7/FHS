"""Train a Random Forest baseline on the cardiovascular dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import ensure_local_copy, load_cardio_dataframe
from . import preprocessing as prep

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "cardio_random_forest.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REPORT_PATH = ARTIFACT_DIR / "classification_report.txt"


def train(test_size: float = 0.2, random_state: int = 42) -> None:
    ensure_local_copy()
    df = load_cardio_dataframe(Path("data") / "cardio_train.csv")
    X = prep.get_feature_frame(df)
    y = prep.get_target_series(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", prep.build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    with METRICS_PATH.open("w") as fp:
        json.dump(metrics, fp, indent=2)

    with REPORT_PATH.open("w") as fp:
        fp.write(classification_report(y_test, y_pred))

    print("Training complete. Artifacts saved to:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Metrics: {METRICS_PATH}")
    print(f"- Report: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cardiovascular Random Forest")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(test_size=args.test_size, random_state=args.seed)


if __name__ == "__main__":
    main()
