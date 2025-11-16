"""Streamlit dashboard for cardiovascular risk modeling."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import ensure_local_copy, load_cardio_dataframe
from src.synthetic_generator import FeatureConstraint, SyntheticCardioGenerator

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "cardio_random_forest.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

st.set_page_config(page_title="Cardio Federated Baseline", layout="wide")


@st.cache_data(show_spinner=False)
def load_dataset_sample(rows: int = 1000) -> pd.DataFrame:
    ensure_local_copy()
    df = load_cardio_dataframe(Path("data") / "cardio_train.csv")
    return df.sample(n=min(rows, len(df)), random_state=42)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        st.warning("Train the Random Forest model first by running src/train_random_forest.py")
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> Dict[str, float]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def overview_tab():
    st.title("Cardiovascular Disease Modeling")
    st.markdown(
        """
        This dashboard demonstrates the local baseline workflow for the federated
        healthcare analytics project. The pipeline:
        1. Downloads the Kaggle cardiovascular dataset.
        2. Engineers BMI/age features and trains a Random Forest classifier.
        3. Generates privacy-friendly synthetic cohorts.
        4. Prepares artifacts for future federated rounds.
        """
    )
    st.info(
        "Add your Kaggle credentials via environment variables (KAGGLE_USERNAME and"
        " KAGGLE_KEY) before running locally so the dataset download succeeds."
    )


def training_tab():
    st.header("Training Metrics")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Run `python -m src.train_random_forest` first.")
        return
    metric_cols = st.columns(len(metrics))
    for (name, value), col in zip(metrics.items(), metric_cols):
        col.metric(label=name.upper(), value=f"{value:.3f}")

    df = load_dataset_sample()
    st.subheader("Feature Snapshot")
    fig = px.histogram(df, x="age", nbins=30, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)


def synthetic_tab():
    st.header("Synthetic Data Generator")
    df = load_dataset_sample(rows=5000)
    generator = SyntheticCardioGenerator(df)
    count = st.slider("Rows", min_value=100, max_value=5000, value=1000, step=100)
    age_range = st.slider("Age (years)", min_value=20, max_value=80, value=(30, 60))
    constraints = {
        "age_years": FeatureConstraint(minimum=age_range[0], maximum=age_range[1]),
    }
    fake_df = generator.sample(count, constraints=constraints, random_state=7)
    st.dataframe(fake_df.head())
    st.download_button(
        label="Download CSV",
        data=fake_df.to_csv(index=False),
        file_name="synthetic_cardio.csv",
        mime="text/csv",
    )


def inference_tab():
    st.header("Inference Sandbox")
    model = load_model()
    if model is None:
        return
    uploaded = st.file_uploader("Upload patient batch (CSV)", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
    else:
        input_df = load_dataset_sample(rows=50).drop(columns=["cardio"])
        st.info("Using a random sample from the training data.")

    preds = model.predict_proba(input_df)[:, 1]
    result = input_df.copy()
    result["cardio_risk"] = preds
    st.dataframe(result.head())


def main():
    tab_overview, tab_train, tab_synth, tab_infer = st.tabs(
        ["Overview", "Training", "Synthetic Lab", "Inference"]
    )
    with tab_overview:
        overview_tab()
    with tab_train:
        training_tab()
    with tab_synth:
        synthetic_tab()
    with tab_infer:
        inference_tab()


if __name__ == "__main__":
    main()
