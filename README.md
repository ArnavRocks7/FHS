# Federated Healthcare Streamlit Toolkit

This repository now includes everything needed to build the local (VS Code / CLI)
baseline for the cardiovascular federated learning proof-of-concept:

- **`requirements.txt`** – reproducible dependency set for VS Code or Colab.
- **`src/`** – Python package containing dataset helpers, preprocessing, training,
  and the synthetic data generator.
- **`streamlit_app.py`** – privacy-preserving analytics dashboard that surfaces
  metrics, synthetic cohorts, and inference utilities.
- **`artifacts/`** – location where the trained Random Forest model, metrics, and
  generated CSVs are stored.

Once these artifacts exist you can federate the model (Flower/Opacus) and deploy
Streamlit; the instructions below focus on reaching that ready-to-federate state.

## 1. Environment setup (VS Code or terminal)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Colab?** The same commands work inside a notebook cell – install the
> `requirements.txt` and run the scripts with `!python -m src.train_random_forest`.

## 2. Configure Kaggle access

The dataset lives at [`sulianova/cardiovascular-disease-dataset`](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
KaggleHub handles downloading once the environment variables are exported:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_key"
```

Alternatively place `cardio_train.csv` (semicolon-delimited) inside the local
`data/` directory. The helper below matches the snippet you provided:

```python
from kagglehub import KaggleDatasetAdapter
import kagglehub

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "sulianova/cardiovascular-disease-dataset",
    "cardio_train.csv",
)
print(df.head())
```

## 3. Train the Random Forest baseline (step-by-step)

1. Ensure the data exists: `python -m src.data_loader` is not required because
   the training script downloads automatically via KaggleHub.
2. Run the training job:

   ```bash
   python -m src.train_random_forest
   ```

   What happens under the hood:

   - `src/data_loader.py` pulls `cardio_train.csv` and caches it under `data/`.
   - `src/preprocessing.py` engineers `age_years`, BMI, and renames the blood
     pressure columns.
   - A `RandomForestClassifier` (balanced class weights, 400 trees) trains on the
     engineered features.
   - Artifacts land in `artifacts/`:
     - `cardio_random_forest.joblib` – end-to-end pipeline (preprocessing + model).
     - `metrics.json` – accuracy/F1/ROC-AUC for the held-out fold.
     - `classification_report.txt` – precision/recall per class.

3. Inspect artifacts or re-run with custom seeds via for example
   `python -m src.train_random_forest --seed 7 --test-size 0.25`.

## 4. Generate synthetic cohorts

The `SyntheticCardioGenerator` bootstraps from real distributions to create fake
patients suitable for demos or unit tests.

```bash
python -m src.synthetic_generator \
  --rows 2000 \
  --output artifacts/synthetic_cardio.csv \
  --seed 7
```

Key behaviors:

- Numeric features sample from clipped normal distributions (respecting observed
  mins/maxes).
- Categoricals (cholesterol, glucose, lifestyle flags) sample using empirical
  probabilities.
- Constraints can be passed programmatically (e.g., forcing an age range) by
  using the `FeatureConstraint` class inside notebooks or tests.

## 5. Launch the Streamlit app

Once the model + metrics exist, the dashboard stitches everything together:

```bash
streamlit run streamlit_app.py
```

The app contains four tabs:

1. **Overview** – project narrative + Kaggle credential reminders.
2. **Training** – displays saved metrics and renders a quick feature histogram.
3. **Synthetic Lab** – interactively generates fake cohorts (age slider, CSV
   download).
4. **Inference** – loads the trained pipeline and scores uploaded CSVs or sample
   records.

## 6. Where this leads next

- Split `cardio_train.csv` into hospital shards and wire them to Flower clients
  for federated averaging.
- Wrap the client optimizers with Opacus to log differential privacy budgets.
- Push the artifacts produced here (`artifacts/*.joblib`, synthetic CSVs) to the
  Streamlit deployment or object storage so the UI always reflects the latest
  training round.

These steps give you a reproducible VS Code workflow **before** federating or
moving to Colab, matching your request for a local-first plan with a Streamlit
front-end.
