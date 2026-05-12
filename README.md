# TS Advisor Refactored Project

This repository refactors the original `YanxingLiu_TS_Advisor.ipynb` notebook into a structured, traceable Python project for classical ML trouble-report routing.

The first goal is faithful refactoring, not model-performance redesign. The original notebook is preserved at the repository root and copied under `notebooks/YanxingLiu_TS_Advisor.ipynb` as a reference.

## Project structure

```text
src/ts_advisor/          Modular Python package
  config.py             Centralized settings and run modes
  data.py               CSV loading, supervised-frame preparation, splitting
  text.py               Text cleaning and section extraction
  features.py           Engineered features and feature matrices
  models.py             Baseline model comparison and metrics
  persistence.py        Joblib model bundle save/load
  inference.py          Reusable prediction path
  pipeline.py           End-to-end training orchestration
scripts/train.py        CLI training entry point
scripts/predict.py      CLI prediction entry point
notebooks/00_start_here.ipynb  User-facing control-panel notebook
docs/TRACEABILITY.md    Mapping from notebook sections to modules
tests/                  Lightweight pytest coverage
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` is kept offline-friendly for this repository; `requirements.txt` lists the scientific Python runtime dependencies needed to train and test.

## Run the starter notebook

Open and run:

```text
notebooks/00_start_here.ipynb
```

The notebook imports the package, sets a `TSAdvisorConfig`, runs the modular training pipeline, and displays model-comparison results.

## Train from CLI

```bash
python scripts/train.py --csv data/raw/4G5G_trr_trs_20000.csv --run-mode fast
```

Useful options:

```bash
python scripts/train.py \
  --csv data/raw/4G5G_trr_trs_20000.csv \
  --run-mode fast \
  --model-out models/ts_advisor_model.joblib \
  --split-mode random_stratified
```

## Predict from CLI

```bash
python scripts/predict.py \
  --model models/ts_advisor_model.joblib \
  --heading "UE traffic stops after handover" \
  --observation "Traffic stopped after mobility procedure." \
  --priority "B"
```

## Preserved baseline model/feature combinations

The baseline comparison intentionally preserves the original notebook's model-specific feature sets:

- Logistic Regression with TF-IDF only
- Logistic Regression with TF-IDF + engineered/structured features
- Linear SVM with TF-IDF only
- Linear SVM with TF-IDF + engineered/structured features
- MultinomialNB with TF-IDF only

MultinomialNB is guarded against negative-valued feature matrices and is not run on the full mixed matrix.

## Model bundle contents

A saved `joblib` bundle includes:

- selected trained model
- selected feature set name
- section TF-IDF vectorizers
- fitted priority scaler
- fitted categorical encoder
- fitted engineered numeric scaler
- fitted engineered categorical encoder
- fitted label encoder
- centralized config
- feature metadata
- validation summary

## Tests

```bash
pytest
```

The lightweight tests cover text cleaning, section extraction, TF-IDF non-negativity, MultinomialNB feature-set safety, model-comparison columns, bundle reload, and inference after reload.

## Known limitations and future ideas

- This pass modularizes the classical ML baseline path first. Transformer/SBERT, calibration plots, duplicate detection, and permutation importance can be modularized in later tasks.
- SBERT is not a required dependency in the refactored core to keep installation simple and reproducible.
- The next performance-improvement pass can tune hyperparameters, add calibrated confidence routing, compare temporal splits, and revisit feature-group importance.
