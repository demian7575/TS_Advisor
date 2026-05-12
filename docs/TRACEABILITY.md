# TS Advisor Notebook-to-Module Traceability

The original notebook (`YanxingLiu_TS_Advisor.ipynb`, also copied to `notebooks/YanxingLiu_TS_Advisor.ipynb`) remains the source-of-truth reference. This table maps the notebook behavior into the refactored modules.

| Original notebook section | Refactored implementation | Preserved? | Notes / intentional differences |
|---|---|---:|---|
| Step 1: CSV loading with C engine, latin-1, quoted multiline fields | `ts_advisor.data.load_csv` | Yes | Uses the same `engine='c'`, `quotechar`, `doublequote`, `encoding='latin-1'`, and bad-line compatibility handling. |
| Run mode configuration | `ts_advisor.config.TSAdvisorConfig`, `RUN_MODE_CONFIGS` | Yes | Keeps fast/balanced/full TF-IDF budgets, n-grams, CV-grid metadata, and slow-model flag. Core refactor documents SBERT as not enabled by default to avoid heavy optional dependencies. |
| Utilities: text cleaning and section extraction | `ts_advisor.text` | Yes | Regexes and fallback behavior are carried over from the notebook. |
| Step 2: deduplicate, drop null MHO, rare-class handling | `ts_advisor.data.prepare_supervised_frame` | Yes | Same ID column, target column, `MIN_SAMPLES=100`, and `DROP_OTHER=False` defaults. |
| Step 4b: engineered features | `ts_advisor.features.add_engineered_features` | Yes | Keyword flags, board type, SW release, temporal, routing, duplicate, product hierarchy, catalog, and 3GPP alarm features are reproduced. Missing columns are tolerated for inference and tests. |
| Step 4: random stratified or temporal split | `ts_advisor.data.split_frame` | Yes | Default remains 60/20/20 random stratified with seed 42. |
| Per-section TF-IDF | `ts_advisor.features.build_feature_matrices` | Yes | Preserves separate heading/effect/config/cause vectorizers, section budgets, stop words, n-gram config, min/max document frequencies, and train-only Answer cause feature. |
| Priority, categorical, engineered numeric/categorical matrices | `ts_advisor.features.build_feature_matrices` | Yes | Preserves priority ordinal map, scaling, one-hot encoding, and full matrix stacking order. |
| Step 5 baseline comparison | `ts_advisor.models.compare_baseline_models` | Yes | Preserves Logistic Regression TF-IDF-only/full, LinearSVM TF-IDF-only/full, and MultinomialNB TF-IDF-only only. Slow RF/MLP remain optional in full/slow mode. |
| Warning handling | `ts_advisor.models.compare_baseline_models` | Improved traceability | Training warnings are captured into the result table and progress logs instead of being globally hidden. |
| Model persistence | `ts_advisor.persistence.ModelBundle` | New support | Saves model, selected feature set, fitted vectorizers/encoders/scalers, label encoder, config, feature metadata, and validation summary. |
| Inference path | `ts_advisor.inference` | New support, behavior-aligned | Applies the same cleaning, section extraction, engineered features, and selected feature-set transformation. Answer cause is blank at inference, matching validation/test behavior. |
| Notebook control panel | `notebooks/00_start_here.ipynb` | New support | Calls modular code and avoids duplicating implementation logic. |
| CLI training/prediction | `scripts/train.py`, `scripts/predict.py` | New support | Provides reproducible non-notebook execution. |

## Known behavior differences

1. The refactor focuses on the notebook's classical ML baseline path. Later exploratory sections such as transformer fine-tuning, calibration plots, duplicate-detection demos, and permutation-importance plotting are documented but not fully automated in the first modular pass.
2. SBERT is not enabled in the core dependency set because it adds heavy optional dependencies and network/model-cache concerns. The full feature-set name therefore means TF-IDF + notebook engineered/structured features unless optional SBERT support is added later.
3. Training warnings are no longer globally suppressed; they are captured and surfaced in logs/results for review.
