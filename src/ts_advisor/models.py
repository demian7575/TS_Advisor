import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def evaluate_model(model, X, y_true, name, y_train=None, X_train=None):
    y_pred = model.predict(X)
    out = {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "n_samples": len(y_true),
    }
    if X_train is not None and y_train is not None:
        train_pred = model.predict(X_train)
        out["train_accuracy"] = accuracy_score(y_train, train_pred)
        out["gap"] = out["train_accuracy"] - out["accuracy"]
    return out


def matrix_min(X):
    if sparse.issparse(X):
        if X.nnz == 0:
            return 0.0
        return float(X.data.min())
    return float(np.min(X))


def make_baseline_specs(config) -> List[Tuple[str, object, str]]:
    specs = [
        ("Logistic Regression", LogisticRegression(max_iter=config.logistic_max_iter, random_state=config.random_seed, C=config.logistic_c), "TF-IDF only"),
        ("Logistic Regression", LogisticRegression(max_iter=config.logistic_max_iter, random_state=config.random_seed, C=config.logistic_c), "TF-IDF + All"),
        ("Linear SVM", LinearSVC(random_state=config.random_seed, max_iter=config.linear_svm_max_iter, C=config.linear_svm_c), "TF-IDF only"),
        ("Linear SVM", LinearSVC(random_state=config.random_seed, max_iter=config.linear_svm_max_iter, C=config.linear_svm_c), "TF-IDF + All"),
        ("Multinomial NB", MultinomialNB(), "TF-IDF only"),
    ]
    if config.slow_models:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        specs.extend([
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=config.random_seed, n_jobs=-1, max_features="sqrt"), "TF-IDF + All"),
            ("MLP", MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=config.random_seed, early_stopping=True), "TF-IDF + All"),
        ])
    return specs


def get_feature_set(matrices, feature_set):
    if feature_set == "TF-IDF only":
        return matrices.X_train_text, matrices.X_val_text, matrices.X_test_text
    if feature_set == "TF-IDF + All":
        return matrices.X_train_full, matrices.X_val_full, matrices.X_test_full
    raise ValueError(f"Unknown feature_set: {feature_set!r}")


def fit_model_fresh(model, X, y):
    return model.__class__(**model.get_params()).fit(X, y)


def compare_baseline_models(matrices, config, logger=None):
    """Notebook Step 5 model comparison with model-specific feature sets preserved."""
    if logger:
        logger.stage("Baseline model comparison")
    rows: List[Dict] = []
    trained_models: Dict[str, object] = {}
    for model_name, model, feature_set in make_baseline_specs(config):
        run_name = f"{model_name} [{feature_set}]"
        Xtr, Xva, _ = get_feature_set(matrices, feature_set)
        if isinstance(model, MultinomialNB) and matrix_min(Xtr) < 0:
            msg = "Skipped: MultinomialNB requires non-negative features."
            rows.append({"model": model_name, "feature_set": feature_set, "name": run_name, "warnings": msg, "time_s": 0.0})
            if logger:
                logger.log(f"{run_name}: {msg}")
            continue
        if logger:
            logger.log(f"Training {model_name} [{feature_set}]...")
        t0 = time.time()
        caught = []
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            fitted = fit_model_fresh(model, Xtr, matrices.y_train)
            row = evaluate_model(fitted, Xva, matrices.y_val, run_name, y_train=matrices.y_train, X_train=Xtr)
            caught = [f"{w.category.__name__}: {w.message}" for w in warning_records if issubclass(w.category, Warning)]
        row.update({
            "model": model_name,
            "feature_set": feature_set,
            "val_accuracy": row.pop("accuracy"),
            "time_s": time.time() - t0,
            "warnings": "; ".join(caught),
        })
        rows.append(row)
        trained_models[run_name] = fitted
        if logger:
            warn_suffix = f" warnings={len(caught)}" if caught else ""
            logger.log(f"Finished {model_name} [{feature_set}]: train={row.get('train_accuracy', float('nan')):.3f} val={row['val_accuracy']:.3f} macroF1={row['macro_f1']:.3f}{warn_suffix}")
    results_df = pd.DataFrame(rows)
    if not results_df.empty and "val_accuracy" in results_df:
        results_df = results_df.sort_values("val_accuracy", ascending=False, na_position="last").reset_index(drop=True)
    return results_df, trained_models
