from pathlib import Path
from .config import TSAdvisorConfig
from .data import load_csv, prepare_supervised_frame, split_frame
from .features import add_engineered_features, build_feature_matrices
from .logging import ProgressLogger
from .models import compare_baseline_models, get_feature_set
from .persistence import ModelBundle, save_model_bundle


def choose_best_model(results_df, trained_models):
    if results_df.empty:
        raise RuntimeError("No baseline model results were produced.")
    best = results_df.iloc[0]
    run_name = best["name"]
    return run_name, best["feature_set"], trained_models[run_name], best.to_dict()


def run_training_pipeline(csv_path, config=None, logger=None, save_bundle=True):
    config = config or TSAdvisorConfig()
    logger = logger or ProgressLogger()
    logger.stage("Load and prepare data")
    raw = load_csv(csv_path, config)
    logger.log(f"Loaded raw data: rows={raw.shape[0]}, columns={raw.shape[1]}")
    prepared = prepare_supervised_frame(raw, config, logger=logger)
    logger.stage("Feature engineering and split")
    engineered = add_engineered_features(prepared, config)
    logger.log(f"Added engineered features: rows={engineered.shape[0]}, columns={engineered.shape[1]}")
    train_df, val_df, test_df = split_frame(engineered, config, logger=logger)
    matrices = build_feature_matrices(train_df, val_df, test_df, config, logger=logger)
    results_df, trained_models = compare_baseline_models(matrices, config, logger=logger)
    best_name, feature_set, model, validation_summary = choose_best_model(results_df, trained_models)
    logger.log(f"Best baseline: {best_name} val={validation_summary.get('val_accuracy', float('nan')):.4f} macroF1={validation_summary.get('macro_f1', float('nan')):.4f}")
    bundle = ModelBundle(
        model=model,
        selected_model_name=best_name,
        selected_feature_set_name=feature_set,
        artifacts=matrices.artifacts,
        config=config.to_dict(),
        feature_metadata=matrices.artifacts.feature_metadata,
        validation_summary=validation_summary,
    )
    saved_path = None
    if save_bundle:
        Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
        saved_path = save_model_bundle(bundle, config.model_path)
        logger.log(f"Saved model bundle: {saved_path}")
    return {
        "raw": raw,
        "prepared": prepared,
        "engineered": engineered,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "matrices": matrices,
        "results_df": results_df,
        "trained_models": trained_models,
        "bundle": bundle,
        "saved_path": saved_path,
    }
