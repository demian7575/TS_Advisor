from scipy import sparse
from ts_advisor.models import compare_baseline_models, make_baseline_specs, matrix_min
from ts_advisor.persistence import ModelBundle, save_model_bundle, load_model_bundle
from ts_advisor.inference import predict_one


def test_tfidf_only_features_are_non_negative(tiny_matrices):
    assert matrix_min(tiny_matrices.X_train_text) >= 0


def test_multinomial_nb_spec_uses_tfidf_only(tiny_config):
    specs = make_baseline_specs(tiny_config)
    nb_specs = [(name, feature_set) for name, _, feature_set in specs if name == "Multinomial NB"]
    assert nb_specs == [("Multinomial NB", "TF-IDF only")]


def test_model_comparison_returns_model_and_feature_set(tiny_matrices, tiny_config):
    results, models = compare_baseline_models(tiny_matrices, tiny_config)
    assert {"model", "feature_set", "val_accuracy", "macro_f1", "weighted_f1", "time_s", "warnings"}.issubset(results.columns)
    assert "Multinomial NB" in set(results["model"])
    assert set(results.loc[results["model"] == "Multinomial NB", "feature_set"]) == {"TF-IDF only"}
    assert models


def test_saved_model_bundle_loads_and_inference_does_not_fail(tiny_matrices, tiny_config):
    results, models = compare_baseline_models(tiny_matrices, tiny_config)
    best = results.iloc[0]
    bundle = ModelBundle(
        model=models[best["name"]],
        selected_model_name=best["name"],
        selected_feature_set_name=best["feature_set"],
        artifacts=tiny_matrices.artifacts,
        config=tiny_config.to_dict(),
        feature_metadata=tiny_matrices.artifacts.feature_metadata,
        validation_summary=best.to_dict(),
    )
    save_model_bundle(bundle, tiny_config.model_path)
    loaded = load_model_bundle(tiny_config.model_path)
    assert loaded.selected_model_name == bundle.selected_model_name
    pred = predict_one(bundle=loaded, heading="UE traffic stops after handover", observation="Cell disabled alarm", priority="B")
    assert "prediction" in pred
