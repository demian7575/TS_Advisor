import pandas as pd
from scipy.sparse import hstack, csr_matrix
from .config import TSAdvisorConfig
from .features import add_engineered_features, encode_priority
from .persistence import load_model_bundle
from .text import clean_text, clean_text_keep_alphanum, extract_observation_sections, extract_cause_of_fault


def _config_from_bundle(bundle):
    cfg = bundle.config if isinstance(bundle.config, dict) else {}
    allowed = TSAdvisorConfig.__dataclass_fields__.keys()
    return TSAdvisorConfig(**{k: v for k, v in cfg.items() if k in allowed})


def frame_from_tr(heading="", observation="", priority="", metadata=None, config=None):
    config = config or TSAdvisorConfig()
    row = {
        config.heading_column: heading,
        config.observation_column: observation,
        config.priority_column: priority,
        config.answer_column: "",
    }
    if metadata:
        row.update(metadata)
    return pd.DataFrame([row])


def transform_for_inference(df, bundle):
    config = _config_from_bundle(bundle)
    artifacts = bundle.artifacts
    df = add_engineered_features(df.copy(), config)
    secs = df[config.observation_column].apply(extract_observation_sections)
    obs = pd.DataFrame(list(secs), index=df.index)
    section_inputs = {
        "heading": df[config.heading_column].fillna("").apply(clean_text).tolist(),
        "effect": obs["effect"].apply(clean_text).tolist(),
        "config": obs["config"].apply(clean_text_keep_alphanum).tolist(),
        "cause": ["" for _ in range(len(df))],
    }
    X_sections = [artifacts.section_vectorizers[name].transform(section_inputs[name]) for name in ["heading", "effect", "config", "cause"]]
    X_text = hstack(X_sections).tocsr()
    if bundle.selected_feature_set_name == "TF-IDF only":
        return X_text
    X_ord = artifacts.priority_scaler.transform(encode_priority(df, config))
    for col in artifacts.base_categorical_columns + artifacts.numeric_columns + artifacts.engineered_categorical_columns:
        if col not in df.columns:
            df[col] = None
    X_cat = artifacts.categorical_encoder.transform(df[artifacts.base_categorical_columns].fillna("unknown"))
    X_num = artifacts.numeric_scaler.transform(df[artifacts.numeric_columns].fillna(0))
    X_new_cat = artifacts.engineered_categorical_encoder.transform(df[artifacts.engineered_categorical_columns].fillna("unknown"))
    return hstack([X_text, csr_matrix(X_ord), X_cat, csr_matrix(X_num), X_new_cat]).tocsr()


def predict_dataframe(df, bundle):
    X = transform_for_inference(df, bundle)
    y = bundle.model.predict(X)
    labels = bundle.artifacts.label_encoder.inverse_transform(y)
    out = pd.DataFrame({"prediction": labels})
    if hasattr(bundle.model, "predict_proba"):
        proba = bundle.model.predict_proba(X)
        out["confidence"] = proba.max(axis=1)
    return out


def predict_one(model_path=None, bundle=None, heading="", observation="", priority="", metadata=None):
    if bundle is None:
        bundle = load_model_bundle(model_path)
    config = _config_from_bundle(bundle)
    df = frame_from_tr(heading, observation, priority, metadata, config)
    return predict_dataframe(df, bundle).iloc[0].to_dict()
