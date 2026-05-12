from dataclasses import dataclass
from typing import Any, Dict
import joblib

@dataclass
class ModelBundle:
    model: Any
    selected_model_name: str
    selected_feature_set_name: str
    artifacts: Any
    config: Dict
    feature_metadata: Dict
    validation_summary: Dict


def save_model_bundle(bundle: ModelBundle, path: str):
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path: str) -> ModelBundle:
    return joblib.load(path)
