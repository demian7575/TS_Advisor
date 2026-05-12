from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

RUN_MODE_CONFIGS = {
    "fast": {
        "max_features": 3000,
        "ngram_range": (1, 1),
        "cv_folds": 2,
        "learning_sizes": 4,
        "grid_C": [0.1, 1.0],
        "grid_cw": [None, "balanced"],
        "grid_loss": ["squared_hinge"],
        "slow_models": False,
        "use_sbert": False,
    },
    "balanced": {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "cv_folds": 2,
        "learning_sizes": 5,
        "grid_C": [0.1, 1.0, 5.0],
        "grid_cw": [None, "balanced"],
        "grid_loss": ["hinge", "squared_hinge"],
        "slow_models": False,
        "use_sbert": False,  # documented deviation: no heavy optional SBERT in core refactor
    },
    "full": {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "cv_folds": 3,
        "learning_sizes": 8,
        "grid_C": [0.01, 0.1, 0.5, 1.0, 5.0],
        "grid_cw": [None, "balanced"],
        "grid_loss": ["hinge", "squared_hinge"],
        "slow_models": True,
        "use_sbert": False,
    },
}


def get_run_mode_config(run_mode: str) -> Dict:
    if run_mode not in RUN_MODE_CONFIGS:
        raise ValueError(f"Unknown run_mode {run_mode!r}. Expected one of {sorted(RUN_MODE_CONFIGS)}")
    return dict(RUN_MODE_CONFIGS[run_mode])


@dataclass
class TSAdvisorConfig:
    run_mode: str = "fast"
    random_seed: int = 42
    split_mode: str = "random_stratified"
    train_size: float = 0.60
    validation_size: float = 0.20
    test_size: float = 0.20
    min_samples: int = 100
    drop_other: bool = False
    id_column: str = "General.Eriref"
    target_column: str = "Faulty product.Design Responsible MHO"
    heading_column: str = "General.Heading"
    observation_column: str = "Observation.Observation"
    answer_column: str = "Answer.Answer"
    registered_column: str = "Registered"
    priority_column: str = "General.Submitter priority"
    categorical_columns: List[str] = field(default_factory=lambda: ["General.Superior MHO", "Observation.Fault type"])
    model_path: str = "models/ts_advisor_model.joblib"
    csv_encoding: str = "latin-1"
    logistic_max_iter: int = 1000
    linear_svm_max_iter: int = 2000
    linear_svm_tuning_max_iter: int = 3000
    logistic_c: float = 1.0
    linear_svm_c: float = 1.0
    enable_slow_models: Optional[bool] = None

    @property
    def mode(self) -> Dict:
        return get_run_mode_config(self.run_mode)

    @property
    def max_features(self) -> int:
        return int(self.mode["max_features"])

    @property
    def ngram_range(self) -> Tuple[int, int]:
        return tuple(self.mode["ngram_range"])

    @property
    def slow_models(self) -> bool:
        if self.enable_slow_models is not None:
            return bool(self.enable_slow_models)
        return bool(self.mode.get("slow_models", False))

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["mode"] = self.mode
        return out
