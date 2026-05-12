import re
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from .text import clean_text, clean_text_keep_alphanum, extract_observation_sections, extract_cause_of_fault, build_text_column

NUMERIC_FEATURE_COLUMNS = [
    "flag_crash", "flag_alarm", "flag_cell_disabled", "flag_restart", "flag_upgrade", "flag_race_cond", "flag_log_traces",
    "flag_coli_output", "hour_of_day", "day_of_week", "reg_quarter", "reg_year", "log_routing_time", "waiting_ratio",
    "auto_routed", "is_duplicate", "num_duplicates", "rstate_major", "cross_component",
]
NEW_CATEGORICAL_COLUMNS = ["board_type", "sw_release", "prod_family", "prod_gen", "prod_tech", "alarm_category", "alarm_severity"]

PRODUCT_CATALOG = {
    "KDU137848": ("Baseband_6630", "G2", "NR"),
    "KDU1370015": ("Baseband_6648", "G2", "LTE_NR"),
    "KDU137925": ("Baseband_5216", "G2", "LTE"),
    "KRC161842": ("IRU1649", "G3", "NR"),
    "KRC161549": ("RRUS2217", "G2", "LTE"),
    "KRD901206": ("AIR6449", "G3", "NR"),
    "KRC161756": ("Radio4449", "G3", "NR"),
    "KRC161707": ("RRU88430", "G2", "LTE"),
}
GPP_ALARM_MAP = {
    "License Key File Fault": ("LicenseManagement", "critical"),
    "Autonomous Mode Activated": ("LicenseManagement", "major"),
    "Emergency Unlock": ("LicenseManagement", "major"),
    "Resource Activation Timeout": ("RadioResource", "major"),
    "SW Error": ("Software", "minor"),
    "Cell.*[Dd]isabled": ("RadioResource", "major"),
    "RI.*[Ll]ink.*[Dd]own": ("Transport", "major"),
    "CPRI.*[Ff]ault": ("Transport", "major"),
    "Synchronization.*[Ll]ost": ("Synchronization", "critical"),
}


def make_one_hot_encoder():
    try:
        return OneHotEncoder(sparse=True, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse_output=True, handle_unknown="ignore")


def lookup_product_catalog(product_no):
    if product_no is None or pd.isna(product_no):
        return ("unknown", "unknown", "unknown")
    clean = str(product_no).replace(" ", "").replace("/", "")
    for prefix, info in PRODUCT_CATALOG.items():
        if clean.startswith(prefix):
            return info
    return ("unknown", "unknown", "unknown")


def extract_3gpp_alarm_features(text):
    if text is None or pd.isna(text):
        return ("none", "none")
    for pattern, (cat, sev) in GPP_ALARM_MAP.items():
        if re.search(pattern, str(text), re.IGNORECASE):
            return (cat, sev)
    return ("unknown", "unknown")


def add_engineered_features(df, config):
    """Notebook Step 4b engineered features, kept simple and deterministic."""
    df = df.copy()
    for col in [config.observation_column, config.registered_column, "routing_time", "routing_waiting_time", "total_routing_time", "routed", "General.Is duplicate TR", "General.Duplicate TRs", "Faulty product.Product no & R-State", "Faulty product.Product no", "Node level.Product no"]:
        if col not in df.columns:
            df[col] = np.nan
    obs = df[config.observation_column].fillna("")
    df["flag_crash"] = obs.str.contains(r"crash|pmd|core.?dump|program.?restart", case=False, regex=True).astype(int)
    df["flag_alarm"] = obs.str.contains(r"\bAL\b|alarm", case=False, regex=True).astype(int)
    df["flag_cell_disabled"] = obs.str.contains(r"cell.*disabled|disabled.*cell|CELLDISABLED", case=False, regex=True).astype(int)
    df["flag_restart"] = obs.str.contains(r"restart|reboot|cold.?start|warm.?start", case=False, regex=True).astype(int)
    df["flag_upgrade"] = obs.str.contains(r"upgrade|UP\s+\d{2}|R\d{2}[A-Z]\d+", case=False, regex=True).astype(int)
    df["flag_race_cond"] = obs.str.contains(r"race.?condition|deadlock|mutex|synchroni", case=False, regex=True).astype(int)
    df["flag_log_traces"] = obs.str.contains(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", regex=True).astype(int)
    df["flag_coli_output"] = obs.str.contains(r"coli>|OFFLINE_", regex=True).astype(int)
    df["board_type"] = obs.str.extract(r"(BB\d{4}|AIR\d{4}|IRU\d{4}|RRU\d{4}|RDS\w+)", expand=False).fillna("unknown")
    df["sw_release"] = obs.str.extract(r"(\d{2}\.Q\d(?:\.\d)?)", expand=False).fillna("unknown")
    reg = pd.to_datetime(df[config.registered_column], errors="coerce")
    df["hour_of_day"] = reg.dt.hour.fillna(-1).astype(int)
    df["day_of_week"] = reg.dt.dayofweek.fillna(-1).astype(int)
    df["reg_quarter"] = reg.dt.quarter.fillna(-1).astype(int)
    df["reg_year"] = reg.dt.year.fillna(-1).astype(int)
    df["log_routing_time"] = np.log1p(pd.to_numeric(df["routing_time"], errors="coerce").fillna(0))
    df["waiting_ratio"] = pd.to_numeric(df["routing_waiting_time"], errors="coerce").fillna(0) / (pd.to_numeric(df["total_routing_time"], errors="coerce").fillna(0) + 1)
    df["auto_routed"] = (df["routed"] == True).astype(int)
    df["is_duplicate"] = (df["General.Is duplicate TR"] == True).astype(int)
    df["num_duplicates"] = df["General.Duplicate TRs"].astype(str).str.count(r"H[A-Z]\d+").fillna(0).astype(int)
    df["rstate_major"] = pd.to_numeric(df["Faulty product.Product no & R-State"].astype(str).str.extract(r"R(\d+)", expand=False), errors="coerce").fillna(-1)
    df["cross_component"] = (df["Faulty product.Product no"].fillna("") != df["Node level.Product no"].fillna("")).astype(int)
    catalog = df["Faulty product.Product no"].apply(lookup_product_catalog)
    df["prod_family"] = catalog.apply(lambda x: x[0])
    df["prod_gen"] = catalog.apply(lambda x: x[1])
    df["prod_tech"] = catalog.apply(lambda x: x[2])
    alarms = df[config.observation_column].apply(extract_3gpp_alarm_features)
    df["alarm_category"] = alarms.apply(lambda x: x[0])
    df["alarm_severity"] = alarms.apply(lambda x: x[1])
    return df


def _obs_sections(frame, config):
    secs = frame[config.observation_column].apply(extract_observation_sections)
    return pd.DataFrame(list(secs), index=frame.index)


def encode_priority(dataframe, config):
    priority_map = {"A": 1, "B": 2, "C": 3}
    if config.priority_column not in dataframe.columns:
        return np.zeros((len(dataframe), 1))
    return dataframe[config.priority_column].map(priority_map).fillna(0).values.reshape(-1, 1)

@dataclass
class FeatureArtifacts:
    section_vectorizers: Dict[str, TfidfVectorizer]
    priority_scaler: StandardScaler
    categorical_encoder: OneHotEncoder
    numeric_scaler: StandardScaler
    engineered_categorical_encoder: OneHotEncoder
    label_encoder: LabelEncoder
    numeric_columns: List[str]
    engineered_categorical_columns: List[str]
    base_categorical_columns: List[str]
    feature_metadata: Dict

@dataclass
class FeatureMatrices:
    X_train_text: object
    X_val_text: object
    X_test_text: object
    X_train_full: object
    X_val_full: object
    X_test_full: object
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    artifacts: FeatureArtifacts
    train_texts: List[str]
    val_texts: List[str]
    test_texts: List[str]


def build_feature_matrices(train_df, val_df, test_df, config, logger=None):
    if logger:
        logger.stage("Feature extraction")
    train_obs, val_obs, test_obs = _obs_sections(train_df, config), _obs_sections(val_df, config), _obs_sections(test_df, config)
    sections = {
        "heading": (int(config.max_features * 0.20), train_df[config.heading_column].fillna("").apply(clean_text).tolist(), val_df[config.heading_column].fillna("").apply(clean_text).tolist(), test_df[config.heading_column].fillna("").apply(clean_text).tolist()),
        "effect": (int(config.max_features * 0.45), train_obs["effect"].apply(clean_text).tolist(), val_obs["effect"].apply(clean_text).tolist(), test_obs["effect"].apply(clean_text).tolist()),
        "config": (int(config.max_features * 0.15), train_obs["config"].apply(clean_text_keep_alphanum).tolist(), val_obs["config"].apply(clean_text_keep_alphanum).tolist(), test_obs["config"].apply(clean_text_keep_alphanum).tolist()),
        "cause": (int(config.max_features * 0.20), train_df[config.answer_column].apply(extract_cause_of_fault).apply(clean_text).tolist(), [""] * len(val_df), [""] * len(test_df)),
    }
    vectorizers, Xtr_parts, Xva_parts, Xte_parts = {}, [], [], []
    for name, (budget, tr, va, te) in sections.items():
        tv = TfidfVectorizer(max_features=max(budget, 100), stop_words="english", ngram_range=config.ngram_range, min_df=2, max_df=0.95)
        try:
            Xtr = tv.fit_transform(tr)
        except ValueError:
            tv = TfidfVectorizer(max_features=max(budget, 100), stop_words=None, ngram_range=(1, 1), min_df=1)
            Xtr = tv.fit_transform([x if x else "empty" for x in tr])
        Xva, Xte = tv.transform(va), tv.transform(te)
        vectorizers[name] = tv
        Xtr_parts.append(Xtr); Xva_parts.append(Xva); Xte_parts.append(Xte)
        if logger:
            logger.log(f"Built {name} TF-IDF: X_train_{name}={Xtr.shape}")
    X_train_text, X_val_text, X_test_text = hstack(Xtr_parts).tocsr(), hstack(Xva_parts).tocsr(), hstack(Xte_parts).tocsr()
    priority_scaler = StandardScaler()
    X_train_ord = priority_scaler.fit_transform(encode_priority(train_df, config))
    X_val_ord = priority_scaler.transform(encode_priority(val_df, config))
    X_test_ord = priority_scaler.transform(encode_priority(test_df, config))
    cat_cols = list(config.categorical_columns)
    for frame in [train_df, val_df, test_df]:
        for col in cat_cols + NUMERIC_FEATURE_COLUMNS + NEW_CATEGORICAL_COLUMNS + [config.heading_column, config.observation_column, config.answer_column]:
            if col not in frame.columns:
                frame[col] = np.nan
    categorical_encoder = make_one_hot_encoder()
    X_train_cat = categorical_encoder.fit_transform(train_df[cat_cols].fillna("unknown"))
    X_val_cat = categorical_encoder.transform(val_df[cat_cols].fillna("unknown"))
    X_test_cat = categorical_encoder.transform(test_df[cat_cols].fillna("unknown"))
    numeric_scaler = StandardScaler()
    X_train_num = numeric_scaler.fit_transform(train_df[NUMERIC_FEATURE_COLUMNS].fillna(0))
    X_val_num = numeric_scaler.transform(val_df[NUMERIC_FEATURE_COLUMNS].fillna(0))
    X_test_num = numeric_scaler.transform(test_df[NUMERIC_FEATURE_COLUMNS].fillna(0))
    engineered_categorical_encoder = make_one_hot_encoder()
    X_train_new_cat = engineered_categorical_encoder.fit_transform(train_df[NEW_CATEGORICAL_COLUMNS].fillna("unknown"))
    X_val_new_cat = engineered_categorical_encoder.transform(val_df[NEW_CATEGORICAL_COLUMNS].fillna("unknown"))
    X_test_new_cat = engineered_categorical_encoder.transform(test_df[NEW_CATEGORICAL_COLUMNS].fillna("unknown"))
    X_train_full = hstack([X_train_text, csr_matrix(X_train_ord), X_train_cat, csr_matrix(X_train_num), X_train_new_cat]).tocsr()
    X_val_full = hstack([X_val_text, csr_matrix(X_val_ord), X_val_cat, csr_matrix(X_val_num), X_val_new_cat]).tocsr()
    X_test_full = hstack([X_test_text, csr_matrix(X_test_ord), X_test_cat, csr_matrix(X_test_num), X_test_new_cat]).tocsr()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["target"])
    y_val = label_encoder.transform(val_df["target"])
    y_test = label_encoder.transform(test_df["target"])
    metadata = {"text_shape": X_train_text.shape, "full_shape": X_train_full.shape, "tfidf_sections": {k: v.vocabulary_.__len__() for k, v in vectorizers.items()}, "feature_sets": ["TF-IDF only", "TF-IDF + All"]}
    if logger:
        logger.log(f"Built TF-IDF features: X_train_text={X_train_text.shape}")
        logger.log(f"Built full features: X_train_full={X_train_full.shape}")
    artifacts = FeatureArtifacts(vectorizers, priority_scaler, categorical_encoder, numeric_scaler, engineered_categorical_encoder, label_encoder, NUMERIC_FEATURE_COLUMNS, NEW_CATEGORICAL_COLUMNS, cat_cols, metadata)
    return FeatureMatrices(X_train_text, X_val_text, X_test_text, X_train_full, X_val_full, X_test_full, y_train, y_val, y_test, artifacts, build_text_column(train_df, config, True), build_text_column(val_df, config, False), build_text_column(test_df, config, False))
