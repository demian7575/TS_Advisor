import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path, config):
    kwargs = dict(sep=",", quotechar='"', doublequote=True, engine="c", encoding=config.csv_encoding, low_memory=False)
    try:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False, **kwargs)


def _ensure_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def prepare_supervised_frame(df, config, logger=None):
    """Notebook Step 2: deduplicate, drop null target, and merge/drop rare classes."""
    df = df.copy()
    _ensure_columns(df, [config.id_column, config.target_column])
    if logger:
        logger.log(f"Raw frame: rows={len(df)}, columns={df.shape[1]}")
    before = len(df)
    if config.id_column in df.columns:
        df = df.drop_duplicates(subset=config.id_column, keep="first").copy()
    if logger:
        logger.log(f"Deduplicated by {config.id_column}: rows={len(df)} (removed={before-len(df)})")
    null_target = df[config.target_column].isna().sum()
    df = df[df[config.target_column].notna()].copy()
    if logger:
        logger.log(f"Dropped null targets: dropped={null_target}, rows={len(df)}")
    counts = df[config.target_column].value_counts()
    frequent = counts[counts >= config.min_samples].index
    if config.drop_other:
        before = len(df)
        df = df[df[config.target_column].isin(frequent)].copy()
        df["target"] = df[config.target_column]
        action = f"dropped rare classes rows={before-len(df)}"
    else:
        df["target"] = df[config.target_column].where(df[config.target_column].isin(frequent), other="other")
        action = "merged rare classes into 'other'"
    if logger:
        logger.log(f"Prepared supervised frame: rows={len(df)}, classes={df['target'].nunique()} ({action})")
    return df


def split_frame(df, config, logger=None):
    if config.split_mode == "random_stratified":
        train_df, temp_df = train_test_split(df, test_size=(1 - config.train_size), random_state=config.random_seed, stratify=df["target"])
        val_fraction_of_temp = config.test_size / (config.validation_size + config.test_size)
        val_df, test_df = train_test_split(temp_df, test_size=val_fraction_of_temp, random_state=config.random_seed, stratify=temp_df["target"])
    elif config.split_mode == "temporal":
        reg = pd.to_datetime(df[config.registered_column].astype(str).str.replace(" - ", " ", regex=False), errors="coerce")
        df_sorted = df.assign(_reg=reg).sort_values("_reg").drop(columns="_reg").reset_index(drop=True)
        n = len(df_sorted)
        i_tr = int(n * config.train_size)
        i_va = int(n * (config.train_size + config.validation_size))
        train_df = df_sorted.iloc[:i_tr].copy()
        val_df = df_sorted.iloc[i_tr:i_va].copy()
        test_df = df_sorted.iloc[i_va:].copy()
    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode!r}")
    if logger:
        logger.log(f"Split frame: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df.copy(), val_df.copy(), test_df.copy()
