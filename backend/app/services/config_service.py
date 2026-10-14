import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

from backend.app.config import CONFIGURED_DATA_DIR, SAVED_CONFIGS_DIR
from backend.app.utilities.registry import SCALER_MAP, ENCODER_MAP
from backend.app.utilities.model_hyperparams import get_json


def _read_pickle_file(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_dataframe_from_item(item: Any, group_type: str) -> pd.DataFrame:
    """Reads a DataFrame from diverse item representations."""
    # Check if item is a dict with path
    if isinstance(item, dict) and "path" in item:
        item_path = Path(item["path"])
        if item_path.suffix.lower() == ".csv":
            return pd.read_csv(item_path)
        else:
            return pd.read_excel(item_path, engine="openpyxl")
    
    # Check if item is a string/Path
    elif isinstance(item, (str, Path)):
        item_path = Path(item)
        if item_path.suffix.lower() == ".csv":
            return pd.read_csv(item_path)
        else:
            return pd.read_excel(item_path, engine="openpyxl")
            
    # Check if item has a read/seek method (e.g., Streamlit UploadedFile or file buffer)
    elif hasattr(item, "seek"):
        item.seek(0)
        name = getattr(item, "name", "").lower()
        if name.endswith(".csv") or group_type == "CSV":
            return pd.read_csv(item)
        else:
            return pd.read_excel(item, engine="openpyxl")
            
    raise ValueError(f"Unsupported item format: {type(item)}")


def load_and_preprocess_dataset(config_path_or_name: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw files from configured dataset, concatenates features and targets,
    and coerces numeric types.
    """
    target_path = Path(config_path_or_name)
    if not target_path.exists():
        # Check in CONFIGURED_DATA_DIR
        target_path = CONFIGURED_DATA_DIR / config_path_or_name
        if not target_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path_or_name}")

    pickle_object = _read_pickle_file(target_path)
    
    feature_pd = pd.DataFrame()
    target_pd = pd.DataFrame()

    for group in pickle_object:
        group_type = group.get("type", "")
        if group_type in ["CSV", "Excel"]:
            items = group.get("item_list", [])
            if not items:
                continue
            item = items[0]
            loaded_item = _load_dataframe_from_item(item, group_type)

            feature_cols = group.get("feature_column", [])
            target_cols = group.get("target_column", [])

            feature = loaded_item[feature_cols] if feature_cols else pd.DataFrame()
            target = loaded_item[target_cols] if target_cols else pd.DataFrame()

            feature_pd = pd.concat([feature_pd, feature], axis=1)
            target_pd = pd.concat([target_pd, target], axis=1)

    # Coerce Types to Numeric where possible
    for col in feature_pd.columns:
        if feature_pd[col].dtype == "object":
            feature_pd[col] = feature_pd[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        try:
            feature_pd[col] = pd.to_numeric(feature_pd[col], errors="raise")
        except (ValueError, TypeError):
            continue

    for col in target_pd.columns:
        if target_pd[col].dtype == "object":
            target_pd[col] = target_pd[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        try:
            target_pd[col] = pd.to_numeric(target_pd[col], errors="raise")
        except (ValueError, TypeError):
            continue

    feature_int_pd = feature_pd.select_dtypes(include=["number"])
    feature_cat_pd = feature_pd.select_dtypes(exclude=["number"])

    return feature_pd, target_pd, feature_int_pd, feature_cat_pd


def inspect_dataset_configuration(name: str) -> Dict[str, Any]:
    """
    Loads dataset, calculates missing values, detects columns, and returns
    full inspection metadata for the frontend.
    """
    feature_pd, target_pd, feature_int_pd, feature_cat_pd = load_and_preprocess_dataset(name)

    combined_df = pd.concat([feature_pd, target_pd], axis=1)
    missing_data = combined_df.isnull().sum()
    cell_total = combined_df.size
    cell_missing_count = int(missing_data.sum())
    cell_missing_pct = float((cell_missing_count / cell_total) * 100) if cell_total > 0 else 0.0

    row_total = len(combined_df.index)
    row_missing_count = int(combined_df.isnull().any(axis=1).sum())
    row_missing_pct = float((row_missing_count / row_total) * 100) if row_total > 0 else 0.0

    # Target classification check
    target_is_cat = not target_pd.select_dtypes(exclude=["number"]).empty
    target_is_num = not target_pd.select_dtypes(include=["number"]).empty
    target_nunique = int(target_pd.iloc[:, 0].nunique(dropna=True)) if not target_pd.empty else 0

    if target_is_cat or (target_nunique in [2, 3, 4, 5] and row_total >= 10):
        type_hint = "Classification"
    else:
        type_hint = "Regression"

    # Clean preview records for JSON
    def to_clean_records(df: pd.DataFrame, n: int = 5):
        if df.empty:
            return []
        sample = df.head(n).copy()
        sample = sample.fillna("NaN")
        return sample.to_dict(orient="records")

    return {
        "dataset_name": name,
        "total_rows": row_total,
        "feature_count": len(feature_pd.columns),
        "target_count": len(target_pd.columns),
        "numerical_features": feature_int_pd.columns.tolist(),
        "categorical_features": feature_cat_pd.columns.tolist(),
        "target_columns": target_pd.columns.tolist(),
        "target_cardinality": target_nunique,
        "missing_values": {
            "cell_missing_count": cell_missing_count,
            "cell_missing_pct": round(cell_missing_pct, 2),
            "row_missing_count": row_missing_count,
            "row_missing_pct": round(row_missing_pct, 2)
        },
        "target_type_hint": type_hint,
        "available_scalers": list(SCALER_MAP.keys()),
        "available_encoders": list(ENCODER_MAP.keys()),
        "previews": {
            "numerical": to_clean_records(feature_int_pd),
            "categorical": to_clean_records(feature_cat_pd),
            "target": to_clean_records(target_pd)
        }
    }


def save_pipeline_configuration(config_dict: Dict[str, Any], save_name: str) -> Path:
    """
    Saves pipeline configuration with embedded dataset into SAVED_CONFIGS_DIR / f"{save_name}.pkl".
    """
    chosen_conf = config_dict.get("chosen_configuration")
    if not chosen_conf:
        raise ValueError("chosen_configuration is required")

    # Load and embed the cleaned dataset
    feature_pd, target_pd, feature_int_pd, feature_cat_pd = load_and_preprocess_dataset(chosen_conf)
    
    config_to_save = dict(config_dict)
    if not feature_int_pd.empty:
        config_to_save["dataset_feature_int"] = feature_int_pd.reset_index(drop=True).copy()
    if not feature_cat_pd.empty:
        config_to_save["dataset_feature_cat"] = feature_cat_pd.reset_index(drop=True).copy()
    if not target_pd.empty:
        config_to_save["dataset_target"] = target_pd.reset_index(drop=True).copy()

    output_path = SAVED_CONFIGS_DIR / f"{save_name}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(config_to_save, f)

    return output_path


def list_saved_configurations() -> List[Dict[str, Any]]:
    """Lists all saved training configurations."""
    configs = []
    if not SAVED_CONFIGS_DIR.exists():
        return configs

    for file in sorted(SAVED_CONFIGS_DIR.glob("*.pkl")):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            has_embedded = (
                "dataset_feature_int" in data
                or "dataset_feature_cat" in data
                or "dataset_target" in data
            )

            row_count = 0
            num_cols = 0
            cat_cols = 0
            if has_embedded:
                if "dataset_target" in data and isinstance(data["dataset_target"], pd.DataFrame):
                    row_count = len(data["dataset_target"])
                if "dataset_feature_int" in data and isinstance(data["dataset_feature_int"], pd.DataFrame):
                    num_cols = len(data["dataset_feature_int"].columns)
                if "dataset_feature_cat" in data and isinstance(data["dataset_feature_cat"], pd.DataFrame):
                    cat_cols = len(data["dataset_feature_cat"].columns)

            configs.append({
                "name": file.stem,
                "file_name": file.name,
                "path": str(file),
                "created_at": file.stat().st_mtime,
                "target_chosen": data.get("target_chosen", "Regression"),
                "use_stacking": data.get("use_stacking", "Ignore"),
                "model_chosen": data.get("model_chosen"),
                "base_model_chosen": data.get("base_model_chosen", []),
                "meta_model_chosen": data.get("meta_model_chosen"),
                "hypertuner_chosen": data.get("hypertuner_chosen", "Optuna"),
                "hypertune_trial": data.get("hypertune_trial", 50),
                "train_split": data.get("train_split", 80),
                "test_split": data.get("test_split", 20),
                "kfold_amount": data.get("kfold_amount", 5),
                "scaler_chosen": data.get("scaler_chosen"),
                "categorical_enc_chosen": data.get("categorical_enc_chosen"),
                "has_embedded_dataset": has_embedded,
                "row_count": row_count,
                "num_features_count": num_cols,
                "cat_features_count": cat_cols
            })
        except Exception as e:
            configs.append({
                "name": file.stem,
                "file_name": file.name,
                "path": str(file),
                "error": str(e)
            })

    return configs


def get_saved_configuration(name: str) -> Optional[Dict[str, Any]]:
    """Loads a specific saved configuration by name (without full dataframes in payload)."""
    target_path = SAVED_CONFIGS_DIR / f"{name}.pkl"
    if not target_path.exists():
        target_path = SAVED_CONFIGS_DIR / name
        if not target_path.exists():
            return None

    with open(target_path, "rb") as f:
        data = pickle.load(f)

    # Sanitize config dict for JSON response (convert DataFrames to metadata)
    sanitized = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            sanitized[f"{k}_summary"] = {
                "shape": list(v.shape),
                "columns": v.columns.tolist()
            }
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            sanitized[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            sanitized[k] = float(v)
        else:
            sanitized[k] = v

    return sanitized


def delete_saved_configuration(name: str) -> bool:
    """Deletes a saved configuration by name."""
    target_path = SAVED_CONFIGS_DIR / f"{name}.pkl"
    if not target_path.exists():
        target_path = SAVED_CONFIGS_DIR / name
    
    if target_path.exists():
        target_path.unlink()
        return True
    return False
