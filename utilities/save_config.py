import pickle
import streamlit as st
import pandas as pd
from pathlib import Path


_IGNORE_KEYS = {
    # App & Navigation State
    "page_nav", 
    
    # data_create.py UI State
    "uploader_key", "item_group_list", "extension_to_search", 
    "input_type", "create_model_name", "validate_data_btn",
    
    # config_create.py UI State
    "show_data", "save_button", "config_name"
}

# Prefixes from dynamic data_create.py elements to ignore
_IGNORE_PREFIXES = (
    "feature_column_", "target_column_", "ignore_column_",
    "group_config_", "delete_btn_", "file_upload_"
)


def collect_config(feature_int: pd.DataFrame | None = None, feature_cat: pd.DataFrame | None = None, target: pd.DataFrame | None = None) -> dict:
    config = {}

    # Dynamically scoop up ALL session state variables
    for key, value in st.session_state.items():
        
        # 1. Ignore hidden Streamlit internal keys (they always start with '_' or 'FormSubmitter')
        if key.startswith("_") or key.startswith("FormSubmitter"):
            continue
            
        # 2. Ignore exact matches in our Blacklist
        if key in _IGNORE_KEYS:
            continue
            
        # 3. Ignore dynamic keys from the data_create page
        if any(key.startswith(prefix) for prefix in _IGNORE_PREFIXES):
            continue
            
        # If it passes the filters and has a value, save it!
        if value is not None:
            config[key] = value

    # Cleaned dataset
    if feature_int is not None and not feature_int.empty:
        config["dataset_feature_int"] = feature_int.reset_index(drop=True).copy()
    if feature_cat is not None and not feature_cat.empty:
        config["dataset_feature_cat"] = feature_cat.reset_index(drop=True).copy()
    if target is not None and not target.empty:
        config["dataset_target"] = target.reset_index(drop=True).copy()
    return config


def save_config(save_name: str | None = None, save_dir: str = "./saved_configs", feature_int: pd.DataFrame | None = None, feature_cat: pd.DataFrame | None = None, target: pd.DataFrame | None = None) -> Path:
    config = collect_config(feature_int=feature_int, feature_cat=feature_cat, target=target)
    if not config:
        raise ValueError("No configuration found in session state. Please complete the setup before saving.")

    # Require at minimum a file config and a target type
    if "chosen_configuration" not in config:
        raise ValueError("No file configuration selected. Please choose a file configuration first.")
    if "target_chosen" not in config:
        raise ValueError("No target type selected. Please choose Regression or Classification first.")

    # Prepare output directory
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{save_name}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(config, f)
    return output_path


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)