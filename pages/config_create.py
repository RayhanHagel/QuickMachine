import streamlit as st
from pathlib import Path
from utilities.model_hyperparams import get_json
import pickle
import pandas as pd
from utilities.save_config import save_config, collect_config
from utilities.registry import SCALER_MAP, ENCODER_MAP
import numpy as np



def update_test():
    st.session_state.test_split = 100 - st.session_state.train_split


def update_train():
    st.session_state.train_split = 100 - st.session_state.test_split


def delete_keys(keys:list):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]


def check_key(key:str):
    if key in st.session_state:
        if st.session_state[key]:
            return True
    return False


def check_key_value(key:str, value:str):
    if check_key(key):
        if st.session_state[key] == value:
            return True
    return False


def create_grid(parent, rows, cols):
    grid = []
    for r in range(rows):
        row_cols = parent.columns(cols)
        grid.append(row_cols)
    return grid


def read_pickle_file(path:str):
    with (open(path, "rb")) as openfile:
        while True:
            try:
                return pickle.load(openfile)
            except EOFError:
                break


def specific_model_hypertune(parent, library_chosen, model_chosen, model_type):
    # Get Hyperparameter
    parent.space("xxsmall")
    parent.markdown(f":orange[**{model_chosen} Configuration**]")
    params = model_options[st.session_state.target_chosen][library_chosen][model_chosen]["params"]
    hyperparameters = list(params.keys())
    
    # Make The Hyperparameter Grid
    column_amount = 2
    row_amount = (len(hyperparameters)+column_amount-1) // column_amount
    grid = create_grid(parent, row_amount, column_amount)

    # Render Hyperparameter
    for index, name in enumerate(hyperparameters):
        current_row = index//column_amount
        current_col = index%column_amount
        hyperparam_container = grid[current_row][current_col].container(border=True, height="stretch")
        hyperparam_container.markdown(f":red[{name}]")
        criteria = params[name]
        is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in criteria)
        if len(criteria) == 2 and is_numeric:
            is_float = any(isinstance(v, float) for v in criteria)
            more_columns = hyperparam_container.columns(2)
            step = 0.0001 if is_float else 1
            fmt = "%.4f" if is_float else None
            min_limit = -1.0 if is_float else -1
            more_columns[0].number_input(label="Min Value", min_value=min_limit, value=float(criteria[0]) if is_float else criteria[0], key=f"{model_type}_{model_chosen}_{name}_min_{index}", step=step, format=fmt)
            more_columns[1].number_input(label="Max Value", min_value=min_limit, value=float(criteria[1]) if is_float else criteria[1], key=f"{model_type}_{model_chosen}_{name}_max_{index}", step=step, format=fmt)
        else:
            hyperparam_container.multiselect("Options", options=criteria, default=criteria, key=f"{model_type}_{model_chosen}_{name}_option_{index}")


@st.cache_data(show_spinner="Loading and processing dataset...")
def load_and_preprocess_data(config_path):
    feature_pd = pd.DataFrame()
    target_pd = pd.DataFrame()
    
    pickle_object = read_pickle_file(config_path)
    
    # Read and Concat Data
    for group in pickle_object:
        if group["type"] in ["CSV", "Excel"]:
            item = group["item_list"][0]
            item.seek(0)
            
            if item.name.endswith(".csv"):
                loaded_item = pd.read_csv(item)
            elif item.name.endswith((".xlsx", ".xls")):
                loaded_item = pd.read_excel(item, engine="openpyxl")
            
            feature = loaded_item[group["feature_column"]]
            target = loaded_item[group["target_column"]]
            
            feature_pd = pd.concat([feature_pd, feature], axis=1)
            target_pd = pd.concat([target_pd, target], axis=1)
        elif group["type"] in ["Image", "CIF"]:
            print("Coming Soon! TBA!")

    # Coerce Types to Numeric where possible
    for col in feature_pd.columns:
        if feature_pd[col].dtype == 'object':
            feature_pd[col] = feature_pd[col].replace(r'^\s+$', np.nan, regex=True)
        try:
            feature_pd[col] = pd.to_numeric(feature_pd[col], errors='raise')
        except (ValueError, TypeError):
            continue
            
    for col in target_pd.columns:
        if target_pd[col].dtype == 'object':
            target_pd[col] = target_pd[col].replace(r'^\s+$', np.nan, regex=True)
        try:
            target_pd[col] = pd.to_numeric(target_pd[col], errors='raise')
        except (ValueError, TypeError):
            continue
            
    # Split into Int and Cat
    feature_int_pd = feature_pd.select_dtypes(include=["number"])
    feature_cat_pd = feature_pd.select_dtypes(exclude=["number"])
    return feature_pd, target_pd, feature_int_pd, feature_cat_pd


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Check If File Configuration Exists
configured_data = [file for file in Path(f"./configured_data").iterdir() if file.is_file()]
if configured_data == []:
    st.header(":red[Create a file configuration first]")
    st.stop()


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# File Configuration
st.subheader("File Configurations")
fc = st.container(border=True)


## Select File Configuration
fc.markdown(":orange[**Select File Configuration**]")
fc.selectbox(label="empty", options=configured_data, label_visibility="collapsed", index=None, key="chosen_configuration")


## Read and Process File Configuration (Cached for speed)
if check_key("chosen_configuration"):
    feature_pd, target_pd, feature_int_pd, feature_cat_pd = load_and_preprocess_data(st.session_state.chosen_configuration)


## Check Missing Value
if not check_key("chosen_configuration"):
    delete_keys(["missing_value_processing"])
else:
    combined_dataframe = pd.concat([feature_pd, target_pd], axis=1)
    missing_data = combined_dataframe.isnull().sum()
    cell_total = combined_dataframe.size
    cell_missing_count = missing_data.sum()
    cell_missing = (cell_missing_count / cell_total) * 100
    row_total = len(combined_dataframe.index)
    row_missing_count = combined_dataframe.isnull().any(axis=1).sum()
    row_missing = (row_missing_count / row_total) * 100
    if cell_missing_count > 0:
        fc.markdown(f":orange[**Detected Missing Value in Dataset**]")
        fc.caption(f"Missing cells: {cell_missing_count} ({cell_missing:.2f}%)")
        fc.caption(f"Missing rows: {row_missing_count} ({row_missing:.2f}%)")
        fc.pills(label="empty", options=["Drop Rows", "Impute (Mean)", "Impute (Median)", "Impute (Mode)"], key="missing_value_processing", selection_mode="single", label_visibility="collapsed")


## Show Pandas Dataframe
if not check_key("chosen_configuration"):
    delete_keys(["show_data"])
else:
    fc.space("xxsmall")
    fc.markdown(":orange[**Selected Data Information**]")
    fc.write(f"Amount of data points: :red[{len(target_pd.index)}]")
    fc.pills(label="Show the data", options=["Numerical Feature", "Non-Numerical Feature", "Target"], selection_mode="multi", key="show_data")
    if check_key("show_data"):
        if "Numerical Feature" in st.session_state.show_data:
            fc.write(":red[Numerical Feature]")
            fc.write(feature_int_pd.head(n=3))
        if "Non-Numerical Feature" in st.session_state.show_data:
            fc.write(":red[Non-Numerical Feature]")
            fc.write(feature_cat_pd.head(n=3))
        if "Target" in st.session_state.show_data:
            fc.write(":red[Target]")
            fc.write(target_pd.head(n=3))


## Scaler for Numerical Feature
if not (check_key("chosen_configuration") and not feature_int_pd.empty):
    delete_keys(["scaler_chosen"])
else:
    fc.space("xxsmall")
    fc.markdown(":orange[**Select Scaler for Numerical Features**]")
    scaler_options = list(SCALER_MAP.keys()) 
    fc.pills(label="empty", options=scaler_options, selection_mode="single", label_visibility="collapsed", key="scaler_chosen")


## Categorical Data Encoding for Non Numerical Feature
## Categorical Data Encoding for Non Numerical Feature
if not (check_key("chosen_configuration") and not feature_cat_pd.empty):
    delete_keys(["categorical_enc_chosen"])
else:
    fc.space("xxsmall")
    fc.markdown(":orange[**Select Categorical Encoder for Non-Numerical Features**]")
    categorical_enc_keys = list(ENCODER_MAP.keys())
    fc.pills(label="empty", options=categorical_enc_keys, selection_mode="single", label_visibility="collapsed", key="categorical_enc_chosen")


## Train Test Split
fc.space("xxsmall")
fc.markdown(":orange[**Train Test Split Percentage**]")
train_test_column = fc.columns(spec=2)
train_test_column[0].number_input(label="Train-Val (%)", min_value=1, max_value=99, step=1, value=80, on_change=update_test, key="train_split")
train_test_column[1].number_input(label="Test (%)", min_value=1, max_value=99, step=1, value=20, on_change=update_train, key="test_split")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Model Configuration
st.space("small")
st.subheader("Model Configurations")
mc = st.container(border=True)


## General Model Configuration
mc.markdown(":orange[**General Model Configuration**]")
model_col = mc.columns(spec=2)
model_col[0].number_input(label="Random State", min_value=0, max_value=1000, step=1, value=42, key="random_state")
model_col[0].number_input(label="Training Epoch", min_value=100, max_value=5000, step=50, value=500, key="train_epoch")
model_col[1].number_input(label="K-Fold Amount", min_value=1, max_value=10, step=1, value=5, key="kfold_amount")
model_col[1].number_input(label="Batch Size", min_value=16, max_value=256, step=16, value=32, key="batch_size")


## Target Configuration
if not check_key("chosen_configuration"):
    delete_keys(["target_chosen"])
else:
    mc.space("xxsmall")
    mc.markdown(":orange[**Select Target Type**]")
    target_option = ["Regression", "Classification"]
    mc.pills(label="empty", options=target_option, selection_mode="single", label_visibility="collapsed", default=None, key="target_chosen")
    target_is_categorical = not target_pd.select_dtypes(exclude=["number"]).empty
    target_is_numerical = not target_pd.select_dtypes(include=["number"]).empty
    if target_is_categorical and target_is_numerical:
        st.toast(body=":red[Target must only be either numerical or categorical]", duration="infinite")
        st.stop()
    elif (target_is_numerical and st.session_state.target_chosen == "Classification") or (target_is_categorical and st.session_state.target_chosen == "Regression"):
        st.toast(body=":red[The chosen target configuration is invalid]", duration="infinite")
        st.stop()


## Stacking Configuration
model_options = get_json()
if not check_key("target_chosen"):
    delete_keys(["use_stacking"])
else:
    mc.space("xxsmall")
    mc.markdown(":orange[**Use Model Stacking**]")
    mc.pills(label="empty", options=["Use", "Ignore"], selection_mode="single", label_visibility="collapsed", key="use_stacking")


### Use Stacking
if not check_key_value("use_stacking", "Use"):
    delete_keys(["base_model_chosen", "meta_library_chosen", "meta_model_chosen"])
else:
    # Base Model (Shows All Model)
    mc.space("xxsmall")
    mc.markdown(":orange[**Select Base Model**]")
    base_model_keys = []
    for library in model_options[st.session_state.target_chosen].keys():
        base_model_keys.extend(list(model for model in model_options[st.session_state.target_chosen][library].keys()))
    mc.multiselect(label="empty", options=base_model_keys, label_visibility="collapsed", key="base_model_chosen")
    
    # Meta Learner (Pick Library First -> Choose Model)
    mc.space("xxsmall")
    mc.markdown(":orange[**Select Meta Learner**]")
    meta_learner_col = mc.columns(spec=2)
    meta_library_option = ["scikit-learn", "tensorflow", "pytorch", "others"]
    meta_learner_col[0].selectbox(label="empty", options=meta_library_option, label_visibility="collapsed", key="meta_library_chosen", index=None)
    if check_key("meta_library_chosen"):
        meta_model_option_keys = list(model_options[st.session_state.target_chosen][st.session_state.meta_library_chosen].keys())
        meta_learner_col[1].selectbox(label="empty", options=meta_model_option_keys, label_visibility="collapsed", key="meta_model_chosen", index=None)
    
    # Meta Learner Data Configuration
    mc.space("xxsmall")
    mc.markdown(":orange[**Meta Learner Data Configuration**]")
    mc.selectbox(label="Meta-Model Passthrough", options=["Disabled", "Enabled"], key="stacking_passthrough")


#### Base Learner Configurations
if check_key("base_model_chosen") and check_key("stacking_passthrough"):
    mc.divider()
    mc.subheader(":red[Base Learner Configurations]")
    for model_chosen in st.session_state.base_model_chosen:
        library_chosen = [i for i, j in model_options[st.session_state.target_chosen].items() if model_chosen in j]
        specific_model_hypertune(mc, library_chosen[0], model_chosen, "base")


#### Meta Learner Configuration
if check_key("meta_model_chosen") and check_key("stacking_passthrough"):
    mc.divider()
    mc.subheader(":red[Meta Learner Configurations]")
    specific_model_hypertune(mc, st.session_state.meta_library_chosen, st.session_state.meta_model_chosen, "meta")


### Doesn't Use Stacking
#### Library Configuration
if not check_key_value("use_stacking", "Ignore"):
    delete_keys(["library_chosen"])
else:
    library_option = ["scikit-learn", "tensorflow", "pytorch", "others"]
    mc.space("xxsmall")
    mc.markdown(":orange[**Select Model Library**]")
    mc.pills(label="empty", options=library_option, selection_mode="single", label_visibility="collapsed", key="library_chosen")


#### Choose Model
if not check_key("library_chosen"):
    delete_keys(["model_chosen"])
else:
    mc.space("xxsmall")
    mc.markdown(":orange[**Select Machine Learning Model**]")
    model_option_keys = list(model_options[st.session_state.target_chosen][st.session_state.library_chosen].keys())
    model_chosen = mc.pills(label="empty", options=model_option_keys, selection_mode="single", label_visibility="collapsed", key="model_chosen")

#### Specific Model Configuration
if check_key("model_chosen"):
    specific_model_hypertune(mc, st.session_state.library_chosen, model_chosen, "base")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Tuning Configuration
st.space("small")
st.subheader("Tuner Configurations")
tc = st.container(border=True)


## Hypertuner Library
hypertuner_options = ["KT Bayesian Optimization", "Optuna"]
tc.markdown(":orange[**Select Hypertuner Library**]")
tc.pills(label="empty", options=hypertuner_options, selection_mode="single", label_visibility="collapsed", key="hypertuner_chosen")


## Hypertuner Config
tc.space("xxsmall")
tc.markdown(":orange[**Tuning Configuration**]")
tuning_column = tc.columns(spec=2)
tuning_column[0].number_input(label="Trial Amount", min_value=1, max_value=200, step=1, value=50, key="hypertune_trial")
tuning_column[1].number_input(label="Tuning Epoch", min_value=10, max_value=1000, step=10, value=100, key="hypertune_epoch")


### Objective Config
objective_column = tc.columns(spec=2)
current_target = st.session_state.get("target_chosen", "Regression")
if current_target == "Regression":
    objective_options = ["val_loss", "val_mae"]
    default_direction = 0
else:
    objective_options = ["val_accuracy", "f1_score", "val_loss"]
    default_direction = 1
objective_column[0].selectbox("Optimization Objective", objective_options, index=0, key="tuning_objective")
objective_column[1].selectbox("Direction", ["minimize", "maximize"], index=default_direction, key="tuning_direction")


### Other Configs
advanced_column = tc.columns(2)
advanced_column[0].number_input("Timeout (Seconds)", min_value=0, value=3600, key="tuning_timeout")
advanced_column[1].number_input("Initial Random Trials", min_value=2, max_value=50, value=5, key="tuning_initial_points")
tc.selectbox("Early Stopping Mode", ["Enabled", "Disabled"], index=0, key="use_early_stopping")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #


# Save Configuration and Dataset
st.space("small")
st.subheader("Save Configuration")
sc = st.container(border=True)
sc.text_input(label="empty", label_visibility="collapsed", placeholder="Enter config name", key="config_name")
sc.button("Validate", width='stretch', type="primary", key="save_button")
if check_key("save_button") and check_key("config_name") and check_key("chosen_configuration"):
    try:
        config = collect_config(feature_int=feature_int_pd, feature_cat=feature_cat_pd, target=target_pd)
        path = save_config(save_name=st.session_state.config_name, feature_int=feature_int_pd, feature_cat=feature_cat_pd, target=target_pd)
        sc.success(f"Configuration saved to `{path}`")
        sc.write(config)
    except ValueError as e:
        sc.error(str(e))


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #