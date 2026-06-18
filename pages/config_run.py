import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import optuna
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
from sklearn.base import clone
import keras_tuner as kt
from io import BytesIO


# Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Model Selection / Evaluation
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc
)
import matplotlib.pyplot as plt
import altair as alt
from sklearn.ensemble import (StackingRegressor, StackingClassifier)

# Utilities
from utilities.save_config import load_config
from utilities.model_hyperparams import get_json
from utilities.registry import SCALER_MAP, ENCODER_MAP, get_model_class
from optuna.trial import TrialState
from optuna.visualization import (
    plot_optimization_history, 
    plot_param_importances, 
    plot_parallel_coordinate
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_embedded = (
        "dataset_feature_int" in config
        or "dataset_feature_cat" in config
        or "dataset_target" in config
    )

    if has_embedded:
        feature_int = config.get("dataset_feature_int", pd.DataFrame())
        feature_cat = config.get("dataset_feature_cat", pd.DataFrame())
        target      = config.get("dataset_target",      pd.DataFrame())
        
        # 1. FIX: Removed the ', True' from the end of this line
        return feature_int, feature_cat, target   

    pickle_object = read_pickle(config["chosen_configuration"])
    feature_pd = pd.DataFrame()
    target_pd  = pd.DataFrame()

    for group in pickle_object:
        if group["type"] in ["CSV", "Excel"]:
            item = group["item_list"][0]
            item.seek(0)
            df = (
                pd.read_csv(item)
                if item.name.endswith(".csv")
                else pd.read_excel(item, engine="openpyxl")
            )
            feature_pd = pd.concat([feature_pd, df[group["feature_column"]]], axis=1)
            target_pd  = pd.concat([target_pd,  df[group["target_column"]]],  axis=1)

    feature_int = feature_pd.select_dtypes(include=["number"])
    feature_cat = feature_pd.select_dtypes(exclude=["number"])
    
    # 2. FIX: Removed the ', False' from the end of this line
    return feature_int, feature_cat, target_pd


def build_sklearn_model(
    model_name: str,
    hyperparams: dict,
    random_state: int,
    is_regression: bool = True,
):
    # Now uses the imported helper from registry!
    cls = get_model_class(model_name, is_regression)
    
    if cls is None:
        target_label = "Regression" if is_regression else "Classification"
        st.warning(f"Model **{model_name}** ({target_label}) is not mapped in the registry.")
        return None

    # --- LightGBM Random Forest Safeguard ---
    hyperparams = dict(hyperparams)  # defensive copy — never mutate caller's dict
    if "LGBM" in model_name or "LightGBM" in model_name:
        if hyperparams.get("boosting_type") == "rf":
            if hyperparams.get("subsample", 1.0) >= 1.0:
                hyperparams["subsample"] = 0.99
            if hyperparams.get("subsample_freq", 0) == 0:
                hyperparams["subsample_freq"] = 1
            if hyperparams.get("bagging_fraction", 1.0) >= 1.0:
                hyperparams["bagging_fraction"] = 0.99
            if hyperparams.get("bagging_freq", 0) == 0:
                hyperparams["bagging_freq"] = 1

    # SAFEGUARD: Use .get_params() to safely catch all hidden **kwargs
    try:
        dummy_model = cls()
        valid_keys = dummy_model.get_params().keys()
    except Exception:
        # Fallback just in case a custom model requires strict arguments
        valid_keys = hyperparams.keys()

    valid_params = {}
    if "random_state" in valid_keys:
        valid_params["random_state"] = random_state
    elif "random_seed" in valid_keys:
        valid_params["random_seed"] = random_state
        
    # --- NEW: Force maximum CPU usage for supported models ---
    if "n_jobs" in valid_keys:
        valid_params["n_jobs"] = -1
    elif "thread_count" in valid_keys: # Catch CatBoost's specific parameter
        valid_params["thread_count"] = -1

    # Transfer parameters without stripping them
    for k, v in hyperparams.items():
        if k in valid_keys:
            valid_params[k] = v

    try:
        return cls(**valid_params)
    except Exception as exc:
        st.error(f"Could not instantiate **{model_name}** with params `{valid_params}`.\n\nError: `{exc}`")
        return None


def build_preprocessor(
    int_cols: list[str],
    cat_cols: list[str],
    scaler_name: str | None,
    encoder_name: str | None,
    impute_strategy: str | None = None
) -> ColumnTransformer | None:
    """Build a ColumnTransformer for numeric/cat imputation + scaling + encoding."""
    transformers = []
    
    # 1. Handle Numerical Columns
    if int_cols:
        num_steps = []
        # Add Imputer
        if impute_strategy in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
            strat = "mean" if "Mean" in impute_strategy else ("median" if "Median" in impute_strategy else "most_frequent")
            num_steps.append(("imputer", SimpleImputer(strategy=strat)))
        
        # Add Scaler
        if scaler_name and scaler_name in SCALER_MAP:
            num_steps.append(("scaler", clone(SCALER_MAP[scaler_name])))
            
        if num_steps:
            transformers.append(("num", Pipeline(num_steps), int_cols))
        else:
            transformers.append(("num", "passthrough", int_cols))
            
    # 2. Handle Categorical / Text Columns
    if cat_cols:
        cat_steps = []
        # Add Imputer (Always use mode for text)
        if impute_strategy in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            
        # Add Encoder
        if encoder_name and encoder_name in ENCODER_MAP:
            cat_steps.append(("encoder", clone(ENCODER_MAP[encoder_name])))
        else:
            cat_steps.append(("encoder", clone(ENCODER_MAP["Ordinal Encoder"])))
            
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
            
    if not transformers:
        return None
        
    return ColumnTransformer(transformers, remainder="passthrough")


def display_metrics(y_true, y_pred, is_regression: bool, container, y_prob=None, key_suffix: str = ""):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if is_regression:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        cols = container.columns(3)
        cols[0].metric("MAE", f"{mae:.4f}")
        cols[1].metric("RMSE", f"{rmse:.4f}")
        cols[2].metric("R²", f"{r2:.4f}")
        
        # --- 1. Interactive Actual vs. Predicted (Altair) ---
        df_chart = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        min_val = float(min(y_true.min(), y_pred.min()))
        max_val = float(max(y_true.max(), y_pred.max()))
        
        scatter = alt.Chart(df_chart).mark_circle(size=60, opacity=0.5, color="#1f77b4").encode(
            x=alt.X("Actual", scale=alt.Scale(domain=[min_val, max_val])),
            y=alt.Y("Predicted", scale=alt.Scale(domain=[min_val, max_val])),
            tooltip=["Actual", "Predicted"]
        )
        line = alt.Chart(pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})).mark_line(color="red", strokeDash=[5, 5]).encode(x="x", y="y")
        
        container.altair_chart((scatter + line).properties(title="Actual vs. Predicted"), width='stretch')
        
        # Download Button: Actual vs. Predicted (.png)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, color="#1f77b4")
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted")
        ax.legend()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        plt.close(fig)
        container.download_button("📥 Download Scatter Plot (.png)", data=buf, file_name="scatter_plot.png", mime="image/png", key=f"dl_scatter_{key_suffix}")

        # --- 2. Interactive Residual Plot (Altair) ---
        container.markdown("#### 📉 Residual Analysis")
        df_residuals = pd.DataFrame({"Predicted": y_pred, "Residuals": y_true - y_pred})
        
        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[5, 5]).encode(y="y")
        residual_scatter = alt.Chart(df_residuals).mark_circle(size=60, opacity=0.5, color="#ff7f0e").encode(
            x=alt.X("Predicted", title="Predicted Values"),
            y=alt.Y("Residuals", title="Error (Actual - Predicted)"),
            tooltip=["Predicted", "Residuals"]
        )
        container.altair_chart((residual_scatter + zero_line).properties(title="Residual Plot"), width='stretch')

        # Download Button: Residual Plot (.png)
        fig_res, ax_res = plt.subplots(figsize=(6, 6))
        ax_res.scatter(y_pred, y_true - y_pred, alpha=0.5, color="#ff7f0e")
        ax_res.axhline(0, color='red', linestyle='--', lw=2)
        ax_res.set_xlabel("Predicted Values")
        ax_res.set_ylabel("Residuals (Actual - Predicted)")
        ax_res.set_title("Residual Plot")
        buf_res = BytesIO()
        fig_res.savefig(buf_res, format="png", bbox_inches="tight", dpi=300)
        buf_res.seek(0)
        plt.close(fig_res)
        container.download_button("📥 Download Residual Plot (.png)", data=buf_res, file_name="residual_plot.png", mime="image/png", key=f"dl_resid_{key_suffix}")

    else:
        # --- Classification Metrics Dashboard ---
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cols = container.columns(2)
        cols[0].metric("Accuracy", f"{acc:.4f}")
        cols[1].metric("Weighted F1", f"{f1:.4f}")
        
        container.markdown("**Classification Report:**")
        container.code(classification_report(y_true, y_pred, zero_division=0))
        
        # --- Confusion Matrix ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")
        container.pyplot(fig)
        
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        plt.close(fig)
        container.download_button("📥 Download Confusion Matrix (.png)", data=buf, file_name="confusion_matrix.png", mime="image/png", key=f"dl_cm_{key_suffix}")

        # --- ROC Curve (Only runs if probabilities were successfully passed) ---
        if y_prob is not None and len(np.unique(y_true)) == 2:
            container.markdown("#### 📈 ROC Curve")
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")
            
            container.pyplot(fig_roc)
            
            buf_roc = BytesIO()
            fig_roc.savefig(buf_roc, format="png", bbox_inches="tight", dpi=300)
            buf_roc.seek(0)
            plt.close(fig_roc)
            container.download_button("📥 Download ROC Curve (.png)", data=buf_roc, file_name="roc_curve.png", mime="image/png", key=f"dl_roc_{key_suffix}")


def optuna_suggest_hyperparams(trial, config: dict, model_type: str, model_name: str) -> dict:
    model_options = get_json()
    target = config.get("target_chosen", "Regression")
    
    library = None
    for lib, models in model_options[target].items():
        if model_name in models:
            library = lib
            break
    if library is None:
        return {}

    params_meta = model_options[target][library][model_name].get("params", {})
    resolved = {}

    for idx, (param_name, criteria) in enumerate(params_meta.items()):
        min_key = f"{model_type}_{model_name}_{param_name}_min_{idx}"
        max_key = f"{model_type}_{model_name}_{param_name}_max_{idx}"
        opt_key = f"{model_type}_{model_name}_{param_name}_option_{idx}"
        
        optuna_param_name = f"{model_type}_{model_name}_{param_name}"

        if min_key in config and max_key in config:
            lo, hi = config[min_key], config[max_key]
            
            if lo > hi:
                lo, hi = hi, lo
                
            if lo == hi:
                resolved[param_name] = trial.suggest_categorical(optuna_param_name, [lo])
            elif isinstance(lo, float) or isinstance(hi, float):
                resolved[param_name] = trial.suggest_float(optuna_param_name, float(lo), float(hi))
            else:
                resolved[param_name] = trial.suggest_int(optuna_param_name, int(lo), int(hi))
                
        elif opt_key in config:
            opts = config[opt_key]
            if len(opts) == 1:
                resolved[param_name] = trial.suggest_categorical(optuna_param_name, opts)
            elif len(opts) > 1:
                try:
                    opts = sorted(opts)
                except TypeError:
                    opts = sorted(opts, key=str)
                
                try:
                    resolved[param_name] = trial.suggest_categorical(optuna_param_name, opts)
                except ValueError as e:
                    if "dynamic value space" in str(e) or "distribution compatibility" in str(e):
                        st.error(f"🛑 **Search Space Conflict for `{param_name}`**\n\n"
                                 f"You are resuming an older tuning session, but the options for `{param_name}` have changed.\n\n"
                                 "**How to fix:** Delete the `optuna_studies.db` file inside your `saved_configs` folder to start fresh.")
                        st.stop()
                    else:
                        raise e

    return resolved


def kt_suggest_hyperparams(hp, config: dict, model_type: str, model_name: str) -> dict:
    """
    Dynamically maps the config min/max/options to Keras-Tuner hp suggestions.
    Includes safeguards for swapped min/max and dynamic search spaces.
    """
    model_options = get_json()
    target = config.get("target_chosen", "Regression")
    
    library = None
    for lib, models in model_options[target].items():
        if model_name in models:
            library = lib
            break
    if library is None:
        return {}

    params_meta = model_options[target][library][model_name].get("params", {})
    resolved = {}

    for idx, (param_name, criteria) in enumerate(params_meta.items()):
        min_key = f"{model_type}_{model_name}_{param_name}_min_{idx}"
        max_key = f"{model_type}_{model_name}_{param_name}_max_{idx}"
        opt_key = f"{model_type}_{model_name}_{param_name}_option_{idx}"
        
        kt_param_name = f"{model_type}_{model_name}_{param_name}"

        if min_key in config and max_key in config:
            lo, hi = config[min_key], config[max_key]
            
            if lo > hi:
                lo, hi = hi, lo
                
            if lo == hi:
                resolved[param_name] = hp.Fixed(kt_param_name, lo)
            elif isinstance(lo, float) or isinstance(hi, float):
                resolved[param_name] = hp.Float(kt_param_name, min_value=float(lo), max_value=float(hi))
            else:
                resolved[param_name] = hp.Int(kt_param_name, min_value=int(lo), max_value=int(hi))
                
        elif opt_key in config:
            opts = config[opt_key]
            if len(opts) == 1:
                resolved[param_name] = hp.Fixed(kt_param_name, opts[0])
            elif len(opts) > 1:
                try:
                    opts = sorted(opts)
                except TypeError:
                    opts = sorted(opts, key=str)
                resolved[param_name] = hp.Choice(kt_param_name, values=opts)

    return resolved


class SklearnCVTuner(kt.Tuner):
    """Custom Keras Tuner class to handle Scikit-Learn K-Fold Cross Validation."""
    def run_trial(self, trial, X_arr, y_arr, cv_splitter, config, is_regression, random_state, feature_int_cols, feature_cat_cols):
        hp = trial.hyperparameters
        cv_scores = []
        
        use_stacking = config.get("use_stacking") == "Use"
        use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled"
        
        # 1. Build Blueprint Model using KT hp object
        if use_stacking:
            fold_estimators = []
            for name in config.get("base_model_chosen", []):
                base_hp = kt_suggest_hyperparams(hp, config, "base", name)
                mdl = build_sklearn_model(name, base_hp, random_state, is_regression)
                if mdl is not None: fold_estimators.append((name, mdl))
                
            meta_name = config.get("meta_model_chosen")
            meta_hp = kt_suggest_hyperparams(hp, config, "meta", meta_name)
            meta_mdl = build_sklearn_model(meta_name, meta_hp, random_state, is_regression)
            
            StackingCls = StackingRegressor if is_regression else StackingClassifier
            blueprint_model = StackingCls(estimators=fold_estimators, final_estimator=meta_mdl, passthrough=use_passthrough)
        else:
            model_name = config.get("model_chosen")
            base_hp = kt_suggest_hyperparams(hp, config, "base", model_name)
            blueprint_model = build_sklearn_model(model_name, base_hp, random_state, is_regression)

        # 2. Cross-Validation Loop
        tuning_obj = config.get("tuning_objective", "val_loss")
        for train_idx, val_idx in cv_splitter.split(X_arr, y_arr):
            fold_preprocessor = build_preprocessor(
                feature_int_cols, feature_cat_cols, 
                config.get("scaler_chosen"), config.get("categorical_enc_chosen"),
                config.get("missing_value_processing")
            )
            
            current_fold_model = clone(blueprint_model)
            
            steps = []
            if fold_preprocessor:
                steps.append(("preprocessor", fold_preprocessor))
            steps.append(("model", current_fold_model))
            fold_pipeline = Pipeline(steps)
            
            fold_pipeline.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
            val_pred = fold_pipeline.predict(X_arr.iloc[val_idx])
            
            if is_regression:
                if tuning_obj in ["val_loss", "val_mae"]:
                    score = mean_absolute_error(y_arr.iloc[val_idx], val_pred)
                else:
                    score = r2_score(y_arr.iloc[val_idx], val_pred)
            else:
                if tuning_obj == "val_accuracy":
                    score = accuracy_score(y_arr.iloc[val_idx], val_pred)
                else:
                    score = f1_score(y_arr.iloc[val_idx], val_pred, average="weighted", zero_division=0)
            
            cv_scores.append(score)
            
        # 3. Send final metric back to KT Oracle
        mean_score = np.mean(cv_scores)
        self.oracle.update_trial(trial.trial_id, {tuning_obj: mean_score})


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: dict, config_name: str, eval_mode: str):
    is_regression = config.get("target_chosen") == "Regression"
    random_state = config.get("random_state", 42)
    kfold = config.get("kfold_amount", 5)
    train_pct = config.get("train_split", 80) / 100

    # ── 1-3. Data Loading, Preprocessing & Split ──────────────────────────────
    with st.status("⚙️ Preparing Data & Pipeline...", expanded=True) as prep_status:
        st.write("📂 Loading data...")
        # Note: Depending on your cleanup from earlier, this might unpack 3 or 4 variables.
        # Assuming you implemented the 3-variable cleanup:
        feature_int, feature_cat, target_pd = load_data(config)
        st.write(f"✅ Loaded **{len(target_pd)}** rows.")

        if not feature_cat.empty:
            feature_cat = feature_cat.astype(str)

        X = pd.concat([feature_int, feature_cat], axis=1)
        y = target_pd.squeeze()
        
        missing_strategy = config.get("missing_value_processing")

        if missing_strategy == "Drop Rows":
            x_cols = list(X.columns)
            y_cols = list(y.columns) if isinstance(y, pd.DataFrame) else [y.name]
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined[x_cols]
            y = combined[y_cols].squeeze()
            st.write(f"🗑️ Dropped rows with missing values. **{len(y)}** rows remaining.")

        if not is_regression and y.isnull().any():
            # Fill just for the stratify logic so it doesn't crash
            stratify_array = y.fillna(y.mode()[0])
        else:
            stratify_array = y if not is_regression else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=train_pct,
            random_state=random_state,
            stratify=stratify_array,
        )
        
        # Impute Target to prevent pipeline crash (X is handled natively by Pipeline)
        if missing_strategy in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
            if y_train.isnull().any() or y_test.isnull().any():
                if is_regression:
                    if missing_strategy == "Impute (Mean)":
                        fill_val = y_train.mean()
                    elif missing_strategy == "Impute (Median)":
                        fill_val = y_train.median()
                    else:
                        fill_val = y_train.mode()[0]
                else:
                    fill_val = y_train.mode()[0]
                
                y_train = y_train.fillna(fill_val)
                y_test = y_test.fillna(fill_val)
                y = y.fillna(fill_val)
                st.write(f"🩹 Imputed missing target values securely using training set statistics.")

        st.write("⚙️ Building global preprocessor...")
        preprocessor = build_preprocessor(
            list(feature_int.columns),
            list(feature_cat.columns),
            config.get("scaler_chosen"),
            config.get("categorical_enc_chosen"),
            missing_strategy
        )
        st.write("✅ Preprocessor ready.")
        
        # Once all steps finish, change the title, mark it complete, and collapse it!
        prep_status.update(label="✅ Data & Pipeline Ready!", state="complete", expanded=False)

    # ── 4. Hyperparameter Tuning (Pausable & Dashboard Ready) ─────────────────
    use_stacking = config.get("use_stacking") == "Use"
    tuner_choice = config.get("hypertuner_chosen", "Optuna")
    n_trials = config.get("hypertune_trial", 50)
    tuning_direction = config.get("tuning_direction", "maximize")
    
    st.markdown("---")
    st.markdown(f"#### 🔎 Running {tuner_choice} Tuning")
    
    if tuner_choice == "Optuna":
        db_path = Path("./saved_configs/optuna_studies.db")
        db_path.parent.mkdir(exist_ok=True)
        storage_name = f"sqlite:///{db_path}"
        
        st.info(f"💡 **Tip:** Open your terminal and run `optuna-dashboard {storage_name}` to view real-time progress.")
        
        cv_splitter = (
            KFold(n_splits=kfold, shuffle=True, random_state=random_state)
            if is_regression
            else StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
        )
        X_arr = X_train.reset_index(drop=True)
        y_arr = y_train.reset_index(drop=True)

        # UI Elements for Verbose Updates
        log_container = st.empty()
        chart_placeholder = st.empty()

        def objective(trial):
            cv_scores = []
            use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled" # <-- ADD THIS
            
            # 1. Create the blueprint for the trial model outside the loop
            if use_stacking:
                fold_estimators = []
                for name in config.get("base_model_chosen", []):
                    hp = optuna_suggest_hyperparams(trial, config, "base", name)
                    mdl = build_sklearn_model(name, hp, random_state, is_regression)
                    if mdl is not None: fold_estimators.append((name, mdl))
                    
                meta_name = config.get("meta_model_chosen")
                meta_hp = optuna_suggest_hyperparams(trial, config, "meta", meta_name)
                meta_mdl = build_sklearn_model(meta_name, meta_hp, random_state, is_regression)
                
                StackingCls = StackingRegressor if is_regression else StackingClassifier
                # --- NEW: Add passthrough=use_passthrough ---
                blueprint_model = StackingCls(estimators=fold_estimators, final_estimator=meta_mdl, passthrough=use_passthrough)
            else:
                model_name = config.get("model_chosen")
                hp = optuna_suggest_hyperparams(trial, config, "base", model_name)
                blueprint_model = build_sklearn_model(model_name, hp, random_state, is_regression)

            # 2. Loop through the folds
            for train_idx, val_idx in cv_splitter.split(X_arr, y_arr):
                fold_preprocessor = build_preprocessor(
                    list(feature_int.columns), list(feature_cat.columns), 
                    config.get("scaler_chosen"), config.get("categorical_enc_chosen"),
                    config.get("missing_value_processing")
                )
                
                # SAFEGUARD: Clone the blueprint so the model starts with 0 memory for this fold!
                current_fold_model = clone(blueprint_model)
                
                steps = []
                if fold_preprocessor:
                    steps.append(("preprocessor", fold_preprocessor))
                steps.append(("model", current_fold_model))
                fold_pipeline = Pipeline(steps)
                
                fold_pipeline.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
                val_pred = fold_pipeline.predict(X_arr.iloc[val_idx])
                
                # Read the objective from the Streamlit UI
                tuning_obj = config.get("tuning_objective", "val_loss")
                
                if is_regression:
                    if tuning_obj in ["val_loss", "val_mae"]:
                        # Calculate Error (Lower is better, matches 'minimize')
                        score = mean_absolute_error(y_arr.iloc[val_idx], val_pred)
                    else:
                        # Calculate R2 (Higher is better, matches 'maximize')
                        score = r2_score(y_arr.iloc[val_idx], val_pred)
                else:
                    if tuning_obj == "val_accuracy":
                        score = accuracy_score(y_arr.iloc[val_idx], val_pred)
                    else:
                        score = f1_score(y_arr.iloc[val_idx], val_pred, average="weighted", zero_division=0)
                
                cv_scores.append(score)
                
            return np.mean(cv_scores)

        # Fetch Advanced UI Settings
        timeout_sec = config.get("tuning_timeout", 3600)
        timeout = timeout_sec if timeout_sec > 0 else None
        
        initial_points = config.get("tuning_initial_points", 5)
        early_stopping = config.get("use_early_stopping", "Enabled") == "Enabled"
        
        # Configure Optuna's Sampler and Pruner based on UI
        sampler = optuna.samplers.TPESampler(n_startup_trials=initial_points, seed=random_state)
        pruner = optuna.pruners.MedianPruner() if early_stopping else optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=config_name, 
            storage=storage_name, 
            load_if_exists=True, 
            direction=tuning_direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # --- NEW: Immediately draw the past history if resuming a paused session ---
        valid_trials = [t for t in study.trials if t.value is not None]
        if valid_trials:
            df = pd.DataFrame({
                "Trial Score": [t.value for t in valid_trials]
            }, index=[t.number for t in valid_trials])
            with chart_placeholder:
                st.line_chart(df, y="Trial Score")
        
        # Calculate how many trials are actually left to run (Move this up here!)
        completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        remaining_trials = n_trials - completed_trials

        # UI Elements for Verbose Updates
        progress_bar = st.progress(min(completed_trials / n_trials, 1.0), text=f"Starting tuning... ({completed_trials}/{n_trials} completed)")
        log_container = st.empty()
        chart_placeholder = st.empty()

        def streamlit_callback(study, trial):
            # 1. Update the Text Log
            val_str = f"{trial.value:.4f}" if trial.value is not None else "Failed/Pruned"
            best_str = f"{study.best_value:.4f}" if len(study.best_trials) > 0 else "N/A"
            with log_container:
                st.success(f"🔄 **Latest Update (Trial {trial.number}):** Score = `{val_str}` | 🏆 **Current Best:** `{best_str}`")
            
            # 2. Update the Progress Bar natively!
            current_total = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            pct_complete = min(current_total / n_trials, 1.0)
            progress_bar.progress(pct_complete, text=f"Tuning progress: {current_total} / {n_trials} trials completed")
                
            # 3. Live Graph Update
            valid_trials = [t for t in study.trials if t.value is not None]
            if valid_trials:
                df = pd.DataFrame({
                    "Trial Score": [t.value for t in valid_trials]
                }, index=[t.number for t in valid_trials])
                
                with chart_placeholder:
                    st.line_chart(df, y="Trial Score")
        
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, timeout=timeout, callbacks=[streamlit_callback])
            progress_bar.progress(1.0, text=f"✅ Tuning Complete! ({len(study.trials)}/{n_trials} trials)")
        else:
            progress_bar.progress(1.0, text=f"✅ Tuning already reached the target of {n_trials} trials!")
        
        # --- NEW: Final Interactive Optuna Plot ---
        st.divider()
        st.markdown("#### 📈 Optimization History & Insights")
        try:
            # 1. Show the interactive plots in tabs
            tab1, tab2, tab3 = st.tabs(["Optimization History", "Parameter Importance", "Parallel Coordinates"])
            
            with tab1:
                st.plotly_chart(plot_optimization_history(study), width='stretch')
            with tab2:
                st.plotly_chart(plot_param_importances(study), width='stretch')
            with tab3:
                st.plotly_chart(plot_parallel_coordinate(study), width='stretch')
                
            # 2. Retain the CSV Download Button below the plots
            trials_df = study.trials_dataframe()
            csv_logs = trials_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Tuning History Logs (.csv)", 
                data=csv_logs, 
                file_name=f"{config_name}_optuna_logs.csv", 
                mime="text/csv"
            )
        except ImportError:
            st.warning("Install `plotly` to see the advanced interactive tuning graphs! (`pip install plotly`)")

        study_best_params = study.best_params

        # ── 5. Build Final Model with Best Params ──────────────────────────────────
        use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled" # <-- ADD THIS
        
        if use_stacking:
            final_estimators = []
            for name in config.get("base_model_chosen", []):
                prefix = f"base_{name}_"
                best_base_hp = {k[len(prefix):]: v for k, v in study.best_params.items() if k.startswith(prefix)}
                mdl = build_sklearn_model(name, best_base_hp, random_state, is_regression)
                if mdl is not None: final_estimators.append((name, mdl))

            meta_name = config.get("meta_model_chosen")
            prefix = f"meta_{meta_name}_"
            best_meta_hp = {k[len(prefix):]: v for k, v in study.best_params.items() if k.startswith(prefix)}
            meta_mdl = build_sklearn_model(meta_name, best_meta_hp, random_state, is_regression)

            StackingCls = StackingRegressor if is_regression else StackingClassifier
            final_model = StackingCls(estimators=final_estimators, final_estimator=meta_mdl, passthrough=use_passthrough)
        else:
            final_model_name = config.get("model_chosen")
            prefix = f"base_{final_model_name}_"
            stripped_params = {k[len(prefix):]: v for k, v in study.best_params.items() if k.startswith(prefix)}
            final_model = build_sklearn_model(final_model_name, stripped_params, random_state, is_regression)
            
        steps = []
        if preprocessor:
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", final_model))
        pipeline = Pipeline(steps)

    elif tuner_choice == "KT Bayesian Optimization":
        kt_dir = Path("./saved_configs/kt_tuning")
        kt_dir.mkdir(exist_ok=True, parents=True)
        
        st.info("💡 **Tip:** Keras-Tuner saves logs locally. If your device shuts down, tuning will safely resume from where it left off.")
        
        cv_splitter = (
            KFold(n_splits=kfold, shuffle=True, random_state=random_state)
            if is_regression
            else StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
        )
        X_arr = X_train.reset_index(drop=True)
        y_arr = y_train.reset_index(drop=True)
        
        tuning_obj = config.get("tuning_objective", "val_loss")
        
        # Initialize custom tuner. overwrite=False makes it Pausable!
        tuner = SklearnCVTuner(
            oracle=kt.oracles.BayesianOptimizationOracle(
                objective=kt.Objective(tuning_obj, direction=tuning_direction),
                max_trials=n_trials,
            ),
            directory=kt_dir,
            project_name=config_name,
            overwrite=False
        )
        
        with st.spinner(f"Running Keras-Tuner for {n_trials} trials..."):
            tuner.search(
                X_arr=X_arr, 
                y_arr=y_arr, 
                cv_splitter=cv_splitter, 
                config=config, 
                is_regression=is_regression, 
                random_state=random_state,
                feature_int_cols=list(feature_int.columns),
                feature_cat_cols=list(feature_cat.columns)
            )
            
        st.success("Tuning Complete!")
        best_hp = tuner.get_best_hyperparameters()[0]
        
        # We extract best params into a standard dictionary to feed into our final model builder
        study_best_params = best_hp.values

        # ── 5b. Build Final Model with Best Params (KT path) ─────────────────────
        use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled"

        if use_stacking:
            final_estimators = []
            for name in config.get("base_model_chosen", []):
                prefix = f"base_{name}_"
                best_base_hp = {k[len(prefix):]: v for k, v in study_best_params.items() if k.startswith(prefix)}
                mdl = build_sklearn_model(name, best_base_hp, random_state, is_regression)
                if mdl is not None: final_estimators.append((name, mdl))

            meta_name = config.get("meta_model_chosen")
            prefix = f"meta_{meta_name}_"
            best_meta_hp = {k[len(prefix):]: v for k, v in study_best_params.items() if k.startswith(prefix)}
            meta_mdl = build_sklearn_model(meta_name, best_meta_hp, random_state, is_regression)

            StackingCls = StackingRegressor if is_regression else StackingClassifier
            final_model = StackingCls(estimators=final_estimators, final_estimator=meta_mdl, passthrough=use_passthrough)
        else:
            final_model_name = config.get("model_chosen")
            prefix = f"base_{final_model_name}_"
            stripped_params = {k[len(prefix):]: v for k, v in study_best_params.items() if k.startswith(prefix)}
            final_model = build_sklearn_model(final_model_name, stripped_params, random_state, is_regression)

        steps = []
        if preprocessor:
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", final_model))
        pipeline = Pipeline(steps)

    # ── 7. Final fit & test evaluation ───────────────────────────────────────
    st.markdown("---")
    
    # Helper to generate feature importances without repeating code
    def render_feature_importance(fitted_pipe, X_input, render_target):
        try:
            fitted_model = fitted_pipe.named_steps["model"]
            if "preprocessor" in fitted_pipe.named_steps and fitted_pipe.named_steps["preprocessor"] is not None:
                input_features = list(fitted_pipe.named_steps["preprocessor"].get_feature_names_out())
            else:
                input_features = list(X_input.columns)

            if use_stacking:
                model_to_inspect = fitted_model.final_estimator_
                base_names = [f"Model: {name}" for name, _ in fitted_model.estimators]
                feature_names = base_names + input_features if fitted_model.passthrough else base_names
            else:
                model_to_inspect = fitted_model
                feature_names = input_features

            if hasattr(model_to_inspect, "feature_importances_"):
                importances = model_to_inspect.feature_importances_
                if len(feature_names) != len(importances):
                    feature_names = [f"Feature {i}" for i in range(len(importances))]
                feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)
                render_target.bar_chart(feat_df, x="Feature", y="Importance")
            elif hasattr(model_to_inspect, "coef_"):
                coefs = np.ravel(model_to_inspect.coef_[0] if len(model_to_inspect.coef_.shape) > 1 else model_to_inspect.coef_)
                if len(feature_names) != len(coefs):
                    feature_names = [f"Feature {i}" for i in range(len(coefs))]
                feat_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
                feat_df["Abs_Coef"] = feat_df["Coefficient"].abs()
                feat_df = feat_df.sort_values(by="Abs_Coef", ascending=False).head(15)
                render_target.bar_chart(feat_df, x="Feature", y="Coefficient")
            else:
                render_target.info("💡 Feature importance extraction is not natively supported by this specific model type.")
        except Exception as e:
            render_target.warning(f"⚠️ Could not generate feature importances: {e}")


    # --- Branch 1: Hold-Out Split ---
    if eval_mode == "Hold-Out Test Split":
        st.markdown("#### 🧪 Hold-Out Test Set Evaluation")
        with st.spinner("Fitting on full training set…"):
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = None
        if not is_regression and hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        result_container = st.container(border=True)
        display_metrics(y_test, y_pred, is_regression, result_container, y_prob=y_prob, key_suffix="holdout")
        
        st.markdown("---")
        st.markdown("#### 🌟 Feature Importance")
        render_feature_importance(pipeline, X_train, st)
        
        preview_actual = y_test.values
        preview_pred = y_pred

    # --- Branch 2: K-Fold on Whole Data ---
    else:
        st.markdown("#### 🧪 K-Fold Cross Validation Evaluation")
        st.info(f"Evaluating across {kfold} folds using the entire dataset ({len(X)} samples).")

        cv_eval = (
            KFold(n_splits=kfold, shuffle=True, random_state=random_state)
            if is_regression
            else StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
        )

        # Uses frontend tabs so clicking them doesn't destroy the button state
        fold_tabs = st.tabs([f"Fold {i+1}" for i in range(kfold)])
        out_of_fold_predictions = []

        with st.spinner(f"Fitting and evaluating all {kfold} folds..."):
            for i, ((train_idx, test_idx), tab) in enumerate(zip(cv_eval.split(X, stratify_array), fold_tabs)):
                X_f_train, X_f_test = X.iloc[train_idx], X.iloc[test_idx]
                y_f_train, y_f_test = y.iloc[train_idx], y.iloc[test_idx]

                # Clone pipeline to guarantee zero data leakage between folds
                fold_pipe = clone(pipeline)
                fold_pipe.fit(X_f_train, y_f_train)
                y_f_pred = fold_pipe.predict(X_f_test)
                y_f_prob = None
                if not is_regression and hasattr(fold_pipe, "predict_proba"):
                    try:
                        y_f_prob = fold_pipe.predict_proba(X_f_test)[:, 1]
                    except Exception:
                        pass

                # Save Out-Of-Fold predictions for the final export
                out_of_fold_predictions.extend(zip(test_idx, y_f_test, y_f_pred))

                with tab:
                    display_metrics(y_f_test, y_f_pred, is_regression, st.container(border=True), y_prob=y_f_prob, key_suffix=f"fold_{i}")
                    st.markdown("##### Feature Importance")
                    render_feature_importance(fold_pipe, X_f_train, st)

        # Re-fit the pipeline on 100% of the data for the final downloadable pickle
        with st.spinner("Fitting final model on the entire dataset for export..."):
            pipeline.fit(X, y)
            
        # Reconstruct the dataset's original order for the preview
        out_of_fold_predictions.sort(key=lambda x: x[0])
        preview_actual = [x[1] for x in out_of_fold_predictions]
        preview_pred = [x[2] for x in out_of_fold_predictions]


    # ── 8. Prediction preview ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Prediction Preview & Export")
    
    full_results_df = pd.DataFrame({
        "Actual": preview_actual,
        "Predicted": preview_pred,
    })
    
    if not is_regression:
        full_results_df["Correct"] = full_results_df["Actual"] == full_results_df["Predicted"]
        
    st.caption(f"Previewing first 20 of {len(full_results_df)} test samples:")
    st.dataframe(full_results_df.head(20), width='stretch')

    csv_data = full_results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📊 Download Full Predictions (.csv)",
        data=csv_data,
        file_name=f"{config_name}_predictions.csv",
        mime="text/csv",
    )

    # ── 9. Save trained pipeline ──────────────────────────────────────────────
    buf = BytesIO()
    pickle.dump(pipeline, buf)
    buf.seek(0)
    
    st.download_button(
        label="⬇️ Download Trained Pipeline (.pkl)",
        data=buf,
        file_name="trained_pipeline.pkl",
        mime="application/octet-stream",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page Layout
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Run Configuration")
st.title("🚀 Run Saved Configuration")

# ── Main area: pick a saved config ───────────────────────────────────────────
st.header("Saved Configurations")
save_dir = Path("./saved_configs")
saved_files = sorted(save_dir.glob("*.pkl")) if save_dir.exists() else []

if not saved_files:
    st.warning("No saved configurations found.\n\nSave one from the **Model Configuration** page first.")
    st.stop()

selected_file = st.selectbox(
    "Choose a configuration",
    options=saved_files,
    format_func=lambda p: p.name,
    index=None,
)

if selected_file:
    cfg = load_config(selected_file)
    with st.expander("📋 Configuration Summary", expanded=False):
        summary_keys = [
            "target_chosen", "use_stacking", "model_chosen",
            "base_model_chosen", "meta_model_chosen",
            "library_chosen", "scaler_chosen", "categorical_enc_chosen",
            "train_split", "test_split", "kfold_amount",
            "random_state", "train_epoch", "batch_size",
            "hypertuner_chosen", "hypertune_trial",
            "stacking_passthrough"
        ]
        for k in summary_keys:
            if k in cfg:
                st.caption(f"**{k}:** {cfg[k]}")
        
        has_dataset = any(
            k in cfg
            for k in ("dataset_feature_int", "dataset_feature_cat", "dataset_target")
        )
        if has_dataset:
            n_rows    = len(cfg.get("dataset_target", pd.DataFrame()))
            n_num     = len(cfg.get("dataset_feature_int", pd.DataFrame()).columns)
            n_cat     = len(cfg.get("dataset_feature_cat", pd.DataFrame()).columns)
            st.caption(f"**Embedded dataset:** ✅ {n_rows} rows")
            st.caption(f"**Numerical features:** {n_num} · **Categorical:** {n_cat}")
        else:
            st.caption("**Embedded dataset:** ❌ not saved")
    
    eval_mode = st.radio(
        "Evaluation Strategy",
        options=["Hold-Out Test Split", "K-Fold Cross Validation (Whole Data)"],
        horizontal=True
    )
    
    if st.button("▶️ Run Training", type="primary", width='stretch'):
        run_pipeline(cfg, selected_file.stem, eval_mode)