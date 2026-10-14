import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
import time
import pickle
import threading
import queue
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO

import optuna
from optuna.trial import TrialState
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import json

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_score, recall_score
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import StackingRegressor, StackingClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.app.config import SAVED_CONFIGS_DIR, CONFIGURED_DATA_DIR, ARTIFACTS_DIR
from backend.app.utilities.registry import SCALER_MAP, ENCODER_MAP, get_model_class
from backend.app.utilities.model_hyperparams import get_json
from backend.app.services.config_service import load_and_preprocess_dataset

# In-memory storage for training jobs and pipelines
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
JOB_EVENT_QUEUES: Dict[str, List[queue.Queue]] = {}
CANCELLED_JOBS: Dict[str, threading.Event] = {}
JOB_LOCK = threading.Lock()

# Pipeline memory cache for instant live inference
_LOADED_PIPELINES: Dict[str, Tuple[float, Any]] = {}
_PIPELINE_LOCK = threading.Lock()


def get_loaded_pipeline(job_id: str) -> Any:
    """Retrieves trained pipeline from memory cache or loads from disk if modified."""
    job_dir = ARTIFACTS_DIR / job_id
    pkl_path = job_dir / "trained_pipeline.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Trained pipeline for job '{job_id}' not found.")

    mtime = pkl_path.stat().st_mtime
    with _PIPELINE_LOCK:
        if job_id in _LOADED_PIPELINES:
            cached_mtime, cached_pipe = _LOADED_PIPELINES[job_id]
            if cached_mtime == mtime:
                return cached_pipe

    with open(pkl_path, "rb") as f:
        pipe = pickle.load(f)

    with _PIPELINE_LOCK:
        if len(_LOADED_PIPELINES) >= 20:
            oldest_key = next(iter(_LOADED_PIPELINES))
            _LOADED_PIPELINES.pop(oldest_key, None)
        _LOADED_PIPELINES[job_id] = (mtime, pipe)

    return pipe


def _read_pickle(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_data_from_config(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_embedded = (
        "dataset_feature_int" in config
        or "dataset_feature_cat" in config
        or "dataset_target" in config
    )

    if has_embedded:
        feature_int = config.get("dataset_feature_int", pd.DataFrame())
        feature_cat = config.get("dataset_feature_cat", pd.DataFrame())
        target = config.get("dataset_target", pd.DataFrame())
        return feature_int, feature_cat, target

    chosen_config = config.get("chosen_configuration")
    if not chosen_config:
        raise ValueError("Configuration has neither embedded data nor chosen_configuration")

    _, target_pd, feature_int, feature_cat = load_and_preprocess_dataset(chosen_config)
    return feature_int, feature_cat, target_pd


def build_sklearn_model(
    model_name: str,
    hyperparams: dict,
    random_state: int,
    is_regression: bool = True,
):
    cls = get_model_class(model_name, is_regression)
    if cls is None:
        return None

    hyperparams = dict(hyperparams)

    # LightGBM Random Forest Safeguard
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

    try:
        dummy_model = cls()
        valid_keys = dummy_model.get_params().keys()
    except Exception:
        valid_keys = hyperparams.keys()

    valid_params = {}
    if "random_state" in valid_keys:
        valid_params["random_state"] = random_state
    elif "random_seed" in valid_keys:
        valid_params["random_seed"] = random_state

    if "n_jobs" in valid_keys:
        valid_params["n_jobs"] = -1
    elif "thread_count" in valid_keys:
        valid_params["thread_count"] = -1

    for k, v in hyperparams.items():
        if k in valid_keys:
            valid_params[k] = v

    try:
        return cls(**valid_params)
    except Exception as exc:
        print(f"Error instantiating {model_name}: {exc}")
        return None


def build_preprocessor(
    int_cols: list[str],
    cat_cols: list[str],
    scaler_name: str | None,
    encoder_name: str | None,
    impute_strategy: str | None = None
) -> ColumnTransformer | None:
    transformers = []

    if int_cols:
        num_steps = []
        if impute_strategy in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
            strat = "mean" if "Mean" in impute_strategy else ("median" if "Median" in impute_strategy else "most_frequent")
            num_steps.append(("imputer", SimpleImputer(strategy=strat)))

        if scaler_name and scaler_name in SCALER_MAP:
            num_steps.append(("scaler", clone(SCALER_MAP[scaler_name])))

        if num_steps:
            transformers.append(("num", Pipeline(num_steps), int_cols))
        else:
            transformers.append(("num", "passthrough", int_cols))

    if cat_cols:
        cat_steps = []
        if impute_strategy in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

        if encoder_name and encoder_name in ENCODER_MAP:
            cat_steps.append(("encoder", clone(ENCODER_MAP[encoder_name])))
        else:
            cat_steps.append(("encoder", clone(ENCODER_MAP["Ordinal Encoder"])))

        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers, remainder="passthrough")


def optuna_suggest_hyperparams(trial, config: dict, model_type: str, model_name: str) -> dict:
    model_options = get_json()
    target = config.get("target_chosen", "Regression")

    library = None
    for lib, models in model_options.get(target, {}).items():
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
                resolved[param_name] = trial.suggest_categorical(optuna_param_name, opts)

    return resolved


def _emit_job_event(job_id: str, event_data: dict):
    with JOB_LOCK:
        if job_id in TRAINING_JOBS:
            # Update job state
            for k, v in event_data.items():
                TRAINING_JOBS[job_id][k] = v
        
        # Broadcast to subscriber queues
        if job_id in JOB_EVENT_QUEUES:
            for q in JOB_EVENT_QUEUES[job_id]:
                try:
                    q.put_nowait(event_data)
                except Exception:
                    pass


def execute_training_pipeline(job_id: str, config: dict, config_name: str, eval_mode: str):
    """Executes model tuning, cross validation, final fit, evaluation and saves artifacts."""
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        _emit_job_event(job_id, {
            "status": "running",
            "progress": 0.05,
            "message": "Preparing data and preprocessing pipeline...",
            "logs": ["Loading dataset and building preprocessor..."]
        })

        is_regression = config.get("target_chosen") == "Regression"
        random_state = config.get("random_state", 42)
        kfold = config.get("kfold_amount", 5)
        train_pct = config.get("train_split", 80) / 100

        feature_int, feature_cat, target_pd = _load_data_from_config(config)

        if not feature_cat.empty:
            feature_cat = feature_cat.astype(str)

        X = pd.concat([feature_int, feature_cat], axis=1)
        if isinstance(target_pd, pd.DataFrame):
            y = target_pd.iloc[:, 0].copy()
        else:
            y = target_pd.squeeze()

        # Targets with NaN cannot serve as training ground truth or evaluation benchmark
        valid_target_mask = y.notnull()
        X = X.loc[valid_target_mask].copy()
        y = y.loc[valid_target_mask].copy()

        missing_strategy = config.get("missing_value_processing")
        if missing_strategy == "Drop Rows":
            valid_x_mask = X.notnull().all(axis=1)
            X = X.loc[valid_x_mask].copy()
            y = y.loc[valid_x_mask].copy()

        stratify_array = y if not is_regression else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=train_pct,
            random_state=random_state,
            stratify=stratify_array
        )

        preprocessor = build_preprocessor(
            list(feature_int.columns),
            list(feature_cat.columns),
            config.get("scaler_chosen"),
            config.get("categorical_enc_chosen"),
            missing_strategy
        )

        _emit_job_event(job_id, {
            "progress": 0.15,
            "message": "Data preprocessor ready. Starting hyperparameter optimization...",
            "logs": [f"Data prepared: {len(X)} total samples, {len(X_train)} train, {len(X_test)} test."]
        })

        # --- 2. Hyperparameter Optimization ---
        use_stacking = config.get("use_stacking") == "Use"
        n_trials = config.get("hypertune_trial", 20)
        tuning_direction = str(config.get("tuning_direction", "minimize")).lower()
        if tuning_direction not in ["minimize", "maximize"]:
            tuning_direction = "minimize" if is_regression else "maximize"
        tuning_obj = config.get("tuning_objective", "val_loss")
        initial_points = config.get("tuning_initial_points", 5)
        early_stopping = config.get("use_early_stopping", "Enabled") == "Enabled"
        timeout_sec = config.get("tuning_timeout", 3600)
        timeout = timeout_sec if timeout_sec > 0 else None

        cv_splitter = (
            KFold(n_splits=kfold, shuffle=True, random_state=random_state)
            if is_regression
            else StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
        )
        X_arr = X_train.reset_index(drop=True)
        y_arr = y_train.reset_index(drop=True)

        def objective(trial):
            if job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set():
                raise optuna.exceptions.OptunaError("Job was cancelled by user")

            try:
                use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled"

                if use_stacking:
                    fold_estimators = []
                    for name in config.get("base_model_chosen", []):
                        hp = optuna_suggest_hyperparams(trial, config, "base", name)
                        mdl = build_sklearn_model(name, hp, random_state, is_regression)
                        if mdl is not None:
                            fold_estimators.append((name, mdl))

                    meta_name = config.get("meta_model_chosen")
                    meta_hp = optuna_suggest_hyperparams(trial, config, "meta", meta_name)
                    meta_mdl = build_sklearn_model(meta_name, meta_hp, random_state, is_regression)

                    StackingCls = StackingRegressor if is_regression else StackingClassifier
                    blueprint_model = StackingCls(estimators=fold_estimators, final_estimator=meta_mdl, passthrough=use_passthrough)
                else:
                    model_name = config.get("model_chosen")
                    hp = optuna_suggest_hyperparams(trial, config, "base", model_name)
                    blueprint_model = build_sklearn_model(model_name, hp, random_state, is_regression)

                if blueprint_model is None:
                    raise optuna.TrialPruned("Failed to instantiate model with suggested hyperparameters")

                cv_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_arr, y_arr)):
                    if job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set():
                        raise optuna.exceptions.OptunaError("Job was cancelled by user")

                    fold_preprocessor = build_preprocessor(
                        list(feature_int.columns), list(feature_cat.columns),
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

                    # Intermediate fold pruning for early stopping (2x-3x speedup)
                    if early_stopping:
                        intermediate_score = float(np.mean(cv_scores))
                        trial.report(intermediate_score, step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned(
                                f"Trial pruned at fold {fold_idx + 1} with intermediate score {intermediate_score:.4f}"
                            )

                return float(np.mean(cv_scores))
            except Exception as exc:
                if isinstance(exc, (optuna.exceptions.OptunaError, optuna.TrialPruned)):
                    raise
                raise optuna.TrialPruned(f"Trial pruned: {str(exc)}")

        sampler = optuna.samplers.TPESampler(n_startup_trials=initial_points, seed=random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=initial_points, n_warmup_steps=1) if early_stopping else optuna.pruners.NopPruner()

        # In-memory Optuna study eliminates SQLite disk write latency and concurrency locks
        study = optuna.create_study(
            study_name=f"{config_name}_{job_id[:6]}",
            direction=tuning_direction,
            sampler=sampler,
            pruner=pruner
        )

        trial_history = []

        def optuna_callback(stdy, trl):
            if job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set():
                stdy.stop()
                return

            current_total = len([t for t in stdy.trials if t.state == TrialState.COMPLETE])
            pct_complete = min(current_total / n_trials, 1.0)
            scaled_prog = 0.15 + (pct_complete * 0.55)  # 15% to 70%

            val_score = float(trl.value) if trl.value is not None else None
            best_score = float(stdy.best_value) if len(stdy.best_trials) > 0 else None

            if val_score is not None:
                trial_history.append({
                    "trial": trl.number,
                    "score": round(val_score, 4),
                    "best": round(best_score, 4) if best_score is not None else round(val_score, 4)
                })

            log_msg = f"Trial {trl.number + 1}/{n_trials}: Score = {val_score:.4f} | Best = {best_score:.4f}" if val_score else f"Trial {trl.number + 1}: Pruned/Failed"
            
            _emit_job_event(job_id, {
                "progress": round(scaled_prog, 3),
                "current_trial": trl.number + 1,
                "total_trials": n_trials,
                "latest_score": val_score,
                "best_score": best_score,
                "message": f"Tuning progress: {current_total}/{n_trials} trials complete",
                "trial_history": list(trial_history),
                "latest_log": log_msg
            })

        with joblib.parallel_backend("threading"):
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[optuna_callback], catch=(Exception,))

        if job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set():
            raise Exception("Training job was cancelled by user.")

        _emit_job_event(job_id, {
            "progress": 0.72,
            "message": "Optimization finished! Building final pipeline...",
            "logs": ["Hyperparameter search completed. Fitting best model..."]
        })

        # --- 3. Build Final Model with Best Params ---
        use_passthrough = config.get("stacking_passthrough", "Disabled") == "Enabled"
        if use_stacking:
            final_estimators = []
            for name in config.get("base_model_chosen", []):
                prefix = f"base_{name}_"
                best_base_hp = {k[len(prefix):]: v for k, v in study.best_params.items() if k.startswith(prefix)}
                mdl = build_sklearn_model(name, best_base_hp, random_state, is_regression)
                if mdl is not None:
                    final_estimators.append((name, mdl))

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

        # Helper to compute feature importances
        def compute_feature_importance(fitted_pipe, X_sample, y_sample=None):
            try:
                fitted_model = fitted_pipe.named_steps["model"]
                if "preprocessor" in fitted_pipe.named_steps and fitted_pipe.named_steps["preprocessor"] is not None:
                    try:
                        input_features = list(fitted_pipe.named_steps["preprocessor"].get_feature_names_out())
                    except Exception:
                        input_features = list(X_sample.columns)
                else:
                    input_features = list(X_sample.columns)

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
                    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
                    df_imp = df_imp.sort_values(by="importance", ascending=False).head(15)
                    return [{"feature": r["feature"], "importance": float(r["importance"])} for _, r in df_imp.iterrows()]
                elif hasattr(model_to_inspect, "coef_"):
                    coefs = np.ravel(model_to_inspect.coef_[0] if len(model_to_inspect.coef_.shape) > 1 else model_to_inspect.coef_)
                    if len(feature_names) != len(coefs):
                        feature_names = [f"Feature {i}" for i in range(len(coefs))]
                    df_imp = pd.DataFrame({"feature": feature_names, "importance": np.abs(coefs), "raw_coef": coefs})
                    df_imp = df_imp.sort_values(by="importance", ascending=False).head(15)
                    return [{"feature": r["feature"], "importance": float(r["importance"]), "raw_coef": float(r["raw_coef"])} for _, r in df_imp.iterrows()]
                elif y_sample is not None and len(X_sample) > 0:
                    # Fallback to Permutation Importance for models without native feature_importances_ (e.g. SVR, KNN, MLPs)
                    sub_n = min(150, len(X_sample))
                    X_sub = X_sample.iloc[:sub_n]
                    y_sub = y_sample.iloc[:sub_n]
                    perm = permutation_importance(fitted_pipe, X_sub, y_sub, n_repeats=3, random_state=random_state, n_jobs=1)
                    imp_mean = np.maximum(0, perm.importances_mean)
                    feat_names = list(X_sample.columns)
                    if len(feat_names) == len(imp_mean):
                        df_imp = pd.DataFrame({"feature": feat_names, "importance": imp_mean})
                        df_imp = df_imp.sort_values(by="importance", ascending=False).head(15)
                        return [{"feature": r["feature"], "importance": float(r["importance"])} for _, r in df_imp.iterrows()]
            except Exception as e:
                print(f"Feature importance error: {e}")
            return []

        # Helper to compute metric details and save PNG charts
        def generate_metrics_and_plots(y_truth, y_predicted, y_probability=None, prefix=""):
            y_truth = np.asarray(y_truth)
            y_predicted = np.asarray(y_predicted)
            metrics = {}

            if is_regression:
                mae = float(mean_absolute_error(y_truth, y_predicted))
                rmse = float(np.sqrt(mean_squared_error(y_truth, y_predicted)))
                r2 = float(r2_score(y_truth, y_predicted))

                # Extra regression metrics
                non_zero = y_truth != 0
                if np.any(non_zero):
                    mape = float(np.mean(np.abs((y_truth[non_zero] - y_predicted[non_zero]) / y_truth[non_zero])) * 100)
                else:
                    mape = 0.0
                medae = float(np.median(np.abs(y_truth - y_predicted)))

                metrics = {
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                    "mape": round(mape, 2),
                    "medae": round(medae, 4)
                }

                # Scatter data for interactive chart
                min_val = float(min(y_truth.min(), y_predicted.min()))
                max_val = float(max(y_truth.max(), y_predicted.max()))
                
                scatter_points = [
                    {"actual": round(float(a), 4), "predicted": round(float(p), 4)}
                    for a, p in zip(y_truth, y_predicted)
                ]
                residual_points = [
                    {"predicted": round(float(p), 4), "residual": round(float(a - p), 4)}
                    for a, p in zip(y_truth, y_predicted)
                ]

                # 1. Save Scatter Plot PNG
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(y_truth, y_predicted, alpha=0.5, color="#2563eb")
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted")
                ax.legend()
                fig.tight_layout()
                scatter_png_path = job_dir / f"{prefix}scatter_plot.png"
                fig.savefig(scatter_png_path, dpi=200)
                plt.close(fig)

                # 2. Save Residual Plot PNG
                fig_res, ax_res = plt.subplots(figsize=(6, 5))
                ax_res.scatter(y_predicted, y_truth - y_predicted, alpha=0.5, color="#f97316")
                ax_res.axhline(0, color="red", linestyle="--", lw=2)
                ax_res.set_xlabel("Predicted Values")
                ax_res.set_ylabel("Residual (Actual - Predicted)")
                ax_res.set_title("Residual Analysis")
                fig_res.tight_layout()
                residual_png_path = job_dir / f"{prefix}residual_plot.png"
                fig_res.savefig(residual_png_path, dpi=200)
                plt.close(fig_res)

                return metrics, {
                    "scatter_points": scatter_points[:500],
                    "residual_points": residual_points[:500],
                    "min_val": min_val,
                    "max_val": max_val,
                    "scatter_plot_url": f"/api/training/download/plot/{job_id}/{prefix}scatter_plot.png",
                    "residual_plot_url": f"/api/training/download/plot/{job_id}/{prefix}residual_plot.png"
                }

            else:
                acc = float(accuracy_score(y_truth, y_predicted))
                f1 = float(f1_score(y_truth, y_predicted, average="weighted", zero_division=0))
                prec = float(precision_score(y_truth, y_predicted, average="weighted", zero_division=0))
                rec = float(recall_score(y_truth, y_predicted, average="weighted", zero_division=0))
                rep_dict = classification_report(y_truth, y_predicted, zero_division=0, output_dict=True)
                metrics = {
                    "accuracy": round(acc, 4),
                    "weighted_f1": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "classification_report": rep_dict
                }

                # Confusion Matrix
                labels = sorted(list(set(y_truth) | set(y_predicted)))
                cm = confusion_matrix(y_truth, y_predicted, labels=labels)
                cm_data = {
                    "labels": [str(l) for l in labels],
                    "matrix": cm.tolist()
                }

                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                       xticklabels=labels, yticklabels=labels,
                       ylabel='True label', xlabel='Predicted label',
                       title='Confusion Matrix')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2. else "black")
                fig.tight_layout()
                cm_png_path = job_dir / f"{prefix}confusion_matrix.png"
                fig.savefig(cm_png_path, dpi=200)
                plt.close(fig)

                # ROC Curve & AUC (Binary and Multiclass OvR)
                roc_data = None
                if y_probability is not None and len(labels) >= 2:
                    try:
                        y_prob_arr = np.asarray(y_probability)
                        if len(labels) == 2:
                            pos_label = labels[1]
                            prob_pos = y_prob_arr[:, 1] if (y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2) else y_prob_arr.ravel()
                            fpr, tpr, _ = roc_curve(y_truth, prob_pos, pos_label=pos_label)
                            roc_auc = float(auc(fpr, tpr))
                            roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
                            roc_data = {
                                "auc": round(roc_auc, 4),
                                "points": roc_points,
                                "pos_label": str(pos_label),
                                "roc_plot_url": f"/api/training/download/plot/{job_id}/{prefix}roc_curve.png"
                            }
                            metrics["roc_auc"] = round(roc_auc, 4)

                            fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax_roc.set_xlim([0.0, 1.0])
                            ax_roc.set_ylim([0.0, 1.05])
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.set_title(f'ROC Curve ({pos_label})')
                            ax_roc.legend(loc="lower right")
                            fig_roc.tight_layout()
                            roc_png_path = job_dir / f"{prefix}roc_curve.png"
                            fig_roc.savefig(roc_png_path, dpi=200)
                            plt.close(fig_roc)

                        elif len(labels) > 2 and y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == len(labels):
                            # Multiclass One-vs-Rest (OvR) ROC & AUC
                            Y_bin = label_binarize(y_truth, classes=labels)
                            roc_auc = float(roc_auc_score(Y_bin, y_prob_arr, multi_class="ovr", average="weighted"))
                            fpr_micro, tpr_micro, _ = roc_curve(Y_bin.ravel(), y_prob_arr.ravel())
                            roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr_micro, tpr_micro)]
                            roc_data = {
                                "auc": round(roc_auc, 4),
                                "points": roc_points[:500],
                                "pos_label": "Multiclass OvR",
                                "roc_plot_url": f"/api/training/download/plot/{job_id}/{prefix}roc_curve.png"
                            }
                            metrics["roc_auc"] = round(roc_auc, 4)

                            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                            ax_roc.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
                            for c_idx, label in enumerate(labels[:4]):
                                fpr_c, tpr_c, _ = roc_curve(Y_bin[:, c_idx], y_prob_arr[:, c_idx])
                                auc_c = auc(fpr_c, tpr_c)
                                ax_roc.plot(fpr_c, tpr_c, lw=1.5, label=f'Class {label} ({auc_c:.2f})')
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax_roc.set_xlim([0.0, 1.0])
                            ax_roc.set_ylim([0.0, 1.05])
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.set_title(f'Multiclass OvR ROC (Weighted AUC = {roc_auc:.2f})')
                            ax_roc.legend(loc="lower right", fontsize=8)
                            fig_roc.tight_layout()
                            roc_png_path = job_dir / f"{prefix}roc_curve.png"
                            fig_roc.savefig(roc_png_path, dpi=200)
                            plt.close(fig_roc)
                    except Exception as e:
                        print(f"ROC curve error: {e}")

                return metrics, {
                    "confusion_matrix": cm_data,
                    "roc_data": roc_data,
                    "cm_plot_url": f"/api/training/download/plot/{job_id}/{prefix}confusion_matrix.png"
                }

        # --- 4. Evaluate Pipeline ---
        _emit_job_event(job_id, {
            "progress": 0.80,
            "message": f"Evaluating pipeline with {eval_mode}...",
            "logs": [f"Running evaluation using {eval_mode}..."]
        })

        results = {
            "eval_mode": eval_mode,
            "is_regression": is_regression,
            "best_params": study.best_params,
            "best_score": round(float(study.best_value), 4)
        }

        if eval_mode == "Hold-Out Test Split":
            with joblib.parallel_backend("threading"):
                pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = None
            if not is_regression and hasattr(pipeline, "predict_proba"):
                try:
                    y_prob = pipeline.predict_proba(X_test)
                except Exception:
                    pass

            metrics, chart_data = generate_metrics_and_plots(y_test, y_pred, y_prob)
            feat_imp = compute_feature_importance(pipeline, X_train, y_train)

            results["metrics"] = metrics
            results["charts"] = chart_data
            results["feature_importance"] = feat_imp

            # Prediction preview dataframe with input features included
            pred_df = X_test.copy().reset_index(drop=True)
            pred_df["Actual"] = y_test.values
            pred_df["Predicted"] = y_pred
            if not is_regression:
                pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]
            else:
                pred_df["Residual"] = np.round(pred_df["Actual"] - pred_df["Predicted"], 4)

        else:
            # K-Fold Cross Validation (Whole Data)
            cv_eval = (
                KFold(n_splits=kfold, shuffle=True, random_state=random_state)
                if is_regression
                else StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
            )

            out_of_fold = []
            fold_results = []

            for i, (train_idx, test_idx) in enumerate(cv_eval.split(X, stratify_array)):
                if job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set():
                    raise Exception("Training job was cancelled by user.")

                X_f_train, X_f_test = X.iloc[train_idx], X.iloc[test_idx]
                y_f_train, y_f_test = y.iloc[train_idx], y.iloc[test_idx]

                fold_pipe = clone(pipeline)
                with joblib.parallel_backend("threading"):
                    fold_pipe.fit(X_f_train, y_f_train)
                y_f_pred = fold_pipe.predict(X_f_test)
                y_f_prob = None
                if not is_regression and hasattr(fold_pipe, "predict_proba"):
                    try:
                        y_f_prob = fold_pipe.predict_proba(X_f_test)
                    except Exception:
                        pass

                out_of_fold.append((test_idx, y_f_test, y_f_pred, y_f_prob))

                f_metrics, f_charts = generate_metrics_and_plots(y_f_test, y_f_pred, y_f_prob, prefix=f"fold_{i+1}_")
                f_imp = compute_feature_importance(fold_pipe, X_f_train, y_f_train)
                fold_results.append({
                    "fold": i + 1,
                    "metrics": f_metrics,
                    "charts": f_charts,
                    "feature_importance": f_imp
                })

            # Re-fit final pipeline on 100% data for export
            with joblib.parallel_backend("threading"):
                pipeline.fit(X, y)

            all_indices = []
            actual_all = []
            pred_all = []
            prob_all_list = []
            has_probs = all(x[3] is not None for x in out_of_fold)

            for t_idx, y_true_f, y_pred_f, y_prob_f in out_of_fold:
                all_indices.extend(t_idx)
                actual_all.extend(y_true_f)
                pred_all.extend(y_pred_f)
                if has_probs:
                    prob_all_list.extend(y_prob_f)

            sort_order = np.argsort(all_indices)
            ordered_indices = [all_indices[idx] for idx in sort_order]
            actual_all = [actual_all[idx] for idx in sort_order]
            pred_all = [pred_all[idx] for idx in sort_order]
            prob_all = np.array([prob_all_list[idx] for idx in sort_order]) if has_probs else None

            overall_metrics, overall_charts = generate_metrics_and_plots(actual_all, pred_all, prob_all)
            overall_imp = compute_feature_importance(pipeline, X, y)

            results["metrics"] = overall_metrics
            results["charts"] = overall_charts
            results["feature_importance"] = overall_imp
            results["folds"] = fold_results

            pred_df = X.iloc[ordered_indices].copy().reset_index(drop=True)
            pred_df["Actual"] = actual_all
            pred_df["Predicted"] = pred_all
            if not is_regression:
                pred_df["Correct"] = pred_df["Actual"] == pred_df["Predicted"]
            else:
                pred_df["Residual"] = np.round(np.array(actual_all) - np.array(pred_all), 4)

        # Build feature schema for interactive live inference
        feature_schema = []
        for col in X.columns:
            is_num = bool(pd.api.types.is_numeric_dtype(X[col]))
            sample_val = X[col].dropna().iloc[0] if not X[col].dropna().empty else 0
            opts = [str(val) for val in X[col].dropna().unique()[:20]] if not is_num else []
            feature_schema.append({
                "name": str(col),
                "type": "number" if is_num else "category",
                "sample": float(sample_val) if is_num and isinstance(sample_val, (int, float, np.number)) else str(sample_val),
                "options": opts
            })
        results["feature_schema"] = feature_schema

        # --- 5. Export Files: Predictions CSV & Trained Pipeline PKL ---
        predictions_csv_path = job_dir / "predictions.csv"
        pred_df.to_csv(predictions_csv_path, index=False)

        pipeline_pkl_path = job_dir / "trained_pipeline.pkl"
        with open(pipeline_pkl_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Pre-populate pipeline cache for instant live inference
        with _PIPELINE_LOCK:
            _LOADED_PIPELINES[job_id] = (pipeline_pkl_path.stat().st_mtime, pipeline)

        # Optuna Trials Log CSV
        optuna_csv_path = job_dir / "optuna_trials.csv"
        study.trials_dataframe().to_csv(optuna_csv_path, index=False)

        # Preview first 20 records
        preview_records = pred_df.head(20).to_dict(orient="records")
        results["prediction_preview"] = preview_records
        results["total_predictions"] = len(pred_df)

        final_job_info = {
            "job_id": job_id,
            "config_name": config_name,
            "eval_mode": eval_mode,
            "status": "completed",
            "progress": 1.0,
            "message": "Training and evaluation completed successfully!",
            "created_at": TRAINING_JOBS.get(job_id, {}).get("created_at", time.time()),
            "completed_at": time.time(),
            "results": results,
            "download_urls": {
                "predictions_csv": f"/api/training/download/predictions/{job_id}",
                "pipeline_pkl": f"/api/training/download/model/{job_id}",
                "optuna_csv": f"/api/training/download/optuna/{job_id}"
            }
        }

        # Save metadata to disk so historical jobs persist across backend restarts
        meta_file = job_dir / "job_info.json"
        try:
            with open(meta_file, "w") as f:
                json.dump(final_job_info, f, default=str)
        except Exception as meta_err:
            print(f"Failed to persist job metadata: {meta_err}")

        _emit_job_event(job_id, final_job_info)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        is_cancelled = job_id in CANCELLED_JOBS and CANCELLED_JOBS[job_id].is_set()
        final_status = "cancelled" if is_cancelled else "failed"
        final_msg = "Training was cancelled by user." if is_cancelled else f"Training failed: {str(e)}"
        print(f"Job {job_id} {final_status}: {tb}")

        failed_job_info = {
            "job_id": job_id,
            "config_name": config_name,
            "eval_mode": eval_mode,
            "status": final_status,
            "error": str(e),
            "traceback": tb,
            "message": final_msg,
            "created_at": TRAINING_JOBS.get(job_id, {}).get("created_at", time.time()),
            "completed_at": time.time()
        }

        # Persist failure or cancellation metadata
        try:
            with open(job_dir / "job_info.json", "w") as f:
                json.dump(failed_job_info, f, default=str)
        except Exception:
            pass

        _emit_job_event(job_id, failed_job_info)


def start_training_job(config: dict, config_name: str, eval_mode: str) -> str:
    """Spawns background training job and returns job_id."""
    job_id = str(uuid.uuid4())
    initial_job_state = {
        "job_id": job_id,
        "config_name": config_name,
        "eval_mode": eval_mode,
        "status": "pending",
        "progress": 0.0,
        "message": "Queued for training...",
        "created_at": time.time(),
        "trial_history": [],
        "logs": []
    }

    with JOB_LOCK:
        TRAINING_JOBS[job_id] = initial_job_state
        JOB_EVENT_QUEUES[job_id] = []
        CANCELLED_JOBS[job_id] = threading.Event()

    thread = threading.Thread(
        target=execute_training_pipeline,
        args=(job_id, config, config_name, eval_mode),
        daemon=True
    )
    thread.start()

    return job_id


def cancel_training_job(job_id: str) -> bool:
    """Signals cancellation to a running job."""
    with JOB_LOCK:
        if job_id in CANCELLED_JOBS:
            CANCELLED_JOBS[job_id].set()
        if job_id in TRAINING_JOBS:
            curr = TRAINING_JOBS[job_id].get("status")
            if curr in ["pending", "running"]:
                TRAINING_JOBS[job_id]["status"] = "cancelled"
                TRAINING_JOBS[job_id]["message"] = "Training job cancelled by user."
                _emit_job_event(job_id, {
                    "status": "cancelled",
                    "message": "Training job cancelled by user."
                })
                return True
    return False


def get_training_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches job status, loading from disk if not present in memory."""
    with JOB_LOCK:
        if job_id in TRAINING_JOBS:
            return TRAINING_JOBS[job_id]

    # Try loading from disk
    meta_path = ARTIFACTS_DIR / job_id / "job_info.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
                with JOB_LOCK:
                    TRAINING_JOBS[job_id] = data
                return data
        except Exception:
            pass

    return None


def list_all_training_jobs() -> List[Dict[str, Any]]:
    """Lists all active and historical training jobs."""
    jobs_dict: Dict[str, Dict[str, Any]] = {}

    # 1. From persistent disk storage
    if ARTIFACTS_DIR.exists():
        for job_folder in ARTIFACTS_DIR.iterdir():
            if job_folder.is_dir():
                meta_file = job_folder / "job_info.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            data = json.load(f)
                            # Light summary for listing
                            results_sum = data.get("results", {})
                            summary = {
                                "job_id": data.get("job_id", job_folder.name),
                                "config_name": data.get("config_name", "Unknown"),
                                "eval_mode": data.get("eval_mode", "Hold-Out Test Split"),
                                "status": data.get("status", "completed"),
                                "progress": data.get("progress", 1.0),
                                "created_at": data.get("created_at", job_folder.stat().st_mtime),
                                "completed_at": data.get("completed_at"),
                                "best_score": results_sum.get("best_score"),
                                "metrics": results_sum.get("metrics", {})
                            }
                            jobs_dict[data.get("job_id", job_folder.name)] = summary
                    except Exception:
                        pass

    # 2. Overlay live in-memory jobs
    with JOB_LOCK:
        for jid, jdata in TRAINING_JOBS.items():
            results_sum = jdata.get("results", {})
            jobs_dict[jid] = {
                "job_id": jid,
                "config_name": jdata.get("config_name", "Unknown"),
                "eval_mode": jdata.get("eval_mode", "Hold-Out Test Split"),
                "status": jdata.get("status", "pending"),
                "progress": jdata.get("progress", 0.0),
                "created_at": jdata.get("created_at", time.time()),
                "completed_at": jdata.get("completed_at"),
                "best_score": results_sum.get("best_score"),
                "metrics": results_sum.get("metrics", {})
            }

    # Sort descending by created_at
    sorted_jobs = sorted(jobs_dict.values(), key=lambda x: x.get("created_at", 0), reverse=True)
    return sorted_jobs


def predict_live(job_id: str, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs live model predictions on single or batch samples using the cached trained pipeline."""
    pipeline = get_loaded_pipeline(job_id)

    if isinstance(input_data, dict):
        df_in = pd.DataFrame([input_data])
    else:
        df_in = pd.DataFrame(input_data)
    predictions = pipeline.predict(df_in)

    probabilities = None
    classes = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(df_in)
            probabilities = proba.tolist()
            if hasattr(pipeline, "classes_"):
                classes = [str(c) for c in pipeline.classes_]
            elif "model" in getattr(pipeline, "named_steps", {}) and hasattr(pipeline.named_steps["model"], "classes_"):
                classes = [str(c) for c in pipeline.named_steps["model"].classes_]
        except Exception:
            pass

    formatted_preds = [
        float(p) if isinstance(p, (np.floating, float))
        else (int(p) if isinstance(p, (np.integer, int)) else str(p))
        for p in predictions
    ]

    return {
        "job_id": job_id,
        "count": len(formatted_preds),
        "predictions": formatted_preds,
        "probabilities": probabilities,
        "classes": classes
    }


def subscribe_to_job(job_id: str) -> queue.Queue:
    q = queue.Queue()
    with JOB_LOCK:
        if job_id not in JOB_EVENT_QUEUES:
            JOB_EVENT_QUEUES[job_id] = []
        JOB_EVENT_QUEUES[job_id].append(q)
    return q


def unsubscribe_from_job(job_id: str, q: queue.Queue):
    with JOB_LOCK:
        if job_id in JOB_EVENT_QUEUES and q in JOB_EVENT_QUEUES[job_id]:
            JOB_EVENT_QUEUES[job_id].remove(q)
