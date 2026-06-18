"""
registry.py
───────────
A centralized registry for all machine learning models, scalers, and encoders.
Provides a Single Source of Truth for both the UI (config_create.py) and 
the Execution Pipeline (config_run.py).
"""

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer,
    OneHotEncoder, OrdinalEncoder,
)

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# ─────────────────────────────────────────────────────────────────────────────
# 1. Optional Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scaler & Encoder Registry (Instantiated Objects)
# ─────────────────────────────────────────────────────────────────────────────
SCALER_MAP = {
    "Standard Scaler": StandardScaler(),
    "Min-Max Scaler": MinMaxScaler(),
    "Robust Scaler": RobustScaler(),
    "Max-Abs Scaler": MaxAbsScaler(),
    "Normalizer": Normalizer(),
    "Quantile Transformer": QuantileTransformer(),
    "Power Transformer": PowerTransformer(),
}

ENCODER_MAP = {
    "One-Hot Encoder": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "Ordinal Encoder": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Class Registry (Uninstantiated Classes)
# ─────────────────────────────────────────────────────────────────────────────
_REGRESSION_MAP: dict[str, type] = {
    "Linear Regression":            LinearRegression,
    "Ridge":                        Ridge,
    "Lasso":                        Lasso,
    "ElasticNet":                   ElasticNet,
    "Random Forest":                RandomForestRegressor,
    "Gradient Boosting":            GradientBoostingRegressor,
    "AdaBoost":                     AdaBoostRegressor,
    "SVR":                          SVR,
    "K-Nearest Neighbors":          KNeighborsRegressor,
    "Decision Tree":                DecisionTreeRegressor,
}

_CLASSIFICATION_MAP: dict[str, type] = {
    "Logistic Regression":          LogisticRegression,
    "Random Forest":                RandomForestClassifier,
    "SVM":                          SVC,
    "Gradient Boosting":            GradientBoostingClassifier,
    "AdaBoost":                     AdaBoostClassifier,
    "K-Nearest Neighbors":          KNeighborsClassifier,
    "Decision Tree":                DecisionTreeClassifier,
    "SGD Classifier":               SGDClassifier,
    "Naive Bayes":                  GaussianNB,
}

# Inject optional boosting libraries if they are installed
if _HAS_XGB:
    _REGRESSION_MAP["XGBoost"] = XGBRegressor
    _CLASSIFICATION_MAP["XGBoost"] = XGBClassifier

if _HAS_LGB:
    _REGRESSION_MAP["LightGBM"] = LGBMRegressor
    _CLASSIFICATION_MAP["LightGBM"] = LGBMClassifier

if _HAS_CAT:
    _REGRESSION_MAP["CatBoost"] = CatBoostRegressor
    _CLASSIFICATION_MAP["CatBoost"] = CatBoostClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper Method
# ─────────────────────────────────────────────────────────────────────────────
def get_model_class(model_name: str, is_regression: bool) -> type | None:
    """Returns the correct sklearn-compatible uninstantiated class."""
    model_map = _REGRESSION_MAP if is_regression else _CLASSIFICATION_MAP
    
    # 1. Exact Match
    if model_name in model_map:
        return model_map[model_name]
    
    # 2. Case-Insensitive fallback Match
    lower = model_name.lower()
    for key, cls in model_map.items():
        if key.lower() == lower:
            return cls
            
    return None