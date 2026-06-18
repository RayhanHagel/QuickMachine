import streamlit as st

@st.cache_data
def get_json():
    model_options = {
        "Regression": {
            "scikit-learn": {
                "Random Forest": {
                    "library": "sklearn.ensemble.RandomForestRegressor",
                    "params": {
                        "n_estimators": [10, 1000],
                        "max_depth": [1, 50],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None],
                        "max_leaf_nodes": [10, 500],
                        "bootstrap": [True, False]
                    }
                },
                "Gradient Boosting": {
                    "library": "sklearn.ensemble.GradientBoostingRegressor",
                    "params": {
                        "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                        "learning_rate": [0.0001, 1.0],
                        "n_estimators": [10, 1000],
                        "subsample": [0.1, 1.0],
                        "criterion": ["friedman_mse", "squared_error"],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None],
                        "max_depth": [1, 10]
                    }
                },
                "SVR": {
                    "library": "sklearn.svm.SVR",
                    "params": {
                        "kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "degree": [1, 10],
                        "gamma": ["scale", "auto"],
                        "coef0": [0.0, 10.0],
                        "tol": [1e-4, 1e-1],
                        "C": [0.1, 100.0],
                        "epsilon": [0.01, 1.0]
                    }
                },
                "Linear Regression": {
                    "library": "sklearn.linear_model.LinearRegression",
                    "params": {
                        "fit_intercept": [True, False],
                        "positive": [True, False]
                    }
                },
                "Ridge": {
                    "library": "sklearn.linear_model.Ridge",
                    "params": {
                        "alpha": [0.01, 100.0],
                        "fit_intercept": [True, False],
                        "max_iter": [100, 2000],
                        "tol": [1e-4, 1e-1],
                        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                    }
                },
                "Lasso": {
                    "library": "sklearn.linear_model.Lasso",
                    "params": {
                        "alpha": [0.01, 100.0],
                        "fit_intercept": [True, False],
                        "max_iter": [100, 2000],
                        "tol": [1e-4, 1e-1],
                        "selection": ["cyclic", "random"]
                    }
                },
                "K-Nearest Neighbors": {
                    "library": "sklearn.neighbors.KNeighborsRegressor",
                    "params": {
                        "n_neighbors": [1, 100],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        "leaf_size": [10, 100],
                        "p": [1, 2],
                        "metric": ["minkowski", "euclidean", "manhattan"]
                    }
                },
                "Decision Tree": {
                    "library": "sklearn.tree.DecisionTreeRegressor",
                    "params": {
                        "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                        "splitter": ["best", "random"],
                        "max_depth": [1, 50],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None]
                    }
                },
                "ElasticNet": {
                    "library": "sklearn.linear_model.ElasticNet",
                    "params": {
                        "alpha": [0.01, 100.0],
                        "l1_ratio": [0.0, 1.0],
                        "fit_intercept": [True, False],
                        "max_iter": [100, 2000],
                        "tol": [1e-4, 1e-1],
                        "selection": ["cyclic", "random"]
                    }
                }
            },
            "tensorflow": {},
            "pytorch": {},
            "others": {
                "XGBoost": {
                    "library": "xgboost.XGBRegressor",
                    "params": {
                        "n_estimators": [10, 1000],
                        "max_depth": [1, 15],
                        "learning_rate": [0.001, 1.0],
                        "booster": ["gbtree", "dart"],
                        "tree_method": ["auto", "exact", "approx", "hist"],
                        "gamma": [0.0, 5.0],
                        "min_child_weight": [1, 20],
                        "subsample": [0.1, 1.0],
                        "colsample_bytree": [0.1, 1.0],
                        "reg_alpha": [0.0, 10.0],
                        "reg_lambda": [0.0, 10.0]
                    }
                },
                "LightGBM": {
                    "library": "lightgbm.LGBMRegressor",
                    "params": {
                        "boosting_type": ["gbdt", "dart", "goss", "rf"],
                        "num_leaves": [10, 256],
                        "max_depth": [-1, 15],
                        "learning_rate": [0.001, 1.0],
                        "n_estimators": [10, 1000],
                        "min_child_samples": [1, 100],
                        "subsample": [0.1, 1.0],
                        "colsample_bytree": [0.1, 1.0],
                        "reg_alpha": [0.0, 10.0],
                        "reg_lambda": [0.0, 10.0]
                    }
                },
                "CatBoost": {
                    "library": "catboost.CatBoostRegressor",
                    "params": {
                        "iterations": [10, 1000],
                        "learning_rate": [0.001, 1.0],
                        "depth": [1, 10],
                        "l2_leaf_reg": [1, 10],
                        "model_size_reg": [0.1, 10.0],
                        "subsample": [0.1, 1.0],
                        "colsample_bylevel": [0.1, 1.0],
                        "min_data_in_leaf": [1, 50],
                        "border_count": [1, 255]
                    }
                }
            }
        },
        "Classification": {
            "scikit-learn": {
                "Random Forest": {
                    "library": "sklearn.ensemble.RandomForestClassifier",
                    "params": {
                        "n_estimators": [10, 1000],
                        "criterion": ["gini", "entropy", "log_loss"],
                        "max_depth": [1, 50],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None],
                        "max_leaf_nodes": [10, 500],
                        "class_weight": ["balanced", "balanced_subsample", None]
                    }
                },
                "Gradient Boosting": {
                    "library": "sklearn.ensemble.GradientBoostingClassifier",
                    "params": {
                        "loss": ["log_loss", "exponential"],
                        "learning_rate": [0.001, 1.0],
                        "n_estimators": [10, 1000],
                        "subsample": [0.1, 1.0],
                        "criterion": ["friedman_mse", "squared_error"],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None],
                        "max_depth": [1, 10]
                    }
                },
                "SVM": {
                    "library": "sklearn.svm.SVC",
                    "params": {
                        "C": [0.1, 100.0],
                        "kernel": ["linear", "poly", "rbf", "sigmoid"],
                        "degree": [1, 10],
                        "gamma": ["scale", "auto"],
                        "coef0": [0.0, 10.0],
                        "shrinking": [True, False],
                        "tol": [1e-4, 1e-1],
                        "class_weight": ["balanced", None]
                    }
                },
                "Logistic Regression": {
                    "library": "sklearn.linear_model.LogisticRegression",
                    "params": {
                        "penalty": ["l1", "l2", "elasticnet", None],
                        "tol": [1e-4, 1e-1],
                        "C": [0.01, 100.0],
                        "fit_intercept": [True, False],
                        "class_weight": ["balanced", None],
                        "solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                        "max_iter": [100, 2000]
                    }
                },
                "K-Nearest Neighbors": {
                    "library": "sklearn.neighbors.KNeighborsClassifier",
                    "params": {
                        "n_neighbors": [1, 100],
                        "weights": ["uniform", "distance"],
                        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                        "leaf_size": [10, 100],
                        "p": [1, 2],
                        "metric": ["minkowski", "euclidean", "manhattan"]
                    }
                },
                "Decision Tree": {
                    "library": "sklearn.tree.DecisionTreeClassifier",
                    "params": {
                        "criterion": ["gini", "entropy", "log_loss"],
                        "splitter": ["best", "random"],
                        "max_depth": [1, 50],
                        "min_samples_split": [2, 20],
                        "min_samples_leaf": [1, 20],
                        "max_features": ["sqrt", "log2", None],
                        "class_weight": ["balanced", None]
                    }
                },
                "Naive Bayes": {
                    "library": "sklearn.naive_bayes.GaussianNB",
                    "params": {
                        "var_smoothing": [1e-11, 1e-1]
                    }
                }
            },
            "tensorflow": {},
            "pytorch": {},
            "others": {
                "XGBoost": {
                    "library": "xgboost.XGBClassifier",
                    "params": {
                        "n_estimators": [10, 1000],
                        "max_depth": [1, 15],
                        "learning_rate": [0.001, 1.0],
                        "booster": ["gbtree", "dart"],
                        "gamma": [0.0, 5.0],
                        "min_child_weight": [1, 20],
                        "subsample": [0.1, 1.0],
                        "colsample_bytree": [0.1, 1.0],
                        "scale_pos_weight": [1, 100],
                        "reg_alpha": [0.0, 10.0],
                        "reg_lambda": [0.0, 10.0]
                    }
                },
                "LightGBM": {
                    "library": "lightgbm.LGBMClassifier",
                    "params": {
                        "boosting_type": ["gbdt", "dart", "goss", "rf"],
                        "num_leaves": [10, 256],
                        "max_depth": [-1, 15],
                        "learning_rate": [0.001, 1.0],
                        "n_estimators": [10, 1000],
                        "class_weight": ["balanced", None],
                        "is_unbalance": [True, False],
                        "min_child_samples": [1, 100],
                        "subsample": [0.1, 1.0],
                        "colsample_bytree": [0.1, 1.0],
                        "reg_alpha": [0.0, 10.0],
                        "reg_lambda": [0.0, 10.0]
                    }
                },
                "CatBoost": {
                    "library": "catboost.CatBoostClassifier",
                    "params": {
                        "iterations": [10, 1000],
                        "learning_rate": [0.001, 1.0],
                        "depth": [1, 10],
                        "l2_leaf_reg": [1, 10],
                        "auto_class_weights": ["None", "Balanced", "SqrtBalanced"],
                        "subsample": [0.1, 1.0],
                        "colsample_bylevel": [0.1, 1.0],
                        "min_data_in_leaf": [1, 50]
                    }
                }
            }
        }
    }
    return model_options