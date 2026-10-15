"""
deep_learning.py
────────────────
Configurable Scikit-Learn-compatible Deep Learning estimators using
PyTorch and TensorFlow / Keras for Tabular Classification & Regression.

Supports:
- Arbitrary number of hidden layers and unit counts
- Configurable activation functions, dropout, and batch normalization
- Configurable optimizers, learning rates, weight decay, and batch sizes
- Built-in early stopping with validation tracking
- StackingClassifier and StackingRegressor compatibility
- Seamless pickle serialization and reloading for inference pipelines
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorch Tabular Estimators
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _get_best_device():
    if _HAS_TORCH:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _get_torch_activation(act_name: str):
    name = str(act_name).lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU()
    elif name in ["leaky_relu", "lrelu"]:
        return nn.LeakyReLU(negative_slope=0.1)
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "elu":
        return nn.ELU()
    elif name in ["silu", "swish"]:
        return nn.SiLU()
    return nn.ReLU()


class PyTorchTabularBase(BaseEstimator):
    """Base class for PyTorch tabular neural networks."""

    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_units: int = 64,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: str = "Disabled",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 32,
        epochs: int = 30,
        early_stopping: str = "Enabled",
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.model_ = None
        self.n_features_in_ = None
        self._state_dict = None

    def _build_network(self, input_dim: int, output_dim: int):
        layers = []
        in_dim = input_dim
        use_bn = str(self.use_batch_norm).lower() in ["enabled", "true", "1"]
        drop_rate = float(self.dropout) if self.dropout else 0.0

        for _ in range(max(1, int(self.hidden_layers))):
            layers.append(nn.Linear(in_dim, int(self.hidden_units)))
            if use_bn:
                layers.append(nn.BatchNorm1d(int(self.hidden_units)))
            layers.append(_get_torch_activation(self.activation))
            if drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))
            in_dim = int(self.hidden_units)

        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def _get_optimizer(self, model: nn.Module):
        opt_name = str(self.optimizer).lower()
        lr = float(self.learning_rate)
        wd = float(self.weight_decay)

        if opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        elif opt_name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            # Move state to CPU for portable serialization
            cpu_state = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            state["_state_dict"] = cpu_state
            state["model_"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        saved_dict = getattr(self, "_state_dict", None)
        if saved_dict is not None and getattr(self, "n_features_in_", None) is not None:
            is_clf = getattr(self, "_estimator_type", "") == "classifier"
            out_dim = len(self.classes_) if is_clf else 1
            model = self._build_network(self.n_features_in_, out_dim)
            model.load_state_dict(saved_dict)
            model.eval()
            self.model_ = model


class PyTorchMLPClassifier(ClassifierMixin, PyTorchTabularBase):
    """Scikit-learn compatible PyTorch Multi-Layer Perceptron Classifier."""
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import ClassifierTags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else ClassifierTags()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed. Please install torch.")

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        device = _get_best_device()
        is_cuda = device.type == "cuda"
        model = self._build_network(self.n_features_in_, n_classes).to(device)
        optimizer = self._get_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # Validation split for early stopping
        use_es = str(self.early_stopping).lower() in ["enabled", "true", "1"]
        n_samples = len(X)
        if use_es and n_samples >= 20:
            val_size = int(n_samples * 0.15)
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[val_size:], indices[:val_size]
            X_train, y_train = X[train_idx], y_encoded[train_idx]
            X_val, y_val = X[val_idx], y_encoded[val_idx]
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None

        train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.long))
        loader = DataLoader(train_ds, batch_size=max(8, int(self.batch_size)), shuffle=True, pin_memory=is_cuda)

        # Pre-allocate validation tensors on device once outside the epoch loop
        t_val_x = torch.as_tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
        t_val_y = torch.as_tensor(y_val, dtype=torch.long, device=device) if y_val is not None else None

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(max(1, int(self.epochs))):
            model.train()
            for b_x, b_y in loader:
                b_x = b_x.to(device, non_blocking=is_cuda)
                b_y = b_y.to(device, non_blocking=is_cuda)
                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

            if t_val_x is not None:
                model.eval()
                with torch.no_grad():
                    v_out = model(t_val_x)
                    val_loss = criterion(v_out, t_val_y).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= int(self.patience):
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.cpu()
        model.eval()
        self.model_ = model
        self._state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32)
            logits = self.model_(t_x)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]


class PyTorchMLPRegressor(RegressorMixin, PyTorchTabularBase):
    """Scikit-learn compatible PyTorch Multi-Layer Perceptron Regressor."""
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import RegressorTags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else RegressorTags()
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X, y):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is not installed. Please install torch.")

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        device = _get_best_device()
        is_cuda = device.type == "cuda"
        model = self._build_network(self.n_features_in_, 1).to(device)
        optimizer = self._get_optimizer(model)
        criterion = nn.MSELoss()

        use_es = str(self.early_stopping).lower() in ["enabled", "true", "1"]
        n_samples = len(X)
        if use_es and n_samples >= 20:
            val_size = int(n_samples * 0.15)
            indices = np.random.permutation(n_samples)
            train_idx, val_idx = indices[val_size:], indices[:val_size]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train, dtype=torch.float32))
        loader = DataLoader(train_ds, batch_size=max(8, int(self.batch_size)), shuffle=True, pin_memory=is_cuda)

        # Pre-allocate validation tensors on device once outside the epoch loop
        t_val_x = torch.as_tensor(X_val, dtype=torch.float32, device=device) if X_val is not None else None
        t_val_y = torch.as_tensor(y_val, dtype=torch.float32, device=device) if y_val is not None else None

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for _ in range(max(1, int(self.epochs))):
            model.train()
            for b_x, b_y in loader:
                b_x = b_x.to(device, non_blocking=is_cuda)
                b_y = b_y.to(device, non_blocking=is_cuda)
                optimizer.zero_grad()
                out = model(b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

            if t_val_x is not None:
                model.eval()
                with torch.no_grad():
                    v_out = model(t_val_x)
                    val_loss = criterion(v_out, t_val_y).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= int(self.patience):
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.cpu()
        model.eval()
        self.model_ = model
        self._state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32)
            preds = self.model_(t_x).numpy().ravel()
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# 2. TensorFlow / Keras Tabular Estimators
# ─────────────────────────────────────────────────────────────────────────────
try:
    import keras
    _HAS_KERAS = True
except ImportError:
    _HAS_KERAS = False


class TensorFlowTabularBase(BaseEstimator):
    """Base class for TensorFlow / Keras tabular neural networks."""

    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_units: int = 64,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: str = "Disabled",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 32,
        epochs: int = 30,
        early_stopping: str = "Enabled",
        patience: int = 5,
        random_state: int = 42,
    ):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.model_ = None
        self.n_features_in_ = None
        self._weights = None

    def _build_model(self, input_dim: int, output_dim: int, is_classifier: bool):
        inputs = keras.layers.Input(shape=(input_dim,))
        x = inputs
        use_bn = str(self.use_batch_norm).lower() in ["enabled", "true", "1"]
        drop_rate = float(self.dropout) if self.dropout else 0.0
        act = str(self.activation).lower()

        for _ in range(max(1, int(self.hidden_layers))):
            x = keras.layers.Dense(int(self.hidden_units))(x)
            if use_bn:
                x = keras.layers.BatchNormalization()(x)
            if act in ["relu", "gelu", "tanh", "elu", "silu", "selu"]:
                x = keras.layers.Activation(act)(x)
            elif act == "leaky_relu":
                x = keras.layers.LeakyReLU(negative_slope=0.1)(x)
            else:
                x = keras.layers.Activation("relu")(x)

            if drop_rate > 0.0:
                x = keras.layers.Dropout(drop_rate)(x)

        if is_classifier:
            outputs = keras.layers.Dense(output_dim, activation="softmax")(x)
        else:
            outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        opt_name = str(self.optimizer).lower()
        lr = float(self.learning_rate)
        wd = float(self.weight_decay)

        if opt_name == "adamw":
            opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        elif opt_name == "sgd":
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt = keras.optimizers.Adam(learning_rate=lr)

        loss = "sparse_categorical_crossentropy" if is_classifier else "mse"
        model.compile(optimizer=opt, loss=loss)
        return model

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            # Extract weights as pure numpy arrays for clean pickling
            try:
                state["_weights"] = [
                    w.detach().cpu().numpy() if hasattr(w, "detach") else np.array(w)
                    for w in self.model_.get_weights()
                ]
            except Exception:
                state["_weights"] = None
            state["model_"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        weights = getattr(self, "_weights", None)
        feat_in = getattr(self, "n_features_in_", None)
        if weights is not None and feat_in is not None:
            is_clf = getattr(self, "_estimator_type", "") == "classifier"
            out_dim = len(self.classes_) if is_clf else 1
            model = self._build_model(feat_in, out_dim, is_clf)
            model.set_weights(weights)
            self.model_ = model


class TensorFlowMLPClassifier(ClassifierMixin, TensorFlowTabularBase):
    """Scikit-learn compatible TensorFlow / Keras Multi-Layer Perceptron Classifier."""
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import ClassifierTags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else ClassifierTags()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        if not _HAS_KERAS:
            raise ImportError("Keras / TensorFlow is not available. Please install tensorflow and keras.")

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        keras.utils.set_random_seed(int(self.random_state))
        model = self._build_model(self.n_features_in_, n_classes, is_classifier=True)

        callbacks = []
        use_es = str(self.early_stopping).lower() in ["enabled", "true", "1"]
        val_split = 0.15 if (use_es and len(X) >= 20) else 0.0

        if val_split > 0.0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.patience),
                restore_best_weights=True
            ))

        model.fit(
            X,
            y_encoded,
            epochs=max(1, int(self.epochs)),
            batch_size=max(8, int(self.batch_size)),
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )

        self.model_ = model
        self._weights = model.get_weights()
        try:
            keras.backend.clear_session()
        except Exception:
            pass
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        preds = self.model_.predict(X, verbose=0)
        return np.asarray(preds, dtype=np.float32)

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]


class TensorFlowMLPRegressor(RegressorMixin, TensorFlowTabularBase):
    """Scikit-learn compatible TensorFlow / Keras Multi-Layer Perceptron Regressor."""
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        from sklearn.utils._tags import RegressorTags
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else RegressorTags()
        tags.estimator_type = "regressor"
        return tags

    def fit(self, X, y):
        if not _HAS_KERAS:
            raise ImportError("Keras / TensorFlow is not available. Please install tensorflow and keras.")

        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.n_features_in_ = X.shape[1]
        keras.utils.set_random_seed(int(self.random_state))
        model = self._build_model(self.n_features_in_, 1, is_classifier=False)

        callbacks = []
        use_es = str(self.early_stopping).lower() in ["enabled", "true", "1"]
        val_split = 0.15 if (use_es and len(X) >= 20) else 0.0

        if val_split > 0.0:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(self.patience),
                restore_best_weights=True
            ))

        model.fit(
            X,
            y,
            epochs=max(1, int(self.epochs)),
            batch_size=max(8, int(self.batch_size)),
            validation_split=val_split,
            callbacks=callbacks,
            verbose=0
        )

        self.model_ = model
        self._weights = model.get_weights()
        try:
            keras.backend.clear_session()
        except Exception:
            pass
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        preds = self.model_.predict(X, verbose=0)
        return np.asarray(preds, dtype=np.float32).ravel()
