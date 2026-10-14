# QuickMachine — Automated Machine Learning & Model Stacking Studio

QuickMachine has been migrated from Streamlit to a full-stack **Next.js (React 19 / TypeScript / Tailwind CSS / Recharts)** frontend and a high-performance **FastAPI (Python 3.11 / Optuna / Scikit-Learn)** backend.

---

## 🚀 Quick Start

### 1. Start Both Backend & Frontend Together
```bash
./run_all.sh
```
This automatically starts:
- **FastAPI backend** on [http://localhost:8000](http://localhost:8000) (Interactive Swagger Docs at [http://localhost:8000/docs](http://localhost:8000/docs))
- **Next.js frontend** on [http://localhost:3000](http://localhost:3000)

### 2. Or Start Separately

**Backend:**
```bash
./run_backend.sh
# or: python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
./run_frontend.sh
# or: cd frontend && npm run dev
```

---

## 🏗️ Architecture

```
QuickMachine/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, route registration
│   │   ├── config.py            # Workspace paths, WindowsPath compatibility patch
│   │   ├── routers/
│   │   │   ├── data.py          # /api/data: uploads, column extraction, dataset config
│   │   │   ├── config.py        # /api/config: data preview, model options, pipeline configs
│   │   │   └── training.py      # /api/training: job runner, SSE stream, file downloads
│   │   ├── services/
│   │   │   ├── data_service.py      # Upload parsing, column detection, dataset pickle save
│   │   │   ├── config_service.py    # Missing value stats, data preview, pipeline config save
│   │   │   └── training_service.py  # Optuna / KT tuning, cross-validation, metrics, artifacts
│   │   └── utilities/
│   │       ├── model_hyperparams.py # Parameter search spaces (no Streamlit dependency)
│   │       └── registry.py          # Model, Scaler, and Encoder registry
│   └── run.py                   # Backend entrypoint
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout, theme, navigation bar & footer
│   │   │   ├── page.tsx         # Dashboard overview, quick stats & recent configurations
│   │   │   ├── data/page.tsx    # Step 1: File uploads, column role assignment (Features/Target)
│   │   │   ├── config/page.tsx  # Step 2: Preprocessing, Stacking ensembles, Hyperparameters
│   │   │   └── run/page.tsx     # Step 3: Real-time training monitor, metrics, downloads
│   │   ├── components/
│   │   │   ├── Navbar.tsx       # Navigation bar with live backend health indicator
│   │   │   └── Footer.tsx       # Studio footer
│   │   └── lib/
│   │       └── api.ts           # Typed API client for FastAPI endpoints
│   ├── next.config.ts           # Next.js config with transparent /api rewrite proxy
│   └── package.json
├── saved_configs/               # Serialized training configurations (.pkl) & Optuna DB
├── configured_data/             # Configured dataset definitions (.pkl)
├── run_all.sh                   # Concurrent launcher script
├── run_backend.sh               # Backend launcher script
├── run_frontend.sh              # Frontend launcher script
└── requirements.txt             # Updated Python dependencies
```

---

## 🌟 Key Features

1. **Step 1: Data Preparation (`/data`)**
   - Supports CSV, Excel (`.xlsx`, `.xls`), CIF, and image datasets.
   - Smart mutual exclusion between Feature, Target, and Ignored columns.
   - Validates that at least one feature and target exist before persisting.

2. **Step 2: Model Configuration (`/config`)**
   - Instant dataset inspection: missing cell %, missing row %, and sample previews.
   - Missing value strategies: Drop Rows, Impute (Mean, Median, Mode).
   - Preprocessing: Scalers (Standard, Min-Max, Robust, MaxAbs, Normalizer, Quantile, Power) and Encoders (One-Hot, Ordinal).
   - Target configuration: Regression vs. Classification validation.
   - Ensembling: Stacking ensemble (multiple base learners + meta learner with passthrough) or single models.
   - Dynamic hyperparameter search space customization per model.
   - Bayesian Hypertuning: Optuna or Keras-Tuner with custom trials, epochs, objective, and direction.
   - Persists `.pkl` pipeline with embedded datasets for reproducibility.

3. **Step 3: Training & Evaluation (`/run`)**
   - Runs training asynchronously in the background with **Job Cancellation** support.
   - Real-time progress monitoring via **Server-Sent Events (SSE)**: live progress bar, trial scores, current best score, and optimization progression curve.
   - **Historical Runs Selector**: inspect previous runs and model diagnostics across server restarts.
   - Evaluation strategies: **Hold-Out Test Split** or **K-Fold Cross Validation**.
   - Comprehensive Diagnostics:
     - **Regression**: MAE, RMSE, R², MAPE (%), Median Absolute Error, interactive Actual vs. Predicted scatter plot, interactive Residual analysis plot.
     - **Classification**: Accuracy, Weighted F1, Precision, Recall, Classification Report, Confusion Matrix heatmap, ROC Curve with AUC (including string label support).
     - **Feature Importance**: Ranked bar chart of top 15 predictors.
     - **Fold Tabs**: Individual fold metrics and feature importances for K-Fold evaluation.
     - **Prediction Preview**: Sample predictions with input features included.
   - One-Click Downloads:
     - Full Predictions (`.csv`) with input features included
     - Trained Scikit-Learn Pipeline (`.pkl`)
     - Tuning History Logs (`.csv`)
     - Publication plots (`.png`) for scatter, residual, confusion matrix, and ROC curves.

4. **Step 4: Live Inference & Model Playground (`/run`)**
   - **Interactive Single Sample Prediction**: Form fields pre-populated with feature schemas for real-time predictions and class probability distributions.
   - **Batch CSV Scoring**: Upload new unseen CSV files to compute batch predictions with one-click download of the scored dataset.
