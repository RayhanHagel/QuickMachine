const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export interface FileMeta {
  id: string;
  name: string;
  path: string;
  type: string;
  size: number;
  columns: string[];
  row_count: number;
  preview: Record<string, any>[];
  error?: string;
}

export interface ItemGroup {
  id: string;
  type: string;
  role: string | null;
  columns: string[];
  feature_column: string[];
  target_column: string[];
  ignore_column: string[];
  item_list: { path: string; name: string }[];
}

export interface ConfiguredDataset {
  name: string;
  path: string;
  created_at?: number;
  types: string[];
  file_count: number;
  feature_count: number;
  target_count: number;
  features: string[];
  targets: string[];
  error?: string;
}

export interface DatasetInspection {
  dataset_name: string;
  total_rows: number;
  feature_count: number;
  target_count: number;
  numerical_features: string[];
  categorical_features: string[];
  target_columns: string[];
  missing_values: {
    cell_missing_count: number;
    cell_missing_pct: number;
    row_missing_count: number;
    row_missing_pct: number;
  };
  target_type_hint: "Regression" | "Classification";
  available_scalers: string[];
  available_encoders: string[];
  previews: {
    numerical: Record<string, any>[];
    categorical: Record<string, any>[];
    target: Record<string, any>[];
  };
}

export interface SavedConfigSummary {
  name: string;
  file_name: string;
  path: string;
  created_at?: number;
  target_chosen: string;
  use_stacking: string;
  model_chosen?: string;
  base_model_chosen?: string[];
  meta_model_chosen?: string;
  hypertuner_chosen: string;
  hypertune_trial: number;
  train_split: number;
  test_split: number;
  kfold_amount: number;
  scaler_chosen?: string;
  categorical_enc_chosen?: string;
  has_embedded_dataset: boolean;
  row_count: number;
  num_features_count: number;
  cat_features_count: number;
  error?: string;
}

export interface FeatureSchemaItem {
  name: string;
  type: "number" | "category";
  sample: any;
  options?: string[];
}

export interface TrainingJobSummary {
  job_id: string;
  config_name: string;
  eval_mode: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  created_at: number;
  completed_at?: number;
  best_score?: number;
  metrics?: Record<string, any>;
}

export interface TrainingJobStatus {
  job_id: string;
  config_name: string;
  eval_mode: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  created_at?: number;
  completed_at?: number;
  current_trial?: number;
  total_trials?: number;
  latest_score?: number;
  best_score?: number;
  trial_history?: { trial: number; score: number; best: number }[];
  latest_log?: string;
  logs?: string[];
  error?: string;
  results?: {
    eval_mode: string;
    is_regression: boolean;
    best_params: Record<string, any>;
    best_score: number;
    metrics: Record<string, any>;
    charts: Record<string, any>;
    feature_importance: { feature: string; importance: number; raw_coef?: number }[];
    feature_schema?: FeatureSchemaItem[];
    folds?: {
      fold: number;
      metrics: Record<string, any>;
      charts: Record<string, any>;
      feature_importance: { feature: string; importance: number }[];
    }[];
    prediction_preview: Record<string, any>[];
    total_predictions: number;
  };
  download_urls?: {
    predictions_csv: string;
    pipeline_pkl: string;
    optuna_csv: string;
  };
}

// ── Data APIs ────────────────────────────────────────────────────────────────
export async function listDataConfigs(): Promise<ConfiguredDataset[]> {
  const res = await fetch(`${API_BASE}/api/data/configs`);
  if (!res.ok) throw new Error("Failed to fetch data configurations");
  const data = await res.json();
  return data.datasets || [];
}

export async function deleteDataConfig(name: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/data/configs/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete dataset '${name}'`);
  return true;
}

export async function uploadFiles(files: File[]): Promise<FileMeta[]> {
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }
  const res = await fetch(`${API_BASE}/api/data/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to upload files");
  }
  const data = await res.json();
  return data.files || [];
}

export async function saveDatasetConfig(name: string, item_groups: ItemGroup[], overwrite = false): Promise<any> {
  const res = await fetch(`${API_BASE}/api/data/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, item_groups, overwrite }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to save dataset configuration");
  }
  return res.json();
}

// ── Config APIs ──────────────────────────────────────────────────────────────
export async function inspectDataset(dataName: string): Promise<DatasetInspection> {
  const res = await fetch(`${API_BASE}/api/config/data-preview/${encodeURIComponent(dataName)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to inspect dataset '${dataName}'`);
  }
  return res.json();
}

export async function getModelOptions(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/config/model-options`);
  if (!res.ok) throw new Error("Failed to fetch model options schema");
  return res.json();
}

export async function listSavedConfigs(): Promise<SavedConfigSummary[]> {
  const res = await fetch(`${API_BASE}/api/config/saved`);
  if (!res.ok) throw new Error("Failed to fetch saved model configurations");
  const data = await res.json();
  return data.saved_configs || [];
}

export async function deleteSavedConfig(name: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/config/saved/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete saved config '${name}'`);
  return true;
}

export async function savePipelineConfig(configName: string, configParams: Record<string, any>): Promise<any> {
  const res = await fetch(`${API_BASE}/api/config/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config_name: configName, config: configParams }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to save pipeline configuration");
  }
  return res.json();
}

// ── Training APIs ────────────────────────────────────────────────────────────
export async function startTraining(configName: string, evalMode: string, customConfig?: Record<string, any>): Promise<{ job_id: string }> {
  const res = await fetch(`${API_BASE}/api/training/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      config_name: configName,
      eval_mode: evalMode,
      custom_config: customConfig,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to start training");
  }
  return res.json();
}

export async function getTrainingStatus(jobId: string): Promise<TrainingJobStatus> {
  const res = await fetch(`${API_BASE}/api/training/status/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error("Failed to get training status");
  return res.json();
}

export function getPredictionsDownloadUrl(jobId: string): string {
  return `${API_BASE}/api/training/download/predictions/${encodeURIComponent(jobId)}`;
}

export function getModelDownloadUrl(jobId: string): string {
  return `${API_BASE}/api/training/download/model/${encodeURIComponent(jobId)}`;
}

export function getOptunaLogsDownloadUrl(jobId: string): string {
  return `${API_BASE}/api/training/download/optuna/${encodeURIComponent(jobId)}`;
}

export async function listTrainingJobs(): Promise<TrainingJobSummary[]> {
  const res = await fetch(`${API_BASE}/api/training/jobs`);
  if (!res.ok) throw new Error("Failed to fetch training jobs list");
  const data = await res.json();
  return data.jobs || [];
}

export async function cancelTrainingJob(jobId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/training/cancel/${encodeURIComponent(jobId)}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to cancel job '${jobId}'`);
  }
  return true;
}

export async function predictLive(
  jobId: string,
  samples: Record<string, any>[] | Record<string, any>
): Promise<{
  job_id: string;
  count: number;
  predictions: any[];
  probabilities?: number[][];
  classes?: string[];
}> {
  const res = await fetch(`${API_BASE}/api/training/predict/${encodeURIComponent(jobId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ samples }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to generate live prediction");
  }
  return res.json();
}

export function getPlotDownloadUrl(jobId: string, plotName: string): string {
  return `${API_BASE}/api/training/download/plot/${encodeURIComponent(jobId)}/${encodeURIComponent(plotName)}`;
}
