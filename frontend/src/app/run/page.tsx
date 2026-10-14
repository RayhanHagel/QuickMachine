"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  PlayCircle,
  Download,
  Activity,
  Layers,
  Sparkles,
  BarChart3,
  Table,
  StopCircle,
  History,
  Send,
  Upload,
  RefreshCw,
  TrendingUp
} from "lucide-react";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  listSavedConfigs,
  startTraining,
  getTrainingStatus,
  listTrainingJobs,
  cancelTrainingJob,
  predictLive,
  getPredictionsDownloadUrl,
  getModelDownloadUrl,
  SavedConfigSummary,
  TrainingJobStatus,
  TrainingJobSummary,
} from "@/lib/api";

function parseCsvText(text: string): Record<string, any>[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((h) => h.trim().replace(/^["']|["']$/g, ""));
  return lines.slice(1).map((line) => {
    const vals = line.split(",").map((v) => v.trim().replace(/^["']|["']$/g, ""));
    const row: Record<string, any> = {};
    headers.forEach((h, idx) => {
      const val = vals[idx] ?? "";
      row[h] = !isNaN(Number(val)) && val !== "" ? Number(val) : val;
    });
    return row;
  });
}

function downloadCsv(records: Record<string, any>[], filename: string) {
  if (records.length === 0) return;
  const headers = Object.keys(records[0]);
  const csvLines = [headers.join(",")];
  for (const r of records) {
    csvLines.push(headers.map((h) => JSON.stringify(r[h] ?? "")).join(","));
  }
  const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function ConfigRunContent() {
  const searchParams = useSearchParams();
  const initialConfig = searchParams.get("config") || "";

  // Saved configs list & selection
  const [configs, setConfigs] = useState<SavedConfigSummary[]>([]);
  const [selectedConfigName, setSelectedConfigName] = useState<string>(initialConfig);
  const [evalMode, setEvalMode] = useState<"Hold-Out Test Split" | "K-Fold Cross Validation (Whole Data)">("Hold-Out Test Split");

  // Historical Jobs
  const [pastJobs, setPastJobs] = useState<TrainingJobSummary[]>([]);

  // Training state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<TrainingJobStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [selectedFoldTab, setSelectedFoldTab] = useState<number>(1);

  // Live Inference Playground state
  const [inferenceInputs, setInferenceInputs] = useState<Record<string, any>>({});
  const [livePrediction, setLivePrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Batch CSV Prediction state
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchPredicting, setBatchPredicting] = useState(false);
  const [batchResult, setBatchResult] = useState<any>(null);
  const [batchError, setBatchError] = useState<string | null>(null);

  const loadPastJobs = async () => {
    try {
      const jobs = await listTrainingJobs();
      setPastJobs(jobs);
    } catch (err) {
      console.error("Error loading past jobs:", err);
    }
  };

  // Load saved configurations and history on mount
  useEffect(() => {
    async function loadData() {
      try {
        const [configList, jobs] = await Promise.all([
          listSavedConfigs(),
          listTrainingJobs(),
        ]);
        setConfigs(configList);
        setPastJobs(jobs);

        if (!selectedConfigName && configList.length > 0) {
          setSelectedConfigName(configList[0].name);
        }
      } catch (err) {
        console.error("Error loading configs/jobs:", err);
      }
    }
    loadData();
  }, []);

  // SSE streaming with polling fallback for real-time progress
  useEffect(() => {
    if (!activeJobId) return;

    let eventSource: EventSource | null = null;
    let pollTimer: NodeJS.Timeout | null = null;

    try {
      eventSource = new EventSource(`/api/training/stream/${encodeURIComponent(activeJobId)}`);
      eventSource.onmessage = (event) => {
        try {
          const data: TrainingJobStatus = JSON.parse(event.data);
          setJobStatus(data);
          if (data.status === "completed" || data.status === "failed" || data.status === "cancelled") {
            eventSource?.close();
            loadPastJobs();
          }
        } catch {
          // ignore parse error
        }
      };

      eventSource.onerror = () => {
        eventSource?.close();
        pollTimer = setInterval(async () => {
          try {
            const status = await getTrainingStatus(activeJobId);
            setJobStatus(status);
            if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
              if (pollTimer) clearInterval(pollTimer);
              loadPastJobs();
            }
          } catch {
            // ignore
          }
        }, 1500);
      };
    } catch {
      pollTimer = setInterval(async () => {
        try {
          const status = await getTrainingStatus(activeJobId);
          setJobStatus(status);
          if (status.status === "completed" || status.status === "failed" || status.status === "cancelled") {
            if (pollTimer) clearInterval(pollTimer);
            loadPastJobs();
          }
        } catch {
          // ignore
        }
      }, 1500);
    }

    return () => {
      if (eventSource) eventSource.close();
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [activeJobId]);

  // Selected config summary object
  const currentConfig = configs.find((c) => c.name === selectedConfigName);

  // Start training handler
  const handleStartTraining = async () => {
    if (!selectedConfigName) return;
    setIsStarting(true);
    setJobStatus(null);
    setLivePrediction(null);
    setBatchResult(null);
    try {
      const res = await startTraining(selectedConfigName, evalMode);
      setActiveJobId(res.job_id);
      loadPastJobs();
    } catch (err: any) {
      alert(err.message || "Failed to start training");
    } finally {
      setIsStarting(false);
    }
  };

  // Cancel training handler
  const handleCancelTraining = async () => {
    if (!activeJobId) return;
    setIsCancelling(true);
    try {
      await cancelTrainingJob(activeJobId);
      loadPastJobs();
    } catch (err: any) {
      alert(err.message || "Failed to cancel training");
    } finally {
      setIsCancelling(false);
    }
  };

  // Select historical job handler
  const handleSelectPastJob = async (jobId: string) => {
    if (!jobId) return;
    try {
      setActiveJobId(jobId);
      setLivePrediction(null);
      setBatchResult(null);
      const status = await getTrainingStatus(jobId);
      setJobStatus(status);
    } catch (err: any) {
      alert(err.message || "Failed to load past run results");
    }
  };

  // Live single sample prediction handler
  const handleLivePredict = async () => {
    if (!activeJobId) return;
    setPredicting(true);
    setPredictError(null);
    try {
      const schema = jobStatus?.results?.feature_schema || [];
      const payload: Record<string, any> = {};
      schema.forEach((f) => {
        const userVal = inferenceInputs[f.name];
        if (userVal !== undefined && userVal !== "") {
          payload[f.name] = userVal;
        } else {
          payload[f.name] = f.sample ?? 0;
        }
      });

      const res = await predictLive(activeJobId, payload);
      setLivePrediction(res);
    } catch (err: any) {
      setPredictError(err.message || "Live prediction failed");
    } finally {
      setPredicting(false);
    }
  };

  // Batch CSV file scoring handler
  const handleBatchPredict = async () => {
    if (!batchFile || !activeJobId) return;
    setBatchPredicting(true);
    setBatchError(null);
    try {
      const text = await batchFile.text();
      const records = parseCsvText(text);
      if (records.length === 0) {
        throw new Error("CSV file contains no valid data rows");
      }

      const res = await predictLive(activeJobId, records);
      const scoredRecords = records.map((r, i) => ({
        ...r,
        Predicted: res.predictions[i] ?? "",
      }));

      downloadCsv(scoredRecords, `${activeJobId.slice(0, 8)}_batch_scored.csv`);
      setBatchResult({ count: records.length });
    } catch (err: any) {
      setBatchError(err.message || "Failed to score batch CSV");
    } finally {
      setBatchPredicting(false);
    }
  };

  const results = jobStatus?.results;
  const isRegression = results?.is_regression ?? (currentConfig?.target_chosen === "Regression");

  return (
    <div className="space-y-10">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-zinc-900 dark:text-zinc-50 flex items-center gap-3">
          <PlayCircle className="h-8 w-8 text-emerald-500" />
          <span>Training Execution & Model Evaluation</span>
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Select a pipeline configuration, run Bayesian hyperparameter optimization, monitor real-time trials, and evaluate performance with interactive diagnostics.
        </p>
      </div>

      {/* Historical Jobs Selector */}
      {pastJobs.length > 0 && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl border border-indigo-200 dark:border-indigo-900/60 bg-indigo-50/50 dark:bg-indigo-950/20">
          <div className="flex items-center gap-2 text-indigo-900 dark:text-indigo-300 text-xs font-bold">
            <History className="h-4 w-4 text-indigo-500" />
            <span>Historical Training Runs ({pastJobs.length})</span>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={activeJobId || ""}
              onChange={(e) => handleSelectPastJob(e.target.value)}
              className="rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">-- Inspect a previous run --</option>
              {pastJobs.map((j) => (
                <option key={j.job_id} value={j.job_id}>
                  {j.config_name} · {j.status} · {new Date(j.created_at * 1000).toLocaleTimeString()} ({j.job_id.slice(0, 8)})
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={loadPastJobs}
              className="p-1.5 rounded-xl border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-600 dark:text-zinc-400 transition-colors"
              title="Refresh History"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* 1. Configuration Selection & Summary Card */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex-1 space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Select Saved Pipeline Configuration
            </label>
            <select
              value={selectedConfigName}
              onChange={(e) => setSelectedConfigName(e.target.value)}
              className="w-full rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-4 py-2.5 text-sm font-semibold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="">-- Choose a saved config --</option>
              {configs.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.target_chosen} · {c.use_stacking === "Use" ? "Stacking" : c.model_chosen} · {c.hypertune_trial} trials)
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Evaluation Strategy
            </label>
            <div className="flex gap-2">
              {[
                { label: "Hold-Out Split", val: "Hold-Out Test Split" },
                { label: "K-Fold (Whole Data)", val: "K-Fold Cross Validation (Whole Data)" },
              ].map((opt) => (
                <button
                  key={opt.val}
                  type="button"
                  onClick={() => setEvalMode(opt.val as any)}
                  className={`rounded-xl px-4 py-2.5 text-xs font-bold border transition-all ${
                    evalMode === opt.val
                      ? "border-emerald-600 bg-emerald-50 text-emerald-700 dark:border-emerald-500 dark:bg-emerald-950/50 dark:text-emerald-300 shadow-sm"
                      : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Configuration Snapshot Card */}
        {currentConfig && (
          <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-800/20 p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-600 dark:text-zinc-400 flex items-center gap-1.5">
              <Layers className="h-4 w-4" />
              <span>Configuration Details: {currentConfig.name}</span>
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-zinc-400">Target Type</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">{currentConfig.target_chosen}</p>
              </div>
              <div>
                <span className="text-zinc-400">Architecture</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">
                  {currentConfig.use_stacking === "Use"
                    ? `Stacking Ensemble (${currentConfig.base_model_chosen?.join(", ")})`
                    : currentConfig.model_chosen || "Single Model"}
                </p>
              </div>
              <div>
                <span className="text-zinc-400">Bayesian Tuner</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">
                  {currentConfig.hypertuner_chosen} ({currentConfig.hypertune_trial} trials)
                </p>
              </div>
              <div>
                <span className="text-zinc-400">Embedded Dataset</span>
                <p className="font-semibold text-emerald-600 dark:text-emerald-400">
                  {currentConfig.has_embedded_dataset ? `✅ ${currentConfig.row_count} rows` : "External"}
                </p>
              </div>
              <div>
                <span className="text-zinc-400">Scaler</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">{currentConfig.scaler_chosen || "None"}</p>
              </div>
              <div>
                <span className="text-zinc-400">Encoder</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">{currentConfig.categorical_enc_chosen || "None"}</p>
              </div>
              <div>
                <span className="text-zinc-400">Split Percentage</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">
                  {currentConfig.train_split}% Train / {currentConfig.test_split}% Test
                </p>
              </div>
              <div>
                <span className="text-zinc-400">K-Fold Splits</span>
                <p className="font-semibold text-zinc-800 dark:text-zinc-200">{currentConfig.kfold_amount} Folds</p>
              </div>
            </div>
          </div>
        )}

        {/* Start Training & Cancel Button */}
        <div className="flex gap-3">
          <button
            type="button"
            disabled={!selectedConfigName || isStarting || jobStatus?.status === "running"}
            onClick={handleStartTraining}
            className={`flex-1 flex items-center justify-center gap-2 rounded-xl py-3 text-sm font-bold shadow-md transition-all ${
              selectedConfigName && !isStarting && jobStatus?.status !== "running"
                ? "bg-emerald-600 text-white hover:bg-emerald-500 active:scale-98"
                : "bg-zinc-200 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-600 cursor-not-allowed"
            }`}
          >
            <PlayCircle className="h-5 w-5" />
            <span>
              {jobStatus?.status === "running"
                ? "Training in progress..."
                : isStarting
                ? "Starting training job..."
                : `Run Training (${evalMode})`}
            </span>
          </button>

          {(jobStatus?.status === "running" || jobStatus?.status === "pending") && (
            <button
              type="button"
              disabled={isCancelling}
              onClick={handleCancelTraining}
              className="px-5 py-3 rounded-xl border border-rose-300 dark:border-rose-800 bg-rose-50 text-rose-700 hover:bg-rose-100 dark:bg-rose-950/50 dark:text-rose-300 font-bold text-sm flex items-center gap-2 transition-all shadow-sm"
            >
              <StopCircle className="h-5 w-5" />
              <span>{isCancelling ? "Cancelling..." : "Cancel Training"}</span>
            </button>
          )}
        </div>
      </div>

      {/* 2. Live Training Monitor */}
      {jobStatus && (
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
              <Activity className="h-5 w-5 text-emerald-500" />
              <span>Training Live Monitor</span>
            </h2>
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider ${
                jobStatus.status === "completed"
                  ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                  : jobStatus.status === "failed" || jobStatus.status === "cancelled"
                  ? "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300"
                  : "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300 animate-pulse"
              }`}
            >
              {jobStatus.status}
            </span>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-medium text-zinc-500">
              <span>{jobStatus.message}</span>
              <span>{Math.round(jobStatus.progress * 100)}%</span>
            </div>
            <div className="h-2.5 w-full overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              <div
                className={`h-full transition-all duration-300 ${
                  jobStatus.status === "failed" || jobStatus.status === "cancelled"
                    ? "bg-rose-500"
                    : "bg-emerald-500"
                }`}
                style={{ width: `${Math.max(5, Math.min(100, jobStatus.progress * 100))}%` }}
              />
            </div>
          </div>

          {/* Live Metric Badges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-3">
              <span className="text-[11px] text-zinc-400">Current Trial</span>
              <p className="text-xl font-bold text-zinc-900 dark:text-zinc-100 mt-0.5">
                {jobStatus.current_trial ?? 0} / {jobStatus.total_trials ?? 0}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-3">
              <span className="text-[11px] text-zinc-400">Latest Trial Score</span>
              <p className="text-xl font-bold text-blue-600 dark:text-blue-400 mt-0.5">
                {jobStatus.latest_score !== undefined ? jobStatus.latest_score.toFixed(4) : "—"}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-3">
              <span className="text-[11px] text-zinc-400">Current Best Score</span>
              <p className="text-xl font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
                {jobStatus.best_score !== undefined ? jobStatus.best_score.toFixed(4) : "—"}
              </p>
            </div>
            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-3">
              <span className="text-[11px] text-zinc-400">Status</span>
              <p className="text-xl font-bold text-purple-600 dark:text-purple-400 mt-0.5 capitalize">
                {jobStatus.status}
              </p>
            </div>
          </div>

          {/* Live Trial History Chart */}
          {jobStatus.trial_history && jobStatus.trial_history.length > 0 && (
            <div className="space-y-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/30 dark:bg-zinc-800/20 p-4">
              <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-600 dark:text-zinc-400 flex items-center gap-1.5">
                <TrendingUp className="h-4 w-4" />
                <span>Optimization Progression Curve (Trials vs Score)</span>
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={jobStatus.trial_history}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="trial" label={{ value: "Trial #", position: "insideBottom", offset: -5 }} />
                    <YAxis domain={["auto", "auto"]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={1.5} dot={{ r: 2 }} name="Trial Score" />
                    <Line type="stepAfter" dataKey="best" stroke="#10b981" strokeWidth={2} dot={false} name="Best Score" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 3. Evaluation Results */}
      {results && (
        <div className="space-y-8">
          {/* Header & Artifact Download Actions */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-extrabold text-zinc-900 dark:text-zinc-50 flex items-center gap-2">
                <BarChart3 className="h-6 w-6 text-purple-600" />
                <span>Model Diagnostics & Performance Evaluation</span>
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                Evaluated with {results.eval_mode} · Best Optuna Score: {results.best_score}
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <a
                href={getPredictionsDownloadUrl(activeJobId!)}
                download
                className="inline-flex items-center gap-1.5 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3.5 py-2 text-xs font-semibold text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 shadow-sm"
              >
                <Download className="h-3.5 w-3.5 text-blue-500" />
                <span>Predictions CSV</span>
              </a>
              <a
                href={getModelDownloadUrl(activeJobId!)}
                download
                className="inline-flex items-center gap-1.5 rounded-xl bg-purple-600 px-3.5 py-2 text-xs font-semibold text-white shadow-sm hover:bg-purple-500"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Trained Pipeline (.pkl)</span>
              </a>
            </div>
          </div>

          {/* Metric KPI Cards */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-500">Evaluation Metrics</h3>

            {isRegression ? (
              /* Regression Metrics */
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">MAE (Mean Absolute Error)</span>
                  <p className="text-2xl font-extrabold text-blue-600 dark:text-blue-400 mt-1">
                    {results.metrics?.mae ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Average magnitude of error</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">RMSE</span>
                  <p className="text-2xl font-extrabold text-purple-600 dark:text-purple-400 mt-1">
                    {results.metrics?.rmse ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Penalizes large errors</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">R² Score</span>
                  <p className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">
                    {results.metrics?.r2 ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Proportion of variance explained</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">MAPE (%)</span>
                  <p className="text-2xl font-extrabold text-amber-600 dark:text-amber-400 mt-1">
                    {results.metrics?.mape !== undefined ? `${results.metrics.mape}%` : "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Mean percentage deviation</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">Median Abs Error</span>
                  <p className="text-2xl font-extrabold text-indigo-600 dark:text-indigo-400 mt-1">
                    {results.metrics?.medae ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Robust to outliers</p>
                </div>
              </div>
            ) : (
              /* Classification Metrics */
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">Accuracy</span>
                  <p className="text-2xl font-extrabold text-blue-600 dark:text-blue-400 mt-1">
                    {results.metrics?.accuracy ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Correct classification rate</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">Weighted F1</span>
                  <p className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400 mt-1">
                    {results.metrics?.weighted_f1 ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Harmonic mean of precision & recall</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">Precision</span>
                  <p className="text-2xl font-extrabold text-purple-600 dark:text-purple-400 mt-1">
                    {results.metrics?.precision ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Positive predictive value</p>
                </div>
                <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-5">
                  <span className="text-xs text-zinc-500">Recall</span>
                  <p className="text-2xl font-extrabold text-amber-600 dark:text-amber-400 mt-1">
                    {results.metrics?.recall ?? "—"}
                  </p>
                  <p className="text-[11px] text-zinc-400 mt-1">Sensitivity / true positive rate</p>
                </div>
              </div>
            )}
          </div>

          {/* Interactive Diagnostic Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Regression Chart 1: Actual vs Predicted */}
            {isRegression && results.charts?.scatter_points && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                      Actual vs. Predicted Values
                    </h3>
                    <p className="text-xs text-zinc-400">Values along the diagonal line indicate perfect prediction</p>
                  </div>
                  {results.charts?.scatter_plot_url && (
                    <a
                      href={results.charts.scatter_plot_url}
                      download="actual_vs_predicted.png"
                      className="text-xs font-semibold text-blue-600 hover:underline flex items-center gap-1"
                    >
                      <Download className="h-3.5 w-3.5" />
                      PNG
                    </a>
                  )}
                </div>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis type="number" dataKey="actual" name="Actual" label={{ value: "Actual", position: "insideBottom", offset: -10 }} />
                      <YAxis type="number" dataKey="predicted" name="Predicted" label={{ value: "Predicted", angle: -90, position: "insideLeft" }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter name="Samples" data={results.charts.scatter_points} fill="#2563eb" fillOpacity={0.6} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Regression Chart 2: Residual Plot */}
            {isRegression && results.charts?.residual_points && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                      Residual Plot (Errors Analysis)
                    </h3>
                    <p className="text-xs text-zinc-400">Residual = Actual − Predicted (should center evenly at 0)</p>
                  </div>
                  {results.charts?.residual_plot_url && (
                    <a
                      href={results.charts.residual_plot_url}
                      download="residual_analysis.png"
                      className="text-xs font-semibold text-amber-600 hover:underline flex items-center gap-1"
                    >
                      <Download className="h-3.5 w-3.5" />
                      PNG
                    </a>
                  )}
                </div>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis type="number" dataKey="predicted" name="Predicted" label={{ value: "Predicted", position: "insideBottom", offset: -10 }} />
                      <YAxis type="number" dataKey="residual" name="Residual" label={{ value: "Residual", angle: -90, position: "insideLeft" }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Scatter name="Residuals" data={results.charts.residual_points} fill="#f97316" fillOpacity={0.6} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Classification Diagnostic 1: Confusion Matrix */}
            {!isRegression && results.charts?.confusion_matrix && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">Confusion Matrix</h3>
                  {results.charts?.cm_plot_url && (
                    <a
                      href={results.charts.cm_plot_url}
                      download="confusion_matrix.png"
                      className="text-xs font-semibold text-blue-600 hover:underline flex items-center gap-1"
                    >
                      <Download className="h-3.5 w-3.5" />
                      PNG
                    </a>
                  )}
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-center text-xs border border-zinc-200 dark:border-zinc-800">
                    <thead className="bg-zinc-50 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
                      <tr>
                        <th className="p-2 border border-zinc-200 dark:border-zinc-800">Actual \ Predicted</th>
                        {results.charts.confusion_matrix.labels.map((lbl: string) => (
                          <th key={lbl} className="p-2 border border-zinc-200 dark:border-zinc-800 font-bold">
                            {lbl}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {results.charts.confusion_matrix.matrix.map((row: number[], rowIdx: number) => (
                        <tr key={rowIdx}>
                          <td className="p-2 font-bold bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-800">
                            {results.charts.confusion_matrix.labels[rowIdx]}
                          </td>
                          {row.map((val: number, colIdx: number) => (
                            <td
                              key={colIdx}
                              className={`p-2 border border-zinc-200 dark:border-zinc-800 font-mono font-bold ${
                                rowIdx === colIdx ? "bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300" : "text-zinc-600"
                              }`}
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Classification Diagnostic 2: ROC Curve */}
            {!isRegression && results.charts?.roc_data && (
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                      Receiver Operating Characteristic (ROC)
                    </h3>
                    <p className="text-xs text-zinc-400">AUC = {results.charts.roc_data.auc}</p>
                  </div>
                  {results.charts.roc_data.roc_plot_url && (
                    <a
                      href={results.charts.roc_data.roc_plot_url}
                      download="roc_curve.png"
                      className="text-xs font-semibold text-orange-600 hover:underline flex items-center gap-1"
                    >
                      <Download className="h-3.5 w-3.5" />
                      PNG
                    </a>
                  )}
                </div>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results.charts.roc_data.points}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "False Positive Rate", position: "insideBottom", offset: -5 }} />
                      <YAxis type="number" domain={[0, 1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Line type="monotone" dataKey="tpr" stroke="#ea580c" strokeWidth={2} dot={false} name="ROC Curve" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>

          {/* Feature Importance Bar Chart */}
          {results.feature_importance && results.feature_importance.length > 0 && (
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-4 shadow-sm">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-500" />
                <span>Feature Importance Ranking (Top 15 Predictors)</span>
              </h3>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={results.feature_importance}
                    margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={90} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Importance" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* K-Fold Details (if K-Fold was chosen) */}
          {results.folds && results.folds.length > 0 && (
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
              <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                <Layers className="h-5 w-5 text-blue-500" />
                <span>K-Fold Cross Validation Fold Details</span>
              </h3>

              {/* Fold Tabs */}
              <div className="flex gap-2 border-b border-zinc-200 dark:border-zinc-800 pb-2">
                {results.folds.map((f: any) => (
                  <button
                    key={f.fold}
                    type="button"
                    onClick={() => setSelectedFoldTab(f.fold)}
                    className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${
                      selectedFoldTab === f.fold
                        ? "bg-blue-600 text-white shadow-sm"
                        : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-400"
                    }`}
                  >
                    Fold {f.fold}
                  </button>
                ))}
              </div>

              {/* Fold Content */}
              {(() => {
                const foldData = results.folds.find((f: any) => f.fold === selectedFoldTab);
                if (!foldData) return null;
                return (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                      {Object.entries(foldData.metrics || {}).map(([k, v]: [string, any]) => (
                        <div key={k} className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-800/30">
                          <span className="text-zinc-400 uppercase">{k}</span>
                          <p className="text-xl font-bold text-zinc-800 dark:text-zinc-200 mt-0.5">{String(v)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* Prediction Preview Table */}
          {results.prediction_preview && (
            <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-4 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                    <Table className="h-5 w-5 text-emerald-500" />
                    <span>Prediction Preview & Full Dataset Export</span>
                  </h3>
                  <p className="text-xs text-zinc-400">
                    Previewing first 20 of {results.total_predictions} test samples (full features included)
                  </p>
                </div>
                <a
                  href={getPredictionsDownloadUrl(activeJobId!)}
                  download
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:bg-emerald-950/60 dark:text-emerald-300 px-3 py-1.5 text-xs font-semibold"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download Full CSV</span>
                </a>
              </div>

              <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-zinc-800">
                <table className="w-full text-left text-xs">
                  <thead className="bg-zinc-50 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
                    <tr>
                      <th className="px-4 py-2.5">Sample #</th>
                      <th className="px-4 py-2.5">Actual Value</th>
                      <th className="px-4 py-2.5">Predicted Value</th>
                      {!isRegression && <th className="px-4 py-2.5">Status</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800 font-mono">
                    {results.prediction_preview.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30">
                        <td className="px-4 py-2 text-zinc-400">{i + 1}</td>
                        <td className="px-4 py-2 font-bold text-zinc-800 dark:text-zinc-200">
                          {typeof row.Actual === "number" ? row.Actual.toFixed(4) : String(row.Actual)}
                        </td>
                        <td className="px-4 py-2 text-blue-600 dark:text-blue-400 font-bold">
                          {typeof row.Predicted === "number" ? row.Predicted.toFixed(4) : String(row.Predicted)}
                        </td>
                        {!isRegression && (
                          <td className="px-4 py-2">
                            {row.Correct ? (
                              <span className="text-emerald-600 font-bold">✓ Match</span>
                            ) : (
                              <span className="text-rose-500 font-bold">✗ Misclassified</span>
                            )}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 4. Live Model Inference & Interactive Playground */}
          <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
            <div>
              <h3 className="text-base sm:text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-indigo-500" />
                <span>Live Model Inference & Interactive Playground</span>
              </h3>
              <p className="text-xs text-zinc-400 mt-0.5">
                Test your trained pipeline live with custom feature values, or score a new batch CSV dataset.
              </p>
            </div>

            {/* Single Sample Interactive Form */}
            <div className="space-y-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                1. Single Sample Interactive Prediction
              </h4>

              {results.feature_schema && results.feature_schema.length > 0 ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                  {results.feature_schema.map((f) => (
                    <div key={f.name} className="space-y-1 text-xs">
                      <label className="font-semibold text-zinc-700 dark:text-zinc-300 truncate block">
                        {f.name}
                      </label>
                      {f.options && f.options.length > 0 ? (
                        <select
                          value={inferenceInputs[f.name] ?? f.sample ?? ""}
                          onChange={(e) =>
                            setInferenceInputs({ ...inferenceInputs, [f.name]: e.target.value })
                          }
                          className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100"
                        >
                          {f.options.map((opt) => (
                            <option key={opt} value={opt}>
                              {opt}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type={f.type === "number" ? "number" : "text"}
                          step="any"
                          placeholder={String(f.sample ?? "")}
                          value={inferenceInputs[f.name] ?? ""}
                          onChange={(e) =>
                            setInferenceInputs({
                              ...inferenceInputs,
                              [f.name]: f.type === "number" && e.target.value !== "" ? parseFloat(e.target.value) : e.target.value,
                            })
                          }
                          className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100"
                        />
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-zinc-400">
                  Feature schema generated with model. Enter values to test predictions.
                </p>
              )}

              <div className="flex items-center gap-3 pt-2">
                <button
                  type="button"
                  disabled={predicting || !activeJobId}
                  onClick={handleLivePredict}
                  className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-bold text-white shadow-md hover:bg-indigo-500 disabled:opacity-50 transition-all"
                >
                  <Send className="h-4 w-4" />
                  <span>{predicting ? "Predicting..." : "Generate Live Prediction"}</span>
                </button>
                {predictError && (
                  <span className="text-xs text-rose-500 font-semibold">{predictError}</span>
                )}
              </div>

              {/* Prediction Result Display */}
              {livePrediction && (
                <div className="rounded-xl border border-indigo-200 dark:border-indigo-900/60 bg-indigo-50/60 dark:bg-indigo-950/30 p-5 space-y-3">
                  <span className="text-xs font-semibold text-indigo-700 dark:text-indigo-300 uppercase tracking-wider">
                    Prediction Result
                  </span>
                  <div className="flex flex-col sm:flex-row sm:items-baseline gap-4">
                    <span className="text-3xl font-extrabold text-indigo-900 dark:text-indigo-100">
                      {typeof livePrediction.predictions[0] === "number"
                        ? livePrediction.predictions[0].toFixed(4)
                        : String(livePrediction.predictions[0])}
                    </span>
                    {livePrediction.classes && livePrediction.probabilities && (
                      <div className="text-xs space-y-1">
                        <p className="font-semibold text-zinc-600 dark:text-zinc-400">Class Probabilities:</p>
                        <div className="flex flex-wrap gap-2">
                          {livePrediction.classes.map((cls: string, idx: number) => {
                            const prob = livePrediction.probabilities[0]?.[idx] ?? 0;
                            return (
                              <span key={cls} className="rounded-md bg-white dark:bg-zinc-900 px-2 py-1 text-xs border border-zinc-200 dark:border-zinc-800">
                                <strong>{cls}:</strong> {(prob * 100).toFixed(1)}%
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Batch Scoring Section */}
            <div className="pt-4 border-t border-zinc-200 dark:border-zinc-800 space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                2. Batch Dataset Scoring (Upload CSV)
              </h4>
              <p className="text-xs text-zinc-400">
                Upload any CSV containing feature columns to compute predictions for all rows and download the scored dataset.
              </p>
              <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
                  className="text-xs text-zinc-500 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-zinc-100 dark:file:bg-zinc-800 file:text-zinc-700 dark:file:text-zinc-300 hover:file:bg-zinc-200"
                />
                <button
                  type="button"
                  disabled={!batchFile || batchPredicting || !activeJobId}
                  onClick={handleBatchPredict}
                  className="inline-flex items-center gap-2 rounded-xl bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 px-4 py-2 text-xs font-bold hover:bg-zinc-800 dark:hover:bg-zinc-200 disabled:opacity-40 transition-all shadow-sm"
                >
                  <Upload className="h-3.5 w-3.5" />
                  <span>{batchPredicting ? "Scoring Batch..." : "Score & Download CSV"}</span>
                </button>
              </div>
              {batchError && <p className="text-xs text-rose-500 font-semibold">{batchError}</p>}
              {batchResult && (
                <p className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold">
                  ✅ Successfully scored {batchResult.count} samples and downloaded results CSV!
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ConfigRunPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-sm text-zinc-500">Loading Training Studio...</div>}>
      <ConfigRunContent />
    </Suspense>
  );
}
