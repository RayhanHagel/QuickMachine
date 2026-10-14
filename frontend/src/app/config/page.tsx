"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Sliders,
  Database,
  Layers,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  TrendingUp,
  Save
} from "lucide-react";
import {
  listDataConfigs,
  inspectDataset,
  getModelOptions,
  savePipelineConfig,
  ConfiguredDataset,
  DatasetInspection
} from "@/lib/api";

function ConfigCreateContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialDataset = searchParams.get("dataset") || "";

  // Datasets & inspection
  const [datasets, setDatasets] = useState<ConfiguredDataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>(initialDataset);
  const [inspection, setInspection] = useState<DatasetInspection | null>(null);
  const [loadingInspection, setLoadingInspection] = useState(false);

  // Model Options Registry Schema
  const [modelOptions, setModelOptions] = useState<Record<string, any>>({});

  // File Configuration & Preprocessing state
  const [missingStrategy, setMissingStrategy] = useState<string>("Drop Rows");
  const [scalerChosen, setScalerChosen] = useState<string>("Standard Scaler");
  const [categoricalEncChosen, setCategoricalEncChosen] = useState<string>("Ordinal Encoder");
  const [trainSplit, setTrainSplit] = useState<number>(80);
  const [testSplit, setTestSplit] = useState<number>(20);

  // General Model Config
  const [randomState, setRandomState] = useState<number>(42);
  const [trainEpoch, setTrainEpoch] = useState<number>(500);
  const [kfoldAmount, setKfoldAmount] = useState<number>(5);
  const [batchSize, setBatchSize] = useState<number>(32);

  // Target Config
  const [targetChosen, setTargetChosen] = useState<"Regression" | "Classification">("Regression");

  // Stacking vs Single Model
  const [useStacking, setUseStacking] = useState<"Use" | "Ignore">("Ignore");
  const [baseModelsChosen, setBaseModelsChosen] = useState<string[]>([]);
  const [metaLibraryChosen, setMetaLibraryChosen] = useState<string>("scikit-learn");
  const [metaModelChosen, setMetaModelChosen] = useState<string>("");
  const [stackingPassthrough, setStackingPassthrough] = useState<"Disabled" | "Enabled">("Disabled");

  // Single Model selection
  const [singleLibrary, setSingleLibrary] = useState<string>("scikit-learn");
  const [singleModel, setSingleModel] = useState<string>("");

  // Dynamic Hyperparameters state
  // key: base_ModelName_paramName_min_idx, etc.
  const [hyperparams, setHyperparams] = useState<Record<string, any>>({});

  // Tuner Configurations
  const [hypertunerChosen, setHypertunerChosen] = useState<string>("Optuna");
  const [hypertuneTrial, setHypertuneTrial] = useState<number>(50);
  const [hypertuneEpoch, setHypertuneEpoch] = useState<number>(100);
  const [tuningObjective, setTuningObjective] = useState<string>("val_loss");
  const [tuningDirection, setTuningDirection] = useState<string>("minimize");
  const [tuningTimeout, setTuningTimeout] = useState<number>(3600);
  const [tuningInitialPoints, setTuningInitialPoints] = useState<number>(5);
  const [useEarlyStopping, setUseEarlyStopping] = useState<"Enabled" | "Disabled">("Enabled");

  // Save Configuration State
  const [configName, setConfigName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Data Preview active tab
  const [previewTab, setPreviewTab] = useState<"numerical" | "categorical" | "target">("numerical");

  // Load datasets and model options on mount
  useEffect(() => {
    async function init() {
      try {
        const [dsList, mOptions] = await Promise.all([
          listDataConfigs(),
          getModelOptions(),
        ]);
        setDatasets(dsList);
        setModelOptions(mOptions);

        if (!selectedDataset && dsList.length > 0) {
          setSelectedDataset(dsList[0].name);
        }
      } catch (err: any) {
        console.error(err);
      }
    }
    init();
  }, []);

  // Inspect dataset when selection changes
  useEffect(() => {
    if (!selectedDataset) {
      setInspection(null);
      return;
    }
    async function fetchInspection() {
      setLoadingInspection(true);
      try {
        const data = await inspectDataset(selectedDataset);
        setInspection(data);
        if (data.target_type_hint) {
          setTargetChosen(data.target_type_hint);
          setTuningObjective(data.target_type_hint === "Regression" ? "val_loss" : "val_accuracy");
          setTuningDirection(data.target_type_hint === "Regression" ? "minimize" : "maximize");
        }
      } catch (err: any) {
        setFeedback({ type: "error", message: err.message || "Failed to inspect dataset" });
      } finally {
        setLoadingInspection(false);
      }
    }
    fetchInspection();
  }, [selectedDataset]);

  // Train / Test split sync
  const handleTrainSplitChange = (val: number) => {
    const clamped = Math.max(1, Math.min(99, val));
    setTrainSplit(clamped);
    setTestSplit(100 - clamped);
  };

  const handleTestSplitChange = (val: number) => {
    const clamped = Math.max(1, Math.min(99, val));
    setTestSplit(clamped);
    setTrainSplit(100 - clamped);
  };

  // Helper to get available models for target and library
  const getAvailableModels = (target: string, library: string) => {
    if (!modelOptions[target] || !modelOptions[target][library]) return [];
    return Object.keys(modelOptions[target][library]);
  };

  // Helper to get all models for target
  const getAllTargetModels = (target: string) => {
    if (!modelOptions[target]) return [];
    const list: string[] = [];
    for (const lib of Object.keys(modelOptions[target])) {
      list.push(...Object.keys(modelOptions[target][lib]));
    }
    return list;
  };

  // Render hyperparameter controls for a model
  const renderModelHyperparams = (modelType: "base" | "meta", modelName: string) => {
    if (!modelOptions[targetChosen]) return null;

    let params: Record<string, any> = {};
    for (const lib of Object.keys(modelOptions[targetChosen])) {
      if (modelOptions[targetChosen][lib][modelName]) {
        params = modelOptions[targetChosen][lib][modelName].params || {};
        break;
      }
    }

    const paramKeys = Object.keys(params);
    if (paramKeys.length === 0) return null;

    return (
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-800/30 p-4 space-y-4">
        <h4 className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
          {modelName} Hyperparameter Search Space
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {paramKeys.map((pName, idx) => {
            const criteria = params[pName];
            const isNumeric =
              Array.isArray(criteria) &&
              criteria.length === 2 &&
              criteria.every((v: any) => typeof v === "number");

            const minKey = `${modelType}_${modelName}_${pName}_min_${idx}`;
            const maxKey = `${modelType}_${modelName}_${pName}_max_${idx}`;
            const optKey = `${modelType}_${modelName}_${pName}_option_${idx}`;

            if (isNumeric) {
              const currentMin = hyperparams[minKey] ?? criteria[0];
              const currentMax = hyperparams[maxKey] ?? criteria[1];
              const isFloat = criteria.some((v: any) => !Number.isInteger(v));

              return (
                <div
                  key={pName}
                  className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-3 space-y-2"
                >
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">{pName}</span>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <label className="text-[10px] text-zinc-400">Min</label>
                      <input
                        type="number"
                        step={isFloat ? "0.001" : "1"}
                        value={currentMin}
                        onChange={(e) =>
                          setHyperparams({
                            ...hyperparams,
                            [minKey]: isFloat ? parseFloat(e.target.value) : parseInt(e.target.value),
                          })
                        }
                        className="w-full rounded border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-2 py-1 text-xs"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-zinc-400">Max</label>
                      <input
                        type="number"
                        step={isFloat ? "0.001" : "1"}
                        value={currentMax}
                        onChange={(e) =>
                          setHyperparams({
                            ...hyperparams,
                            [maxKey]: isFloat ? parseFloat(e.target.value) : parseInt(e.target.value),
                          })
                        }
                        className="w-full rounded border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-2 py-1 text-xs"
                      />
                    </div>
                  </div>
                </div>
              );
            } else {
              const currentOptions = hyperparams[optKey] ?? criteria;
              return (
                <div
                  key={pName}
                  className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-3 space-y-2"
                >
                  <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">{pName}</span>
                  <div className="flex flex-wrap gap-1">
                    {criteria.map((optVal: any) => {
                      const isSel = Array.isArray(currentOptions) && currentOptions.includes(optVal);
                      return (
                        <button
                          key={String(optVal)}
                          type="button"
                          onClick={() => {
                            const next = isSel
                              ? currentOptions.filter((o: any) => o !== optVal)
                              : [...(currentOptions || []), optVal];
                            setHyperparams({
                              ...hyperparams,
                              [optKey]: next,
                            });
                          }}
                          className={`rounded px-2 py-0.5 text-[11px] font-medium border ${
                            isSel
                              ? "bg-blue-600 text-white border-blue-600"
                              : "bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border-zinc-200 dark:border-zinc-700"
                          }`}
                        >
                          {String(optVal)}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            }
          })}
        </div>
      </div>
    );
  };

  // Build full config payload
  const buildConfigPayload = () => {
    const payload: Record<string, any> = {
      chosen_configuration: selectedDataset,
      missing_value_processing: missingStrategy,
      scaler_chosen: scalerChosen,
      categorical_enc_chosen: categoricalEncChosen,
      train_split: trainSplit,
      test_split: testSplit,
      random_state: randomState,
      train_epoch: trainEpoch,
      kfold_amount: kfoldAmount,
      batch_size: batchSize,
      target_chosen: targetChosen,
      use_stacking: useStacking,
      hypertuner_chosen: hypertunerChosen,
      hypertune_trial: hypertuneTrial,
      hypertune_epoch: hypertuneEpoch,
      tuning_objective: tuningObjective,
      tuning_direction: tuningDirection,
      tuning_timeout: tuningTimeout,
      tuning_initial_points: tuningInitialPoints,
      use_early_stopping: useEarlyStopping,
      ...hyperparams,
    };

    if (useStacking === "Use") {
      payload.base_model_chosen = baseModelsChosen;
      payload.meta_library_chosen = metaLibraryChosen;
      payload.meta_model_chosen = metaModelChosen;
      payload.stacking_passthrough = stackingPassthrough;
    } else {
      payload.library_chosen = singleLibrary;
      payload.model_chosen = singleModel;
    }

    return payload;
  };

  // Save Pipeline Configuration
  const handleSaveConfig = async () => {
    if (!configName.trim()) {
      setFeedback({ type: "error", message: "Please provide a configuration name" });
      return;
    }
    if (!selectedDataset) {
      setFeedback({ type: "error", message: "Please select a dataset configuration first" });
      return;
    }
    if (useStacking === "Use" && (baseModelsChosen.length === 0 || !metaModelChosen)) {
      setFeedback({ type: "error", message: "Please select at least one base model and a meta learner" });
      return;
    }
    if (useStacking === "Ignore" && !singleModel) {
      setFeedback({ type: "error", message: "Please select a machine learning model" });
      return;
    }

    setIsSaving(true);
    setFeedback(null);
    try {
      const payload = buildConfigPayload();
      await savePipelineConfig(configName.trim(), payload);
      setFeedback({
        type: "success",
        message: `Pipeline configuration '${configName.trim()}' saved with embedded dataset!`,
      });
    } catch (err: any) {
      setFeedback({ type: "error", message: err.message || "Failed to save configuration" });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-10">
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-zinc-900 dark:text-zinc-50 flex items-center gap-3">
          <Sliders className="h-8 w-8 text-purple-600" />
          <span>Model Architecture & Hyperparameter Configuration</span>
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Select your dataset, configure preprocessing (scalers, encoders, missing values), build single models or Stacking ensembles, and define Bayesian optimization search spaces.
        </p>
      </div>

      {feedback && (
        <div
          className={`flex items-start justify-between rounded-xl p-4 border text-sm ${
            feedback.type === "success"
              ? "bg-emerald-50 border-emerald-200 text-emerald-800 dark:bg-emerald-950/40 dark:border-emerald-800 dark:text-emerald-300"
              : "bg-rose-50 border-rose-200 text-rose-800 dark:bg-rose-950/40 dark:border-rose-800 dark:text-rose-300"
          }`}
        >
          <div className="flex items-center gap-2.5">
            {feedback.type === "success" ? <CheckCircle className="h-5 w-5" /> : <AlertCircle className="h-5 w-5" />}
            <span>{feedback.message}</span>
          </div>
          {feedback.type === "success" && (
            <button
              onClick={() => router.push(`/run?config=${encodeURIComponent(configName.trim())}`)}
              className="inline-flex items-center gap-1.5 font-bold text-emerald-700 dark:text-emerald-300 hover:underline"
            >
              <span>Proceed to Training & Run</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          )}
        </div>
      )}

      {/* 1. Dataset Selection & Inspection */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <Database className="h-5 w-5 text-blue-500" />
          <span>1. Dataset & Preprocessing Setup</span>
        </h2>

        {/* Dataset dropdown */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Select Configured Dataset
          </label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="w-full rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-4 py-2.5 text-sm font-semibold text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">-- Choose a dataset --</option>
            {datasets.map((ds) => (
              <option key={ds.name} value={ds.name}>
                {ds.name} ({ds.feature_count} features, {ds.target_count} target)
              </option>
            ))}
          </select>
        </div>

        {loadingInspection && (
          <p className="text-sm text-blue-600 animate-pulse">Loading and inspecting dataset...</p>
        )}

        {inspection && (
          <div className="space-y-6 pt-2">
            {/* Summary KPI Badges */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-4">
                <span className="text-xs text-zinc-500">Data Points</span>
                <p className="text-2xl font-bold text-zinc-900 dark:text-zinc-100 mt-1">{inspection.total_rows}</p>
              </div>
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-4">
                <span className="text-xs text-zinc-500">Numerical Features</span>
                <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">{inspection.numerical_features.length}</p>
              </div>
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-4">
                <span className="text-xs text-zinc-500">Categorical Features</span>
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">{inspection.categorical_features.length}</p>
              </div>
              <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-800/40 p-4">
                <span className="text-xs text-zinc-500">Missing Values</span>
                <p className="text-2xl font-bold text-amber-600 dark:text-amber-400 mt-1">{inspection.missing_values.cell_missing_pct}%</p>
              </div>
            </div>

            {/* Missing Value Processing */}
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Missing Value Handling Strategy
              </label>
              <div className="flex flex-wrap gap-2">
                {["Drop Rows", "Impute (Mean)", "Impute (Median)", "Impute (Mode)"].map((strat) => (
                  <button
                    key={strat}
                    type="button"
                    onClick={() => setMissingStrategy(strat)}
                    className={`rounded-xl px-4 py-2 text-xs font-bold border transition-all ${
                      missingStrategy === strat
                        ? "border-blue-600 bg-blue-50 text-blue-700 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-400 shadow-sm"
                        : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                    }`}
                  >
                    {strat}
                  </button>
                ))}
              </div>
              {inspection.missing_values.cell_missing_count > 0 && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  Detected {inspection.missing_values.cell_missing_count} missing cells ({inspection.missing_values.cell_missing_pct}%) across {inspection.missing_values.row_missing_count} rows.
                </p>
              )}
            </div>

            {/* Scalers & Encoders */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Scaler */}
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  Numerical Feature Scaler
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {inspection.available_scalers.map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setScalerChosen(s)}
                      className={`rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                        scalerChosen === s
                          ? "border-blue-600 bg-blue-50 text-blue-700 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-400 font-bold"
                          : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                      }`}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>

              {/* Categorical Encoder */}
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  Categorical Feature Encoder
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {inspection.available_encoders.map((enc) => (
                    <button
                      key={enc}
                      type="button"
                      onClick={() => setCategoricalEncChosen(enc)}
                      className={`rounded-lg px-3 py-1.5 text-xs font-medium border transition-all ${
                        categoricalEncChosen === enc
                          ? "border-purple-600 bg-purple-50 text-purple-700 dark:border-purple-500 dark:bg-purple-950/50 dark:text-purple-400 font-bold"
                          : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                      }`}
                    >
                      {enc}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Train / Test Split Sliders */}
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Train-Test Split Percentage
              </label>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-medium">
                    <span>Train-Validation: {trainSplit}%</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="95"
                    value={trainSplit}
                    onChange={(e) => handleTrainSplitChange(parseInt(e.target.value))}
                    className="w-full accent-blue-600"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-medium">
                    <span>Test Split: {testSplit}%</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="90"
                    value={testSplit}
                    onChange={(e) => handleTestSplitChange(parseInt(e.target.value))}
                    className="w-full accent-purple-600"
                  />
                </div>
              </div>
            </div>

            {/* Data Previews */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center gap-2 border-b border-zinc-200 dark:border-zinc-800 pb-2">
                {(["numerical", "categorical", "target"] as const).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setPreviewTab(tab)}
                    className={`px-3 py-1 text-xs font-bold rounded-lg capitalize ${
                      previewTab === tab
                        ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                        : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
                    }`}
                  >
                    {tab} Preview
                  </button>
                ))}
              </div>

              <div className="overflow-x-auto max-h-48 border border-zinc-200 dark:border-zinc-800 rounded-xl">
                <table className="w-full text-left text-xs">
                  <thead className="bg-zinc-50 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300">
                    <tr>
                      {inspection.previews[previewTab].length > 0 &&
                        Object.keys(inspection.previews[previewTab][0]).map((col) => (
                          <th key={col} className="px-3 py-2 whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                    {inspection.previews[previewTab].map((row, i) => (
                      <tr key={i} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30">
                        {Object.values(row).map((val: any, j) => (
                          <td key={j} className="px-3 py-1.5 whitespace-nowrap text-zinc-700 dark:text-zinc-300">
                            {String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 2. Target Type & Stacking Configuration */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <Layers className="h-5 w-5 text-purple-500" />
          <span>2. Model Architecture & Ensembles</span>
        </h2>

        {/* General Hyperparams Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Random State</label>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">K-Fold Amount</label>
            <input
              type="number"
              min="2"
              max="10"
              value={kfoldAmount}
              onChange={(e) => setKfoldAmount(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Training Epoch</label>
            <input
              type="number"
              value={trainEpoch}
              onChange={(e) => setTrainEpoch(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
        </div>

        {/* Target Type */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Target Type
          </label>
          <div className="flex gap-2">
            {(["Regression", "Classification"] as const).map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => {
                  setTargetChosen(t);
                  setBaseModelsChosen([]);
                  setSingleModel("");
                  setTuningObjective(t === "Regression" ? "val_loss" : "val_accuracy");
                  setTuningDirection(t === "Regression" ? "minimize" : "maximize");
                }}
                className={`rounded-xl px-5 py-2.5 text-xs font-bold border transition-all ${
                  targetChosen === t
                    ? "border-purple-600 bg-purple-50 text-purple-700 dark:border-purple-500 dark:bg-purple-950/50 dark:text-purple-300 shadow-sm"
                    : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Stacking Mode */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Use Model Stacking Ensemble
          </label>
          <div className="flex gap-2">
            {(["Ignore", "Use"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setUseStacking(mode)}
                className={`rounded-xl px-5 py-2.5 text-xs font-bold border transition-all ${
                  useStacking === mode
                    ? "border-emerald-600 bg-emerald-50 text-emerald-700 dark:border-emerald-500 dark:bg-emerald-950/50 dark:text-emerald-300 shadow-sm"
                    : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                }`}
              >
                {mode === "Use" ? "Stacking Ensemble (Multi-Model)" : "Single Model"}
              </button>
            ))}
          </div>
        </div>

        {/* Stacking Models Configuration */}
        {useStacking === "Use" ? (
          <div className="space-y-6 rounded-xl border border-purple-200 dark:border-purple-900 bg-purple-50/30 dark:bg-purple-950/20 p-5">
            {/* Base Models */}
            <div className="space-y-2">
              <label className="text-xs font-bold uppercase tracking-wider text-zinc-600 dark:text-zinc-300">
                Select Base Learners ({baseModelsChosen.length} selected)
              </label>
              <div className="flex flex-wrap gap-2">
                {getAllTargetModels(targetChosen).map((mdl) => {
                  const isSel = baseModelsChosen.includes(mdl);
                  return (
                    <button
                      key={mdl}
                      type="button"
                      onClick={() => {
                        const next = isSel
                          ? baseModelsChosen.filter((m) => m !== mdl)
                          : [...baseModelsChosen, mdl];
                        setBaseModelsChosen(next);
                      }}
                      className={`rounded-lg px-3 py-1.5 text-xs font-semibold border transition-all ${
                        isSel
                          ? "bg-purple-600 text-white border-purple-600 shadow-sm"
                          : "bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100"
                      }`}
                    >
                      {mdl}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Meta Model */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Meta Learner Library</label>
                <select
                  value={metaLibraryChosen}
                  onChange={(e) => {
                    setMetaLibraryChosen(e.target.value);
                    setMetaModelChosen("");
                  }}
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-2 text-xs font-semibold"
                >
                  {["scikit-learn", "pytorch", "tensorflow", "others"].map((lib) => (
                    <option key={lib} value={lib}>
                      {lib === "pytorch"
                        ? "PyTorch"
                        : lib === "tensorflow"
                        ? "TensorFlow"
                        : lib === "scikit-learn"
                        ? "Scikit-Learn"
                        : "Boosters (XGB/LGB/Cat)"}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Meta Model</label>
                <select
                  value={metaModelChosen}
                  onChange={(e) => setMetaModelChosen(e.target.value)}
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-2 text-xs font-semibold"
                >
                  <option value="">-- Choose Meta Model --</option>
                  {getAvailableModels(targetChosen, metaLibraryChosen).map((mdl) => (
                    <option key={mdl} value={mdl}>
                      {mdl}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Passthrough */}
            <div className="space-y-1">
              <label className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Meta-Model Passthrough</label>
              <div className="flex gap-2">
                {(["Disabled", "Enabled"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setStackingPassthrough(mode)}
                    className={`rounded-lg px-3 py-1 text-xs font-semibold border ${
                      stackingPassthrough === mode
                        ? "bg-purple-600 text-white border-purple-600"
                        : "bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border-zinc-300 dark:border-zinc-700"
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Hyperparameters for Base Learners */}
            {baseModelsChosen.length > 0 && (
              <div className="space-y-4 pt-2">
                <h3 className="text-sm font-bold text-zinc-800 dark:text-zinc-200">
                  Base Learner Search Spaces
                </h3>
                {baseModelsChosen.map((m) => (
                  <div key={m}>{renderModelHyperparams("base", m)}</div>
                ))}
              </div>
            )}

            {/* Hyperparameters for Meta Learner */}
            {metaModelChosen && (
              <div className="space-y-4 pt-2">
                <h3 className="text-sm font-bold text-zinc-800 dark:text-zinc-200">
                  Meta Learner Search Space
                </h3>
                {renderModelHyperparams("meta", metaModelChosen)}
              </div>
            )}
          </div>
        ) : (
          /* Single Model Configuration */
          <div className="space-y-6 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/40 dark:bg-zinc-800/20 p-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Model Library</label>
                <div className="flex flex-wrap gap-2">
                  {["scikit-learn", "pytorch", "tensorflow", "others"].map((lib) => (
                    <button
                      key={lib}
                      type="button"
                      onClick={() => {
                        setSingleLibrary(lib);
                        setSingleModel("");
                      }}
                      className={`rounded-lg px-3 py-1.5 text-xs font-semibold border ${
                        singleLibrary === lib
                          ? "bg-blue-600 text-white border-blue-600"
                          : "bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100"
                      }`}
                    >
                      {lib === "pytorch"
                        ? "PyTorch"
                        : lib === "tensorflow"
                        ? "TensorFlow"
                        : lib === "scikit-learn"
                        ? "Scikit-Learn"
                        : "Boosters (Others)"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-zinc-600 dark:text-zinc-300">Machine Learning Model</label>
                <select
                  value={singleModel}
                  onChange={(e) => setSingleModel(e.target.value)}
                  className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-2 text-xs font-semibold"
                >
                  <option value="">-- Choose Model --</option>
                  {getAvailableModels(targetChosen, singleLibrary).map((mdl) => (
                    <option key={mdl} value={mdl}>
                      {mdl}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {singleModel && renderModelHyperparams("base", singleModel)}
          </div>
        )}
      </div>

      {/* 3. Tuner Configuration */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-500" />
          <span>3. Bayesian Hypertuner Setup</span>
        </h2>

        {/* Tuner library */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
            Hypertuner Engine
          </label>
          <div className="flex gap-2">
            {["Optuna", "KT Bayesian Optimization"].map((tuner) => (
              <button
                key={tuner}
                type="button"
                onClick={() => setHypertunerChosen(tuner)}
                className={`rounded-xl px-5 py-2.5 text-xs font-bold border transition-all ${
                  hypertunerChosen === tuner
                    ? "border-emerald-600 bg-emerald-50 text-emerald-700 dark:border-emerald-500 dark:bg-emerald-950/50 dark:text-emerald-300 shadow-sm"
                    : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                }`}
              >
                {tuner}
              </button>
            ))}
          </div>
        </div>

        {/* Trials & Epochs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Trial Amount</label>
            <input
              type="number"
              min="1"
              max="200"
              value={hypertuneTrial}
              onChange={(e) => setHypertuneTrial(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Tuning Epoch</label>
            <input
              type="number"
              value={hypertuneEpoch}
              onChange={(e) => setHypertuneEpoch(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Objective Metric</label>
            <select
              value={tuningObjective}
              onChange={(e) => setTuningObjective(e.target.value)}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm font-medium"
            >
              {targetChosen === "Regression" ? (
                <>
                  <option value="val_loss">val_loss (MAE)</option>
                  <option value="val_mae">val_mae</option>
                  <option value="r2">r2 score</option>
                </>
              ) : (
                <>
                  <option value="val_accuracy">val_accuracy</option>
                  <option value="f1_score">f1_score</option>
                  <option value="val_loss">val_loss</option>
                </>
              )}
            </select>
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Direction</label>
            <select
              value={tuningDirection}
              onChange={(e) => setTuningDirection(e.target.value)}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm font-medium"
            >
              <option value="minimize">minimize</option>
              <option value="maximize">maximize</option>
            </select>
          </div>
        </div>

        {/* Advanced Tuner Settings */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Timeout (Seconds)</label>
            <input
              type="number"
              value={tuningTimeout}
              onChange={(e) => setTuningTimeout(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Initial Random Trials</label>
            <input
              type="number"
              value={tuningInitialPoints}
              onChange={(e) => setTuningInitialPoints(parseInt(e.target.value))}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm"
            />
          </div>
          <div className="space-y-1 text-xs">
            <label className="font-semibold text-zinc-500">Early Stopping Mode</label>
            <select
              value={useEarlyStopping}
              onChange={(e) => setUseEarlyStopping(e.target.value as any)}
              className="w-full rounded-lg border border-zinc-300 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 p-2 text-sm font-medium"
            >
              <option value="Enabled">Enabled (MedianPruner)</option>
              <option value="Disabled">Disabled</option>
            </select>
          </div>
        </div>
      </div>

      {/* 4. Save Configuration & Dataset Embedding */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-4 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <Save className="h-5 w-5 text-blue-600" />
          <span>4. Save Configuration</span>
        </h2>

        <div className="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            placeholder="Enter configuration name (e.g. Lithium_RF_XGB_Ridge)"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            className="flex-1 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-4 py-2.5 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="button"
            disabled={isSaving}
            onClick={handleSaveConfig}
            className={`flex items-center justify-center gap-2 rounded-xl px-6 py-2.5 text-sm font-bold shadow-md transition-all ${
              !isSaving
                ? "bg-purple-600 text-white hover:bg-purple-500 active:scale-95"
                : "bg-zinc-200 text-zinc-400 cursor-not-allowed"
            }`}
          >
            <CheckCircle className="h-4 w-4" />
            <span>{isSaving ? "Saving Configuration..." : "Save Configuration (.pkl)"}</span>
          </button>
        </div>
        <p className="text-xs text-zinc-400">
          Saves complete parameter definitions and embeds the cleaned feature/target dataset into the <code>saved_configs/</code> directory for immediate reproducible training.
        </p>
      </div>
    </div>
  );
}

export default function ConfigCreatePage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-sm text-zinc-500">Loading Configuration Studio...</div>}>
      <ConfigCreateContent />
    </Suspense>
  );
}

