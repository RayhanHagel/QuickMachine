"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Database,
  Sliders,
  PlayCircle,
  ArrowRight,
  Sparkles,
  Layers,
  CheckCircle2
} from "lucide-react";
import { listDataConfigs, listSavedConfigs, ConfiguredDataset, SavedConfigSummary } from "@/lib/api";

export default function HomePage() {
  const [datasets, setDatasets] = useState<ConfiguredDataset[]>([]);
  const [savedConfigs, setSavedConfigs] = useState<SavedConfigSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [d, c] = await Promise.all([
          listDataConfigs().catch(() => []),
          listSavedConfigs().catch(() => []),
        ]);
        setDatasets(d);
        setSavedConfigs(c);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-900 via-indigo-900 to-zinc-950 p-8 sm:p-12 text-white shadow-xl">
        <div className="relative z-10 max-w-3xl space-y-5">
          <div className="inline-flex items-center gap-2 rounded-full border border-blue-400/30 bg-blue-500/10 px-3.5 py-1 text-xs font-semibold text-blue-300 backdrop-blur-md">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Modern AutoML & Stacking Studio</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
            Automated Machine Learning with <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300">FastAPI & Next.js</span>
          </h1>

          <p className="text-base sm:text-lg text-blue-100/80 leading-relaxed">
            QuickMachine simplifies model training, hyperparameter optimization via Bayesian algorithms (Optuna & Keras-Tuner),
            multi-model stacking ensembles, and rich model evaluations — completely migrated to a high-performance modern web stack.
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <Link
              href="/data"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-500 px-5 py-3 text-sm font-semibold text-white shadow-lg hover:bg-blue-400 transition-all hover:scale-105 active:scale-95"
            >
              <span>Get Started: Data Prep</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/run"
              className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/10 px-5 py-3 text-sm font-semibold text-white backdrop-blur-md hover:bg-white/20 transition-all"
            >
              <PlayCircle className="h-4 w-4 text-cyan-300" />
              <span>Run Existing Models</span>
            </Link>
          </div>
        </div>

        {/* Ambient background blur */}
        <div className="absolute -right-20 -top-20 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-20 right-40 h-80 w-80 rounded-full bg-cyan-500/20 blur-3xl pointer-events-none" />
      </section>

      {/* Metrics / Status Banner */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">Configured Datasets</p>
            <p className="text-3xl font-bold text-zinc-900 dark:text-zinc-50 mt-1">
              {loading ? "..." : datasets.length}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Ready for feature configuration</p>
          </div>
          <div className="h-12 w-12 rounded-xl bg-blue-50 dark:bg-blue-950/60 flex items-center justify-center text-blue-600 dark:text-blue-400">
            <Database className="h-6 w-6" />
          </div>
        </div>

        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">Saved Configurations</p>
            <p className="text-3xl font-bold text-zinc-900 dark:text-zinc-50 mt-1">
              {loading ? "..." : savedConfigs.length}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Stacking & single model setups</p>
          </div>
          <div className="h-12 w-12 rounded-xl bg-purple-50 dark:bg-purple-950/60 flex items-center justify-center text-purple-600 dark:text-purple-400">
            <Sliders className="h-6 w-6" />
          </div>
        </div>

        <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">Model Engines</p>
            <p className="text-3xl font-bold text-zinc-900 dark:text-zinc-50 mt-1">12+</p>
            <p className="text-xs text-zinc-400 mt-1">Scikit-Learn, XGBoost, LightGBM, CatBoost</p>
          </div>
          <div className="h-12 w-12 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
            <Layers className="h-6 w-6" />
          </div>
        </div>
      </section>

      {/* 3-Step Guided Workflow */}
      <section className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">AutoML Workflow</h2>
          <p className="text-sm text-zinc-500 dark:text-zinc-400">From raw data files to trained ensemble pipelines in three seamless steps.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Step 1 */}
          <Link
            href="/data"
            className="group relative rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm hover:shadow-md hover:border-blue-500/50 transition-all flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/60 font-bold text-xs text-blue-700 dark:text-blue-300">
                  01
                </span>
                <Database className="h-5 w-5 text-zinc-400 group-hover:text-blue-600 transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 group-hover:text-blue-600 transition-colors">
                Data Preparation
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Upload CSV or Excel files. Assign columns into Features, Targets, and Ignored sets with smart role validation.
              </p>
            </div>
            <div className="pt-6 flex items-center gap-1.5 text-xs font-semibold text-blue-600 dark:text-blue-400">
              <span>Configure Data</span>
              <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>

          {/* Step 2 */}
          <Link
            href="/config"
            className="group relative rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm hover:shadow-md hover:border-purple-500/50 transition-all flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/60 font-bold text-xs text-purple-700 dark:text-purple-300">
                  02
                </span>
                <Sliders className="h-5 w-5 text-zinc-400 group-hover:text-purple-600 transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 group-hover:text-purple-600 transition-colors">
                Model Configuration
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Inspect missing values, pick imputation & scaling strategies, design multi-model Stacking ensembles, and define Bayesian hyperparameter search spaces.
              </p>
            </div>
            <div className="pt-6 flex items-center gap-1.5 text-xs font-semibold text-purple-600 dark:text-purple-400">
              <span>Set Hyperparameters</span>
              <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>

          {/* Step 3 */}
          <Link
            href="/run"
            className="group relative rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 shadow-sm hover:shadow-md hover:border-emerald-500/50 transition-all flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100 dark:bg-emerald-900/60 font-bold text-xs text-emerald-700 dark:text-emerald-300">
                  03
                </span>
                <PlayCircle className="h-5 w-5 text-zinc-400 group-hover:text-emerald-600 transition-colors" />
              </div>
              <h3 className="text-lg font-bold text-zinc-900 dark:text-zinc-50 group-hover:text-emerald-600 transition-colors">
                Train & Evaluate
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400 leading-relaxed">
                Watch Optuna tuning live with real-time trial progression charts. Inspect MAE, RMSE, R², Confusion Matrices, ROC Curves, and download trained pipelines (.pkl).
              </p>
            </div>
            <div className="pt-6 flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
              <span>Start Training</span>
              <ArrowRight className="h-3.5 w-3.5 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        </div>
      </section>

      {/* Existing Saved Configurations Table */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50">Existing Configurations</h2>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Pre-saved pipelines in your workspace ready for training or inspection.</p>
          </div>
          <Link
            href="/config"
            className="text-xs font-semibold text-blue-600 hover:text-blue-500 dark:text-blue-400"
          >
            Create New Config →
          </Link>
        </div>

        {loading ? (
          <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-8 text-center text-sm text-zinc-500">
            Loading configurations...
          </div>
        ) : savedConfigs.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-300 dark:border-zinc-800 p-8 text-center space-y-3">
            <Sliders className="h-8 w-8 text-zinc-400 mx-auto" />
            <p className="text-sm text-zinc-500">No saved configurations found yet.</p>
            <Link
              href="/config"
              className="inline-flex items-center gap-1 text-xs font-semibold text-blue-600 hover:underline"
            >
              Create your first configuration
            </Link>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 shadow-sm">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-zinc-200 bg-zinc-50 text-xs font-semibold text-zinc-600 dark:border-zinc-800 dark:bg-zinc-800/50 dark:text-zinc-400">
                <tr>
                  <th className="px-5 py-3">Configuration Name</th>
                  <th className="px-5 py-3">Target Type</th>
                  <th className="px-5 py-3">Architecture</th>
                  <th className="px-5 py-3">Tuning Trials</th>
                  <th className="px-5 py-3">Dataset</th>
                  <th className="px-5 py-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {savedConfigs.map((cfg) => (
                  <tr key={cfg.name} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30 transition-colors">
                    <td className="px-5 py-3 font-semibold text-zinc-900 dark:text-zinc-100">
                      {cfg.name}
                    </td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                        cfg.target_chosen === "Classification"
                          ? "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300"
                          : "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                      }`}>
                        {cfg.target_chosen}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-zinc-600 dark:text-zinc-400">
                      {cfg.use_stacking === "Use" ? (
                        <span className="font-medium text-emerald-600 dark:text-emerald-400">
                          Stacking ({cfg.base_model_chosen?.length || 0} Base + {cfg.meta_model_chosen || "Meta"})
                        </span>
                      ) : (
                        <span>Single: {cfg.model_chosen || "Model"}</span>
                      )}
                    </td>
                    <td className="px-5 py-3 text-zinc-600 dark:text-zinc-400">
                      {cfg.hypertuner_chosen} ({cfg.hypertune_trial} trials)
                    </td>
                    <td className="px-5 py-3 text-zinc-600 dark:text-zinc-400">
                      {cfg.has_embedded_dataset ? (
                        <span className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                          <CheckCircle2 className="h-3.5 w-3.5" />
                          {cfg.row_count} rows embedded
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-400">External reference</span>
                      )}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <Link
                        href={`/run?config=${encodeURIComponent(cfg.name)}`}
                        className="inline-flex items-center gap-1.5 rounded-lg bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-600 hover:bg-blue-100 dark:bg-blue-950/50 dark:text-blue-400 dark:hover:bg-blue-900/50 transition-colors"
                      >
                        <PlayCircle className="h-3.5 w-3.5" />
                        <span>Run</span>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
