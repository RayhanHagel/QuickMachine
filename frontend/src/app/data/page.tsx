"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Upload,
  FolderOpen,
  FileSpreadsheet,
  FileText,
  Image as ImageIcon,
  Layers,
  Trash2,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Database,
  Search,
} from "lucide-react";
import {
  uploadFiles,
  saveDatasetConfig,
  listDataConfigs,
  deleteDataConfig,
  FileMeta,
  ItemGroup,
  ConfiguredDataset
} from "@/lib/api";

const EXTENSION_MAP: Record<string, string[]> = {
  CSV: [".csv"],
  Excel: [".xlsx", ".xls"],
  CIF: [".cif"],
  Image: [".jpg", ".jpeg", ".png"],
};

export default function DataCreatePage() {
  const router = useRouter();

  // Extension and selection mode
  const [extensionType, setExtensionType] = useState<string>("CSV");
  const [searchOption, setSearchOption] = useState<"Files" | "Folder">("Files");

  // File upload state
  const [isUploading, setIsUploading] = useState(false);
  const [stagedFiles, setStagedFiles] = useState<FileMeta[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Item groups state (the groups ready to be assigned roles)
  const [itemGroups, setItemGroups] = useState<ItemGroup[]>([]);
  const [columnFilter, setColumnFilter] = useState<Record<string, string>>({});

  // Validation & Save state
  const [datasetName, setDatasetName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Existing datasets list
  const [existingDatasets, setExistingDatasets] = useState<ConfiguredDataset[]>([]);
  const [loadingExisting, setLoadingExisting] = useState(true);

  // Load existing datasets
  const loadExisting = async () => {
    setLoadingExisting(true);
    try {
      const data = await listDataConfigs();
      setExistingDatasets(data);
    } catch {
      // ignore
    } finally {
      setLoadingExisting(false);
    }
  };

  useEffect(() => {
    loadExisting();
  }, []);

  // Handle file input change
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const filesArray = Array.from(e.target.files);

    setIsUploading(true);
    setFeedback(null);
    try {
      const uploaded = await uploadFiles(filesArray);
      setStagedFiles(uploaded);
    } catch (err: any) {
      setFeedback({ type: "error", message: err.message || "Failed to upload files" });
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Confirm Staged Files into an Item Group
  const addStagedGroup = () => {
    if (stagedFiles.length === 0) return;

    // Use columns from first file if CSV/Excel
    const primaryFile = stagedFiles[0];
    const columns = primaryFile.columns || [];

    const newGroup: ItemGroup = {
      id: crypto.randomUUID(),
      type: extensionType,
      role: null,
      columns: columns,
      feature_column: [],
      target_column: [],
      ignore_column: [...columns],
      item_list: stagedFiles.map((f) => ({ path: f.path, name: f.name })),
    };

    setItemGroups([...itemGroups, newGroup]);
    setStagedFiles([]);
  };

  // Cancel Staged Files
  const cancelStaged = () => {
    setStagedFiles([]);
  };

  // Remove Item Group
  const removeGroup = (groupId: string) => {
    setItemGroups(itemGroups.filter((g) => g.id !== groupId));
  };

  // Column Role Change with mutual exclusivity
  const handleColumnRoleChange = (
    groupId: string,
    role: "feature_column" | "target_column",
    selectedCols: string[]
  ) => {
    setItemGroups(
      itemGroups.map((g) => {
        if (g.id !== groupId) return g;

        let updatedFeatures = [...g.feature_column];
        let updatedTargets = [...g.target_column];

        if (role === "feature_column") {
          updatedFeatures = selectedCols;
          // Remove selected features from targets
          updatedTargets = updatedTargets.filter((c) => !selectedCols.includes(c));
        } else {
          // Enforce single target selection (latest selected column only)
          updatedTargets = selectedCols.length > 0 ? [selectedCols[selectedCols.length - 1]] : [];
          // Remove selected target from features
          updatedFeatures = updatedFeatures.filter((c) => !updatedTargets.includes(c));
        }

        const chosenSet = new Set([...updatedFeatures, ...updatedTargets]);
        const updatedIgnore = g.columns.filter((c) => !chosenSet.has(c));

        return {
          ...g,
          feature_column: updatedFeatures,
          target_column: updatedTargets,
          ignore_column: updatedIgnore,
        };
      })
    );
  };

  // Non-tabular (CIF / Image) Role Change
  const handleNonTabularRoleChange = (groupId: string, role: string) => {
    setItemGroups(
      itemGroups.map((g) => {
        if (g.id !== groupId) return g;
        return { ...g, role };
      })
    );
  };

  // Check validity
  const hasFeature = itemGroups.some(
    (g) =>
      (["Image", "CIF"].includes(g.type) && g.role === "Feature") ||
      (["CSV", "Excel"].includes(g.type) && g.feature_column.length > 0)
  );
  const hasTarget = itemGroups.some(
    (g) =>
      (["Image", "CIF"].includes(g.type) && g.role === "Target") ||
      (["CSV", "Excel"].includes(g.type) && g.target_column.length > 0)
  );
  const isValid = hasFeature && hasTarget && datasetName.trim().length > 0;

  // Save Configuration
  const handleSave = async () => {
    if (!isValid) return;
    setIsSaving(true);
    setFeedback(null);
    try {
      await saveDatasetConfig(datasetName.trim(), itemGroups, true);
      setFeedback({
        type: "success",
        message: `Dataset '${datasetName.trim()}' configured and saved successfully!`,
      });
      loadExisting();
    } catch (err: any) {
      setFeedback({ type: "error", message: err.message || "Failed to save configuration" });
    } finally {
      setIsSaving(false);
    }
  };

  // Delete existing dataset
  const handleDeleteExisting = async (name: string) => {
    if (!confirm(`Are you sure you want to delete dataset '${name}'?`)) return;
    try {
      await deleteDataConfig(name);
      loadExisting();
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="space-y-10">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-zinc-900 dark:text-zinc-50 flex items-center gap-3">
          <Database className="h-8 w-8 text-blue-600" />
          <span>Data Preparation & Column Assignment</span>
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          Upload tabular datasets (CSV, Excel) or CIF/Image files, inspect columns, and designate input features versus target labels.
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
              onClick={() => router.push(`/config?dataset=${encodeURIComponent(datasetName.trim())}`)}
              className="inline-flex items-center gap-1.5 font-bold text-emerald-700 dark:text-emerald-300 hover:underline"
            >
              <span>Continue to Model Config</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          )}
        </div>
      )}

      {/* Section 1: File Selection & Upload */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <FolderOpen className="h-5 w-5 text-blue-500" />
          <span>Step 1: Select & Upload Files</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* File Extension Type */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              File Extension Type
            </label>
            <div className="grid grid-cols-4 gap-2">
              {Object.keys(EXTENSION_MAP).map((ext) => (
                <button
                  key={ext}
                  type="button"
                  onClick={() => {
                    setExtensionType(ext);
                    if (["CSV", "Excel"].includes(ext)) {
                      setSearchOption("Files");
                    }
                  }}
                  className={`flex flex-col items-center justify-center rounded-xl p-3 border text-xs font-semibold transition-all ${
                    extensionType === ext
                      ? "border-blue-600 bg-blue-50 text-blue-700 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-400 shadow-sm"
                      : "border-zinc-200 bg-zinc-50 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-800/50 dark:text-zinc-300"
                  }`}
                >
                  {ext === "CSV" ? (
                    <FileText className="h-5 w-5 mb-1 text-emerald-500" />
                  ) : ext === "Excel" ? (
                    <FileSpreadsheet className="h-5 w-5 mb-1 text-green-600" />
                  ) : ext === "Image" ? (
                    <ImageIcon className="h-5 w-5 mb-1 text-purple-500" />
                  ) : (
                    <Layers className="h-5 w-5 mb-1 text-amber-500" />
                  )}
                  <span>{ext}</span>
                </button>
              ))}
            </div>
            <p className="text-xs text-zinc-400">
              Supported: {EXTENSION_MAP[extensionType].join(", ")}
            </p>
          </div>

          {/* Search Mode */}
          <div className="space-y-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Search Option
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(["Files", "Folder"] as const).map((opt) => {
                const disabled = ["CSV", "Excel"].includes(extensionType) && opt === "Folder";
                return (
                  <button
                    key={opt}
                    type="button"
                    disabled={disabled}
                    onClick={() => setSearchOption(opt)}
                    className={`flex items-center justify-center rounded-xl p-3 border text-xs font-semibold transition-all ${
                      searchOption === opt && !disabled
                        ? "border-blue-600 bg-blue-50 text-blue-700 dark:border-blue-500 dark:bg-blue-950/50 dark:text-blue-400 shadow-sm"
                        : disabled
                        ? "opacity-40 border-zinc-200 bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-800/20 cursor-not-allowed text-zinc-400"
                        : "border-zinc-200 bg-zinc-50 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-800/50 dark:text-zinc-300"
                    }`}
                  >
                    <span>{opt}</span>
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-zinc-400">
              {["CSV", "Excel"].includes(extensionType)
                ? "Individual file upload for tabular parsing"
                : "Folder or multi-file selection"}
            </p>
          </div>
        </div>

        {/* Drag & Drop Upload Zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className="group relative cursor-pointer rounded-2xl border-2 border-dashed border-zinc-300 dark:border-zinc-700 hover:border-blue-500 dark:hover:border-blue-400 bg-zinc-50 dark:bg-zinc-800/40 p-8 text-center transition-all hover:bg-blue-50/20 dark:hover:bg-blue-950/10"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple={["Image", "CIF"].includes(extensionType)}
            accept={EXTENSION_MAP[extensionType].join(",")}
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="flex flex-col items-center justify-center space-y-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400 group-hover:scale-110 transition-transform">
              <Upload className="h-6 w-6" />
            </div>
            <div className="space-y-1">
              <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                Click to browse or drop {extensionType} files here
              </p>
              <p className="text-xs text-zinc-400">
                Files are processed and column schemas are extracted automatically
              </p>
            </div>
            {isUploading && (
              <p className="text-xs font-semibold text-blue-600 animate-pulse">
                Uploading and parsing columns...
              </p>
            )}
          </div>
        </div>

        {/* Staged Upload Confirmation */}
        {stagedFiles.length > 0 && (
          <div className="rounded-xl border border-blue-200 dark:border-blue-900 bg-blue-50/60 dark:bg-blue-950/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="h-5 w-5 text-blue-600" />
                <span className="font-semibold text-sm text-zinc-900 dark:text-zinc-100">
                  {stagedFiles.length} file{stagedFiles.length > 1 ? "s" : ""} selected for group
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={cancelStaged}
                  className="rounded-lg border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={addStagedGroup}
                  className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-blue-500"
                >
                  Add Files to Group
                </button>
              </div>
            </div>

            <div className="max-h-40 overflow-y-auto space-y-1 text-xs">
              {stagedFiles.map((f, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg bg-white/70 dark:bg-zinc-900/70 px-3 py-2 border border-zinc-200 dark:border-zinc-800"
                >
                  <span className="font-medium text-zinc-800 dark:text-zinc-200 truncate max-w-sm">
                    📄 {f.name}
                  </span>
                  <span className="text-zinc-500">
                    {f.columns.length > 0 ? `${f.columns.length} columns · ${f.row_count} rows` : `${Math.round(f.size / 1024)} KB`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Section 2: Assign Features & Targets per Item Group */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
            <Layers className="h-5 w-5 text-purple-500" />
            <span>Step 2: Assign Roles (Features vs Targets)</span>
          </h2>
          <span className="text-xs font-medium text-zinc-500">
            {itemGroups.length} configured item group{itemGroups.length !== 1 ? "s" : ""}
          </span>
        </div>

        {itemGroups.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-300 dark:border-zinc-800 p-8 text-center space-y-2">
            <FileSpreadsheet className="h-8 w-8 text-zinc-400 mx-auto" />
            <p className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
              No file groups created yet.
            </p>
            <p className="text-xs text-zinc-500">
              Upload files in Step 1 and click &quot;Add Files to Group&quot; to configure them.
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {itemGroups.map((group, idx) => (
              <div
                key={group.id}
                className="rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-950/40 p-6 space-y-6 shadow-sm"
              >
                {/* Group Header */}
                <div className="flex items-center justify-between border-b border-zinc-200 dark:border-zinc-800 pb-4">
                  <div className="flex items-center gap-3">
                    <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/50 text-xs font-bold text-purple-700 dark:text-purple-300">
                      #{idx + 1}
                    </span>
                    <div>
                      <h3 className="text-sm font-bold text-zinc-900 dark:text-zinc-100">
                        Item Group {idx + 1} · {group.type}
                      </h3>
                      <p className="text-xs text-zinc-500">
                        {group.item_list.map((f) => f.name).join(", ")}
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeGroup(group.id)}
                    className="flex items-center gap-1.5 rounded-lg border border-rose-200 dark:border-rose-900/60 bg-white dark:bg-zinc-900 px-3 py-1.5 text-xs font-semibold text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Delete Group</span>
                  </button>
                </div>

                {/* Tabular Column Assignment (CSV / Excel) */}
                {["CSV", "Excel"].includes(group.type) ? (
                  <div className="space-y-5">
                    {/* Column Search Filter */}
                    <div className="relative">
                      <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-zinc-400" />
                      <input
                        type="text"
                        placeholder="Filter columns by keyword..."
                        value={columnFilter[group.id] || ""}
                        onChange={(e) =>
                          setColumnFilter({ ...columnFilter, [group.id]: e.target.value })
                        }
                        className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 pl-9 pr-3 py-1.5 text-xs text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                    </div>

                    {/* Feature Columns */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <label className="text-xs font-bold uppercase tracking-wider text-blue-600 dark:text-blue-400">
                          Feature Columns ({group.feature_column.length})
                        </label>
                        <div className="flex gap-2 text-xs">
                          <button
                            type="button"
                            onClick={() =>
                              handleColumnRoleChange(
                                group.id,
                                "feature_column",
                                group.columns.filter((c) => !group.target_column.includes(c))
                              )
                            }
                            className="text-blue-600 dark:text-blue-400 hover:underline"
                          >
                            Select All (except target)
                          </button>
                          <span>·</span>
                          <button
                            type="button"
                            onClick={() => handleColumnRoleChange(group.id, "feature_column", [])}
                            className="text-zinc-500 hover:underline"
                          >
                            Clear
                          </button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
                        {group.columns
                          .filter((col) =>
                            col.toLowerCase().includes((columnFilter[group.id] || "").toLowerCase())
                          )
                          .map((col) => {
                            const isSelected = group.feature_column.includes(col);
                            const isTarget = group.target_column.includes(col);
                            return (
                              <button
                                key={col}
                                type="button"
                                onClick={() => {
                                  const next = isSelected
                                    ? group.feature_column.filter((c) => c !== col)
                                    : [...group.feature_column, col];
                                  handleColumnRoleChange(group.id, "feature_column", next);
                                }}
                                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                                  isSelected
                                    ? "bg-blue-600 text-white shadow-sm"
                                    : isTarget
                                    ? "bg-amber-100 text-amber-800 dark:bg-amber-950/50 dark:text-amber-300 opacity-60"
                                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                                }`}
                              >
                                {col} {isTarget ? "(Target)" : ""}
                              </button>
                            );
                          })}
                      </div>
                    </div>

                    {/* Target Column */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <label className="text-xs font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
                          Target Column ({group.target_column.length > 0 ? group.target_column[0] : "None"})
                        </label>
                        <button
                          type="button"
                          onClick={() => handleColumnRoleChange(group.id, "target_column", [])}
                          className="text-xs text-zinc-500 hover:underline"
                        >
                          Clear
                        </button>
                      </div>
                      <p className="text-[11px] text-zinc-400">
                        Click any column to designate as the prediction target (single-target modeling).
                      </p>
                      <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900">
                        {group.columns
                          .filter((col) =>
                            col.toLowerCase().includes((columnFilter[group.id] || "").toLowerCase())
                          )
                          .map((col) => {
                            const isSelected = group.target_column.includes(col);
                            const isFeature = group.feature_column.includes(col);
                            return (
                              <button
                                key={col}
                                type="button"
                                onClick={() => {
                                  const next = isSelected ? [] : [col];
                                  handleColumnRoleChange(group.id, "target_column", next);
                                }}
                                className={`rounded-lg px-2.5 py-1 text-xs font-medium transition-all ${
                                  isSelected
                                    ? "bg-amber-600 text-white shadow-sm font-bold"
                                    : isFeature
                                    ? "bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-300 opacity-60"
                                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300"
                                }`}
                              >
                                {col} {isFeature ? "(Feature)" : ""}
                              </button>
                            );
                          })}
                      </div>
                    </div>

                    {/* Ignored Columns Preview */}
                    <div className="space-y-1">
                      <label className="text-xs font-semibold text-zinc-500">
                        Ignored Columns ({group.ignore_column.length}):
                      </label>
                      <p className="text-xs text-zinc-400 truncate">
                        {group.ignore_column.length > 0
                          ? group.ignore_column.join(", ")
                          : "None (all columns are assigned)"}
                      </p>
                    </div>
                  </div>
                ) : (
                  /* CIF / Image Role Assignment */
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-xs font-bold uppercase tracking-wider text-zinc-500">
                        File Group Assignment
                      </label>
                      <div className="flex gap-2">
                        {["Not Used", "Feature", "Target"].map((r) => (
                          <button
                            key={r}
                            type="button"
                            onClick={() => handleNonTabularRoleChange(group.id, r)}
                            className={`rounded-xl px-4 py-2 text-xs font-bold border transition-all ${
                              group.role === r
                                ? "border-purple-600 bg-purple-50 text-purple-700 dark:border-purple-500 dark:bg-purple-950/50 dark:text-purple-300"
                                : "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-100 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                            }`}
                          >
                            {r}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Section 3: Validation & Save Dataset */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-6 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <CheckCircle className="h-5 w-5 text-emerald-500" />
          <span>Step 3: Validate & Save Configuration</span>
        </h2>

        {/* Validation Status Badges */}
        <div className="flex flex-wrap gap-4 text-xs font-semibold">
          <div
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 border ${
              hasFeature
                ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800"
                : "bg-zinc-100 text-zinc-500 border-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:border-zinc-700"
            }`}
          >
            {hasFeature ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
            <span>At least 1 Feature assigned</span>
          </div>

          <div
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 border ${
              hasTarget
                ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800"
                : "bg-zinc-100 text-zinc-500 border-zinc-200 dark:bg-zinc-800 dark:text-zinc-400 dark:border-zinc-700"
            }`}
          >
            {hasTarget ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
            <span>At least 1 Target assigned</span>
          </div>
        </div>

        {/* Name input and Save button */}
        <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
          <input
            type="text"
            placeholder="Enter dataset configuration name (e.g. Battery_Lithium_Data)"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
            className="flex-1 rounded-xl border border-zinc-300 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-4 py-2.5 text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="button"
            disabled={!isValid || isSaving}
            onClick={handleSave}
            className={`flex items-center justify-center gap-2 rounded-xl px-6 py-2.5 text-sm font-bold shadow-md transition-all ${
              isValid && !isSaving
                ? "bg-blue-600 text-white hover:bg-blue-500 active:scale-95"
                : "bg-zinc-200 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500 cursor-not-allowed"
            }`}
          >
            <CheckCircle className="h-4 w-4" />
            <span>{isSaving ? "Saving..." : "Validate & Save Dataset"}</span>
          </button>
        </div>
      </div>

      {/* Section 4: Existing Configured Datasets List */}
      <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-6 sm:p-8 space-y-4 shadow-sm">
        <h2 className="text-lg font-bold text-zinc-900 dark:text-zinc-100">
          Saved Dataset Configurations in Workspace
        </h2>

        {loadingExisting ? (
          <p className="text-sm text-zinc-500">Loading datasets...</p>
        ) : existingDatasets.length === 0 ? (
          <p className="text-sm text-zinc-500">No configured datasets saved yet.</p>
        ) : (
          <div className="overflow-hidden rounded-xl border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-left text-sm">
              <thead className="bg-zinc-50 text-xs font-semibold text-zinc-600 dark:bg-zinc-800/50 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800">
                <tr>
                  <th className="px-4 py-3">Dataset Name</th>
                  <th className="px-4 py-3">Types</th>
                  <th className="px-4 py-3">Features</th>
                  <th className="px-4 py-3">Targets</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                {existingDatasets.map((ds) => (
                  <tr key={ds.name} className="hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30">
                    <td className="px-4 py-3 font-semibold text-zinc-900 dark:text-zinc-100">
                      {ds.name}
                    </td>
                    <td className="px-4 py-3 text-xs text-zinc-500">
                      {ds.types?.join(", ") || "Tabular"}
                    </td>
                    <td className="px-4 py-3 text-xs text-blue-600 dark:text-blue-400 font-medium">
                      {ds.feature_count} features
                    </td>
                    <td className="px-4 py-3 text-xs text-amber-600 dark:text-amber-400 font-medium">
                      {ds.target_count} target(s)
                    </td>
                    <td className="px-4 py-3 text-right flex items-center justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => router.push(`/config?dataset=${encodeURIComponent(ds.name)}`)}
                        className="rounded-lg bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-600 hover:bg-blue-100 dark:bg-blue-950/50 dark:text-blue-400"
                      >
                        Configure Model →
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeleteExisting(ds.name)}
                        className="rounded-lg p-1 text-zinc-400 hover:text-rose-600 transition-colors"
                        title="Delete Dataset"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
