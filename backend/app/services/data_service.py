import os
import uuid
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from backend.app.config import CONFIGURED_DATA_DIR, UPLOADS_DIR


def extract_columns_and_preview(file_path: Path | str, file_type: str) -> Dict[str, Any]:
    """Reads columns, row count, and first few rows for preview with fast chunked counting."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    if file_type.upper() == "CSV" or file_path.suffix.lower() == ".csv":
        df_preview = pd.read_csv(file_path, nrows=10)
        # Fast binary chunk line counting (10x faster than line-by-line for large CSVs)
        total_rows = 0
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                total_rows += chunk.count(b"\n")
        total_rows = max(0, total_rows - 1)
    elif file_type.upper() in ["EXCEL", "XLSX", "XLS"] or file_path.suffix.lower() in [".xlsx", ".xls"]:
        df_preview = pd.read_excel(file_path, engine="openpyxl", nrows=10)
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True, keep_links=False)
            ws = wb.active
            total_rows = max(0, (ws.max_row or 1) - 1)
            wb.close()
        except Exception:
            df_full = pd.read_excel(file_path, engine="openpyxl", usecols=[0])
            total_rows = len(df_full)
    else:
        return {
            "columns": [],
            "row_count": 0,
            "preview": []
        }

    # Clean nan for json
    df_preview_cleaned = df_preview.fillna("")
    preview_records = df_preview_cleaned.to_dict(orient="records")

    return {
        "columns": df_preview.columns.tolist(),
        "row_count": total_rows,
        "preview": preview_records
    }


def _extract_file_metadata(save_path: Path, sanitized_name: str, file_id: str, file_size: int) -> Dict[str, Any]:
    suffix = save_path.suffix.lower()
    if suffix == ".csv":
        file_type = "CSV"
    elif suffix in [".xlsx", ".xls"]:
        file_type = "Excel"
    elif suffix in [".png", ".jpg", ".jpeg"]:
        file_type = "Image"
    elif suffix == ".cif":
        file_type = "CIF"
    else:
        file_type = "Other"

    info = {
        "id": file_id,
        "name": sanitized_name,
        "path": str(save_path),
        "type": file_type,
        "size": file_size,
        "columns": [],
        "row_count": 0,
        "preview": []
    }

    if file_type in ["CSV", "Excel"]:
        try:
            meta = extract_columns_and_preview(save_path, file_type)
            info["columns"] = meta["columns"]
            info["row_count"] = meta["row_count"]
            info["preview"] = meta["preview"]
        except Exception as e:
            info["error"] = str(e)

    return info


def save_uploaded_stream(file_obj: Any, filename: str) -> Dict[str, Any]:
    """Streams uploaded file chunks directly to disk to minimize memory footprint."""
    file_id = str(uuid.uuid4())[:8]
    sanitized_name = Path(filename).name
    save_path = UPLOADS_DIR / f"{file_id}_{sanitized_name}"

    total_size = 0
    with open(save_path, "wb") as f_out:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)
            total_size += len(chunk)

    return _extract_file_metadata(save_path, sanitized_name, file_id, total_size)


def save_uploaded_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Saves uploaded file bytes to UPLOADS_DIR and extracts metadata."""
    file_id = str(uuid.uuid4())[:8]
    sanitized_name = Path(filename).name
    save_path = UPLOADS_DIR / f"{file_id}_{sanitized_name}"
    
    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return _extract_file_metadata(save_path, sanitized_name, file_id, len(file_bytes))


def list_configured_datasets() -> List[Dict[str, Any]]:
    """Lists all configured datasets stored in CONFIGURED_DATA_DIR."""
    datasets = []
    if not CONFIGURED_DATA_DIR.exists():
        return datasets

    for file in sorted(CONFIGURED_DATA_DIR.iterdir()):
        if file.is_file():
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                
                features = []
                targets = []
                types = []
                file_count = 0

                if isinstance(data, list):
                    for group in data:
                        types.append(group.get("type", "Unknown"))
                        file_count += len(group.get("item_list", []))
                        features.extend(group.get("feature_column", []))
                        targets.extend(group.get("target_column", []))
                
                datasets.append({
                    "name": file.name,
                    "path": str(file),
                    "created_at": file.stat().st_mtime,
                    "types": list(set(types)),
                    "file_count": file_count,
                    "feature_count": len(features),
                    "target_count": len(targets),
                    "features": features,
                    "targets": targets
                })
            except Exception as e:
                datasets.append({
                    "name": file.name,
                    "path": str(file),
                    "error": str(e)
                })

    return datasets


def get_configured_dataset(name: str) -> Optional[List[Dict[str, Any]]]:
    """Loads a specific dataset configuration."""
    target_path = CONFIGURED_DATA_DIR / name
    if not target_path.exists():
        return None
    
    with open(target_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_dataset_configuration(name: str, item_groups: List[Dict[str, Any]], overwrite: bool = False) -> Path:
    """
    Saves item_group_list into CONFIGURED_DATA_DIR / name as a pickle,
    compatible with the original QuickMachine structure.
    """
    target_path = CONFIGURED_DATA_DIR / name
    if target_path.exists() and not overwrite:
        raise ValueError(f"Configuration '{name}' already exists. Please use another name or overwrite.")

    # Format the groups to be compatible with original structure:
    # item_list elements: either a string path or an object with 'path' and 'name'
    serialized_groups = []
    for g in item_groups:
        group_copy = {
            "id": g.get("id", str(uuid.uuid4())),
            "type": g.get("type"),
            "role": g.get("role"),
            "columns": g.get("columns", []),
            "feature_column": g.get("feature_column", []),
            "target_column": g.get("target_column", []),
            "ignore_column": g.get("ignore_column", []),
            "item_list": g.get("item_list", [])
        }
        serialized_groups.append(group_copy)

    with open(target_path, "wb") as f:
        pickle.dump(serialized_groups, f)

    return target_path


def delete_configured_dataset(name: str) -> bool:
    """Deletes a configured dataset by name."""
    target_path = CONFIGURED_DATA_DIR / name
    if target_path.exists():
        target_path.unlink()
        return True
    return False
