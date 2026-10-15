from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool
from typing import List, Optional
import json

from backend.app.services.data_service import (
    save_uploaded_file,
    save_uploaded_stream,
    list_configured_datasets,
    get_configured_dataset,
    save_dataset_configuration,
    delete_configured_dataset,
    extract_columns_and_preview
)

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/configs")
def list_configs():
    """Lists all configured datasets."""
    return {"datasets": list_configured_datasets()}


@router.get("/configs/{name}")
def get_config(name: str):
    """Gets details of a configured dataset."""
    data = get_configured_dataset(name)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Configuration '{name}' not found")
    return {"name": name, "groups": data}


@router.delete("/configs/{name}")
def delete_config(name: str):
    """Deletes a configured dataset."""
    success = delete_configured_dataset(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Configuration '{name}' not found")
    return {"success": True, "message": f"Configuration '{name}' deleted"}


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Uploads one or multiple files and returns parsed metadata without blocking event loop."""
    uploaded_items = []
    for f in files:
        # Offload streaming write and file parsing to threadpool to keep event loop responsive
        item_info = await run_in_threadpool(save_uploaded_stream, f.file, f.filename)
        uploaded_items.append(item_info)
    return {"files": uploaded_items}


@router.post("/save")
def save_config_endpoint(payload: dict):
    """
    Saves a dataset configuration group list.
    Payload:
    {
        "name": str,
        "item_groups": list,
        "overwrite": bool (optional)
    }
    """
    name = payload.get("name")
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Configuration name is required")

    item_groups = payload.get("item_groups", [])
    if not item_groups:
        raise HTTPException(status_code=400, detail="At least one file group is required")

    # Validate that at least one feature and one target exist
    has_feature = any(
        (g.get("type") in ["Image", "CIF"] and g.get("role") == "Feature")
        or (g.get("type") in ["CSV", "Excel"] and bool(g.get("feature_column")))
        for g in item_groups
    )
    total_targets = sum(
        1 if g.get("type") in ["Image", "CIF"] and g.get("role") == "Target"
        else len(g.get("target_column", []))
        for g in item_groups
    )
    if not has_feature:
        raise HTTPException(status_code=400, detail="Must assign at least one Feature column/role")
    if total_targets != 1:
        raise HTTPException(status_code=400, detail="Exactly one Target column must be selected (multi-output target is not supported)")

    overwrite = payload.get("overwrite", False)
    try:
        saved_path = save_dataset_configuration(name.strip(), item_groups, overwrite=overwrite)
        return {
            "success": True,
            "name": name.strip(),
            "path": str(saved_path),
            "message": f"Dataset configuration '{name}' saved successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
