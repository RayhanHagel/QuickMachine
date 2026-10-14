from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from backend.app.services.config_service import (
    inspect_dataset_configuration,
    save_pipeline_configuration,
    list_saved_configurations,
    get_saved_configuration,
    delete_saved_configuration
)
from backend.app.utilities.model_hyperparams import get_json

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("/data-preview/{data_name}")
def data_preview(data_name: str):
    """Inspects a dataset configuration: computes missing values, column types, and previews."""
    try:
        inspection = inspect_dataset_configuration(data_name)
        return inspection
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Configured dataset '{data_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-options")
def model_options():
    """Returns available models, libraries, and their hyperparameter options."""
    return get_json()


@router.get("/saved")
def list_saved():
    """Lists all saved training configurations."""
    return {"saved_configs": list_saved_configurations()}


@router.get("/saved/{name}")
def get_saved(name: str):
    """Retrieves metadata and settings for a saved configuration."""
    cfg = get_saved_configuration(name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Saved config '{name}' not found")
    return {"name": name, "config": cfg}


@router.delete("/saved/{name}")
def delete_saved(name: str):
    """Deletes a saved configuration."""
    success = delete_saved_configuration(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Saved config '{name}' not found")
    return {"success": True, "message": f"Saved config '{name}' deleted"}


@router.post("/save")
def save_config(payload: Dict[str, Any]):
    """
    Saves a full model training configuration with embedded dataset.
    Payload:
    {
        "config_name": str,
        "config": dict
    }
    """
    config_name = payload.get("config_name")
    if not config_name or not config_name.strip():
        raise HTTPException(status_code=400, detail="Config name is required")

    config_dict = payload.get("config", {})
    if not config_dict:
        raise HTTPException(status_code=400, detail="Configuration parameters are required")

    try:
        saved_path = save_pipeline_configuration(config_dict, config_name.strip())
        return {
            "success": True,
            "name": config_name.strip(),
            "path": str(saved_path),
            "message": f"Configuration '{config_name}' successfully saved"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
