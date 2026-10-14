from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import asyncio
import queue

from backend.app.services.training_service import (
    start_training_job,
    cancel_training_job,
    get_training_job_status,
    list_all_training_jobs,
    predict_live,
    subscribe_to_job,
    unsubscribe_from_job
)
from backend.app.services.config_service import _read_pickle_file
from backend.app.config import SAVED_CONFIGS_DIR, ARTIFACTS_DIR

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/jobs")
def get_all_jobs():
    """Lists all active and historical training jobs."""
    return {"jobs": list_all_training_jobs()}


@router.post("/start")
def start_training(payload: Dict[str, Any]):
    """
    Starts a training job in the background.
    Payload:
    {
        "config_name": str, (name of saved config, e.g. 'Najwa RF-XGB')
        "eval_mode": "Hold-Out Test Split" | "K-Fold Cross Validation (Whole Data)",
        "custom_config": dict (optional: override saved config or run unsaved)
    }
    """
    config_name = payload.get("config_name")
    eval_mode = payload.get("eval_mode", "Hold-Out Test Split")
    custom_config = payload.get("custom_config")

    if custom_config:
        cfg = custom_config
        name = config_name or "adhoc_run"
    elif config_name:
        target_path = SAVED_CONFIGS_DIR / f"{config_name}.pkl"
        if not target_path.exists():
            target_path = SAVED_CONFIGS_DIR / config_name
            if not target_path.exists():
                raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found")
        cfg = _read_pickle_file(target_path)
        name = config_name
    else:
        raise HTTPException(status_code=400, detail="Must provide either config_name or custom_config")

    job_id = start_training_job(cfg, name, eval_mode)
    return {
        "job_id": job_id,
        "status": "pending",
        "config_name": name,
        "eval_mode": eval_mode,
        "message": "Training job started"
    }


@router.post("/cancel/{job_id}")
def cancel_job(job_id: str):
    """Cancels a pending or currently running training job."""
    safe_job_id = Path(job_id).name
    success = cancel_training_job(safe_job_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Job '{safe_job_id}' is not currently running or not found")
    return {"success": True, "message": f"Job '{safe_job_id}' cancelled"}


@router.get("/status/{job_id}")
def get_status(job_id: str):
    """Fetches current job status and results."""
    safe_job_id = Path(job_id).name
    status = get_training_job_status(safe_job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{safe_job_id}' not found")
    return status


@router.post("/predict/{job_id}")
def predict_endpoint(job_id: str, payload: Dict[str, Any]):
    """
    Runs live prediction using the trained model for job_id.
    Payload:
    {
        "samples": [ { "feature_1": 12.3, ... } ] or { "feature_1": 12.3 }
    }
    """
    safe_job_id = Path(job_id).name
    samples = payload.get("samples")
    if samples is None:
        # Check if payload itself is sample data
        samples = payload.get("data", payload)

    if isinstance(samples, dict):
        samples_list = [samples]
    elif isinstance(samples, list):
        samples_list = samples
    else:
        raise HTTPException(status_code=400, detail="Samples must be a dictionary or list of dictionaries")

    if not samples_list:
        raise HTTPException(status_code=400, detail="No sample data provided")

    try:
        results = predict_live(safe_job_id, samples_list)
        return results
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(exc)}")


@router.get("/stream/{job_id}")
async def stream_job_events(job_id: str):
    """Server-Sent Events (SSE) stream for real-time progress and trial updates."""
    safe_job_id = Path(job_id).name
    status = get_training_job_status(safe_job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job '{safe_job_id}' not found")

    async def event_generator():
        q = subscribe_to_job(safe_job_id)
        try:
            # Emit current snapshot first
            current = get_training_job_status(safe_job_id)
            if current:
                yield f"data: {json.dumps(current)}\n\n"

            while True:
                # Poll queue asynchronously
                try:
                    event = q.get_nowait()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ["completed", "failed", "cancelled"]:
                        break
                except queue.Empty:
                    # Check if job completed while queue was empty
                    current = get_training_job_status(safe_job_id)
                    if current and current.get("status") in ["completed", "failed", "cancelled"]:
                        yield f"data: {json.dumps(current)}\n\n"
                        break
                    await asyncio.sleep(0.5)
        finally:
            unsubscribe_from_job(safe_job_id, q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/download/predictions/{job_id}")
def download_predictions(job_id: str):
    """Downloads predictions CSV file."""
    safe_job_id = Path(job_id).name
    csv_path = ARTIFACTS_DIR / safe_job_id / "predictions.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Predictions CSV not found")
    return FileResponse(
        path=str(csv_path),
        filename=f"{safe_job_id[:8]}_predictions.csv",
        media_type="text/csv"
    )


@router.get("/download/model/{job_id}")
def download_model(job_id: str):
    """Downloads trained pipeline pickle file."""
    safe_job_id = Path(job_id).name
    pkl_path = ARTIFACTS_DIR / safe_job_id / "trained_pipeline.pkl"
    if not pkl_path.exists():
        raise HTTPException(status_code=404, detail="Trained pipeline not found")
    return FileResponse(
        path=str(pkl_path),
        filename="trained_pipeline.pkl",
        media_type="application/octet-stream"
    )


@router.get("/download/optuna/{job_id}")
def download_optuna_logs(job_id: str):
    """Downloads optuna trials history CSV."""
    safe_job_id = Path(job_id).name
    csv_path = ARTIFACTS_DIR / safe_job_id / "optuna_trials.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Optuna logs CSV not found")
    return FileResponse(
        path=str(csv_path),
        filename=f"{safe_job_id[:8]}_optuna_trials.csv",
        media_type="text/csv"
    )


@router.get("/download/plot/{job_id}/{plot_name}")
def download_plot(job_id: str, plot_name: str):
    """Downloads generated matplotlib plot PNG."""
    safe_job_id = Path(job_id).name
    safe_plot = Path(plot_name).name
    plot_path = ARTIFACTS_DIR / safe_job_id / safe_plot
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot file not found")
    return FileResponse(
        path=str(plot_path),
        filename=safe_plot,
        media_type="image/png"
    )
