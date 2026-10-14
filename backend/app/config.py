import pathlib
import sys
from pathlib import Path

# Fix: On Linux, unpickling pickles saved on Windows with WindowsPath raises NotImplementedError.
# Patch pathlib so WindowsPath resolves to PosixPath on non-Windows platforms.
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAVED_CONFIGS_DIR = BASE_DIR / "saved_configs"
CONFIGURED_DATA_DIR = BASE_DIR / "configured_data"
UPLOADS_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "training_artifacts"

SAVED_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
CONFIGURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
