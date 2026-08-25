"""Project-root anchored paths.

The backend must work regardless of the folder it is started from. All
runtime paths (chroma, chunks, laws, output, models, .env) are resolved
against PROJECT_ROOT so launching uvicorn from `backend/` or the repo root
behaves identically.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_env_paths = {
    "CHROMA_DB_PATH",
    "PDF_DIRECTORY",
}


def resolve(path) -> Path:
    """Return an absolute path. Relative inputs are anchored to PROJECT_ROOT;
    absolute paths and env-var overrides are returned unchanged."""
    if path is None:
        return PROJECT_ROOT
    p = Path(str(path))
    if p.is_absolute():
        return p
    if str(path) in _env_paths and os.getenv(str(path)):
        return Path(os.getenv(str(path)))
    return PROJECT_ROOT / p
