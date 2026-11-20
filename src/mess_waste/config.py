"""Loading settings from params.yaml and small file helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml

# Repository root: src/mess_waste/config.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    """Read params.yaml with yaml.safe_load."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from params.yaml (relative to the repo root) into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's raw bytes (logged as data_sha256 in every MLflow run)."""
    with open(resolve(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV exactly as given (no cleaning)."""
    return pd.read_csv(resolve(path))
