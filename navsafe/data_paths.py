"""Locations in a local snapshot of the public NavSafe dataset."""
import os
from pathlib import Path

def data_root():
    return Path(os.environ.get("NAVSAFE_DATA_ROOT", "~/data/NavSafe")).expanduser()

def model_path(relative):
    root = Path(os.environ.get("NAVSAFE_MODEL_ZOO", str(data_root() / "model_zoo"))).expanduser()
    return str(root / relative)
