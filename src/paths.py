import os
from pathlib import Path

PARENT = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT / "data"
CLIENTS_DATA_DIR = DATA_DIR / "clients_data"
MODEL_DIR = PARENT / "model"


if not Path(DATA_DIR).exists():
    os.makedirs(DATA_DIR, exist_ok=True)

if not Path(CLIENTS_DATA_DIR).exists():
    os.makedirs(CLIENTS_DATA_DIR, exist_ok=True)

if not Path(MODEL_DIR).exists():
    os.makedirs(MODEL_DIR, exist_ok=True)
