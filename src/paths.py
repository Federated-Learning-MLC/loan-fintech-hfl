import os
from pathlib import Path

PARENT = Path(__file__).parent.resolve().parent

DATA_DIR = PARENT / "data"
CLIENTS_DATA_DIR = DATA_DIR / "clients_data"


if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(CLIENTS_DATA_DIR).exists():
    os.mkdir(CLIENTS_DATA_DIR)
