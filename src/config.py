from typing import Dict 
import pandas as pd
import paths

# Constants and Configuration
IF_TRAIN_VAL = 0
QUANTISATION = 0
EPOCHS = 10
BATCH_SIZE = 500
NUM_FEATURES = 10 #verify number of features in dataset
LEARNING_RATE = 0.001
NUM_UNITS_1 = 15 #unclear # Extracting top results for NUM_UNITS_1 and NUM_UNITS_2
NUM_UNITS_2 = 5 #unclear

# Server Configuration
server_config = {
    "num_clients": 5,
    "num_rounds": 350 #why 350
}

# Dataset Configuration
dataset_config = {
    "path": paths.DATA_DIR / "Base.csv",
    "seed": 42,
    "num_clients": server_config["num_clients"],
    "num_features": NUM_FEATURES
}

# Server Configuration
server_config = {
    "num_clients": 10,
    "num_rounds": 350
}


BASE_DATA_PATH = dataset_config["path"]
SEED = dataset_config["seed"]

# Constructing a unique run name based on our configuration
run_name = f"Running_{server_config['num_clients']}clients_{server_config['num_rounds']}rounds_{EPOCHS}epochs_{QUANTISATION}"

if __name__ == "__main__":
    print(f"Run name: {run_name}")
    print(f"Top tuning results for NUM_UNITS_1: {NUM_UNITS_1}, NUM_UNITS_2: {NUM_UNITS_2}")
