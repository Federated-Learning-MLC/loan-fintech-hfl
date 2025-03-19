from typing import Dict 
import pandas as pd
from src import paths


# Features & Label Composition
TARGET_COLS = "fraud_bool"
CAT_COLS = ['payment_type', 'employment_status','housing_status', 'source', 'device_os']
NUM_COLS = ['income', 'name_email_similarity','prev_address_months_count', 
            'current_address_months_count','customer_age', 'days_since_request', 
            'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 
            'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
            'credit_risk_score', 'email_is_free', 'phone_home_valid','phone_mobile_valid',
            'bank_months_count', 'has_other_cards','proposed_credit_limit', 'foreign_request', 
            'session_length_in_minutes', 'keep_alive_session', 'device_distinct_emails_8w','month']

# Constants and Configuration
IF_TRAIN_VAL = 0  # Include validation dataset in training
QUANTISATION = 0
SMPC_NOISE = 1
EPOCHS = 12 #10
BATCH_SIZE = 64
NUM_FEATURES = 51
LEARNING_RATE = 0.01
NUM_CLASSES = 2
DROPOUT = 0.1
NUM_HEADS = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 64
INPUT_DIM = 51 #<--- Num of features in train data


# Server Configuration
SERVER_CONFIG = {
    "num_clients": 5,
    "num_rounds": 20
}

# Dataset Configuration
dataset_config = {
    "path": paths.DATA_DIR,
    "data": paths.DATA_DIR / "Base.csv",
    "sampled_data": paths.DATA_DIR / "base_downsampled.csv",
    "seed": 42,
    "num_clients": SERVER_CONFIG["num_clients"],
    "num_features": NUM_FEATURES
}


BASE_DATA = dataset_config["data"]
BASE_DATA_DOWNSAMPLED = dataset_config["sampled_data"]
SEED = dataset_config["seed"]


# Constructing a unique run name based on our configuration
run_name = f"Running_{SERVER_CONFIG['num_clients']}clients_{SERVER_CONFIG['num_rounds']}rounds_{EPOCHS}epochs_{QUANTISATION}quant_{SMPC_NOISE}SMPCn"


if __name__ == "__main__":
    print(f"Run name: {run_name}")