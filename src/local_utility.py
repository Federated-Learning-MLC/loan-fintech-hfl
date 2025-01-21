import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from typing import Optional, Union, List

from src.config import BASE_DATA_PATH, NUM_FEATURES, SEED
from src.paths import DATA_DIR, CLIENTS_DATA_DIR


# ---------------------- CONTROL FOR RANDOMNESS: ----------------------------------------------------------------

def set_seed(seed: int = SEED, seed_torch: bool = True):
    """
    Seeds the random number generators of PyTorch, NumPy, and Python's `random` module to ensure
    reproducibility of results across runs when using PyTorch for deep learning experiments.

    This function sets the seed for PyTorch (both CPU and CUDA), NumPy, and the Python `random` module,
    enabling CuDNN benchmarking and deterministic algorithms. It is crucial for experiments requiring
    reproducibility, like model performance comparisons. Note that enabling CuDNN benchmarking and
    deterministic operations may impact performance and limit certain optimizations.

    Args:
        seed (int, optional):
            A non-negative integer that defines the random state. Defaults to 'SEED' value in config file.

        seed_torch (bool, optional): 
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Defaults to True.
    Returns:
        None
            This function does not return a value but sets the random seed for various libraries.

    Notes:
        - When using multiple GPUs, `th.cuda.manual_seed_all(seed)` ensures all GPUs are seeded, 
        crucial for reproducibility in multi-GPU setups.

    Example:
        >>> SEED = 42
        >>> set_seed(SEED)
    """
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #print(f'Random seed {seed} has been set.')


# ---------------------- DATASET CHECKS: ----------------------------------------------------------------------------

def row_check(clients: int = 5):
    """
    Validates the integrity of the dataset across multiple clients by checking the total number of rows, 
    and the sum of +ve fraud labels across training, validation, and test datasets.

    This function ensures that the combined dataset from multiple clients, along with the test dataset,
    matches expected values for total row count, and total positive fraud count. These checks are critical for
    verifying data integrity and consistency before proceeding with further data analysis or model training.

    Parameters:
        clients : int, optional
            The number of clients (or partitions) for which training and validation datasets are available.
            Defaults to 5.

    Raises:
        AssertionError
            If the total number of rows or total +ve fraud count do not match expected values, 
            an AssertionError is raised indicating which specific integrity check has failed.

    Notes:
        - Assumes existence of CSV files in '../data/' following specific naming conventions.
        - Useful for data preprocessing in machine learning workflows involving multiple sources or clients.
        - 'fraud_bool' is assumed to be column name in the respective CSV files for class label.

    Example:
        >>> row_check(clients=3)
        # Checks datasets for 3 agents, along with the test dataset, and prints the status of each check.
    """
    # Helper function to load a CSV file into a DataFrame
    def load_data_frame(prefix: str, index: int) -> pd.DataFrame:
        return pd.read_csv(f"{CLIENTS_DATA_DIR}/{prefix}_{index}.csv")

    # Initialize expected values
    expected_row_count = 1_000_000
    expected_label_sum = 11_029

    # Load and aggregate datasets
    datasets = {"X_train": [], "X_val": [], "y_train": [], "y_val": []}
    for prefix in datasets.keys():
        for i in range(clients):
            datasets[prefix].append(load_data_frame(prefix, i))
    X_test = pd.read_csv(CLIENTS_DATA_DIR/"X_test.csv")
    y_test = pd.read_csv(CLIENTS_DATA_DIR/"y_test.csv")

    # Calculate totals
    total_row_count = sum(len(df) for prefix in ["X_train", "X_val"] for df in datasets[prefix]) + len(X_test)
    total_label_sum = sum(df["y"].sum() for prefix in ["y_train", "y_val"] for df in datasets[prefix]) + y_test["y"].sum()

    # Validate dataset integrity
    assert total_row_count == expected_row_count, f"Total row count mismatch: expected {expected_row_count}, got {total_row_count}"
    assert total_label_sum == expected_label_sum, f"Total +ve class mismatch: expected {expected_label_sum}, got {total_label_sum}"

    print('All checks passed successfully.')


# ----------------------------------- EDA PLOTING: -----------------------------------------------------------


class EDAPlotter:
    def __init__(self) -> None:
        pass

    def plot_skewness(self, df):
        # Filter numerical features in the DataFrame
        numerical_features = df.select_dtypes(include=["number"])

        # Calculate skewness of each numerical feature
        skew_values = numerical_features.skew()

        # Create a plot of skewness values
        plt.figure(figsize=(20, 8))
        skew_values.plot(kind="bar")
        plt.title("Skewness of Numerical Features")
        plt.xlabel("Features")
        plt.ylabel("Skewness Value")
        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.show()

        return skew_values

    def plot_numerical_features(self, dataframe):
        df_numerical = dataframe.select_dtypes(include=["int64", "float64"])
        num_cols = len(df_numerical.columns)
        # Determine the number of rows needed
        num_rows = (num_cols // 4) + (num_cols % 4 > 0)

        # Adjust the figsize to fit 4 columns
        fig, axes = plt.subplots(num_rows, 4, figsize=(20, 4 * num_rows))

        for i, feature in enumerate(df_numerical.columns):
            row = i // 4
            col = i % 4

            ax = axes[row, col]
            ax.hist(dataframe[feature].dropna(), bins=30, edgecolor="black")
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")

        # Hide empty subplots if the number of features is not a multiple of 4
        for j in range(i + 1, num_rows * 4):
            fig.delaxes(axes.flatten()[j])

        fig.tight_layout()
        plt.show()

    def plot_categorical_features(self, df):
        categorical_features = df.select_dtypes(include=["object", "category"])

        cat_cols = len(categorical_features.columns)
        cat_rows = (cat_cols // 2) + (cat_cols % 2)

        fig, axes = plt.subplots(cat_rows, 2, figsize=(15, 4 * cat_rows))

        for i, feature in enumerate(categorical_features.columns):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            value_counts = df[feature].value_counts()
            ax.bar(
                value_counts.index,
                value_counts.values,
                color="skyblue",
                edgecolor="black",
            )
            ax.set_title(f"Countplot of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            ax.tick_params(
                axis="x", rotation=45
            )  # Rotate x-axis labels for better readability if necessary

        # Hide empty subplots if the number of features is odd
        if cat_cols % 2 != 0:
            axes[-1, -1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_missing_values_proportion(
        self, df: pd.DataFrame, cols_missing_neg1: list[str]
    ):
        # Replace -1 with NaN in the specified columns
        df[cols_missing_neg1] = df[cols_missing_neg1].replace(-1, np.nan)

        # Calculate the percentage of missing values by feature
        null_X = df.isna().sum() / len(df) * 100

        # Plot the missing values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = (null_X.loc[null_X > 0].sort_values().plot(
            kind="bar", title="Percentage of Missing Values", ax=ax))

        # Annotate the bars with the percentage of missing values
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
                color="red",
            )

        ax.set_ylabel("Missing %")
        ax.set_xlabel("Feature")

        # Remove gridlines from the x-axis
        ax.xaxis.grid(False)

        plt.show()


# ----------------------- Data Preprocessing Utils: --------------------------------

def preprocess_dataframe(df):
    """
    Applies preprocessing steps to the dataframe, including shuffling, data type transformations,
    and value capping based on specified criteria.

    Parameters:
        df : DataFrame 
            The pandas DataFrame to preprocess.

    Returns:
        DataFrame
            The preprocessed DataFrame.

    Usage:
    ```
    df_preprocessed = preprocess_dataframe(df)
    ```
    """

    # Shuffle dataset
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Drop unused variable
    df = df.drop(['device_fraud_count'], axis=1)

    # Replacing the negative values in velocity_6h with 0
    df['velocity_6h'] = np.where(df['velocity_6h'] < 0, 0, df['velocity_6h'])

    # Drop all rows with missing data (EXPERIMENT ON THIS. DOES FILLING WORKS BETTER?)
    df = df.dropna(axis=0)

    return df


def encode_and_scale_dataframe(df):
    """
    Encodes categorical variables and scales numerical features within the DataFrame.

    Parameters:
        df : DataFrame 
            The DataFrame to encode and scale.

    Returns:
        DataFrame 
            The encoded and scaled DataFrame.
        MinMaxScaler 
            The scaler used for numerical feature scaling.

    Usage:
    ```
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    ```
    """

    cat_cols = ['payment_type', 'employment_status',
                'housing_status', 'source', 'device_os']

    num_cols = ['income', 'name_email_similarity',
                'prev_address_months_count', 'current_address_months_count',
                'customer_age', 'days_since_request', 'intended_balcon_amount',
                'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w',
                'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
                'credit_risk_score', 'email_is_free', 'phone_home_valid',
                'phone_mobile_valid', 'bank_months_count', 'has_other_cards',
                'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes',
                'keep_alive_session', 'device_distinct_emails_8w','month']

    # Encode categorical variables

    df_encoded = pd.get_dummies(df, columns=cat_cols)

    # Scale features
    scaler = MinMaxScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    return df_encoded, scaler


def split_data(df_encoded):
    """
    Splits the encoded DataFrame into training, validation, and test sets.

    Parameters:
        df_encoded : DataFrame 
            The encoded DataFrame from which to split the data.

    Returns:
        tuple
            Contains training, validation, and test sets (X_train, X_val, X_test, y_train, y_val, y_test).

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)
    ```
    """
    X = df_encoded.iloc[:, 1:].to_numpy()
    y = df_encoded.iloc[:, 0].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=SEED)

    return X_train, X_val, X_test, y_train, y_val, y_test


def upload_dataset():
    """
    Uploads, preprocesses, encodes, scales, and splits the dataset into training, validation, and test sets.

    Assumes the existence of a global `DATA_PATH` variable pointing to the dataset's location and a `SEED` for reproducibility.

    Returns:
        tuple
            Contains the training, validation, and test sets, feature names, and the scaler.

    Usage:
    ```
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = upload_dataset()
    ```
    """
    set_seed()

    df = pd.read_csv(BASE_DATA_PATH)
    df_preprocessed = preprocess_dataframe(df)
    df_encoded, scaler = encode_and_scale_dataframe(df_preprocessed)
    #print(f"The shape of the main dataframe {df_encoded.shape}") 
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)

    feature_names = df_encoded.columns.tolist()[1:]

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler)


def load_individual_data(client_id, include_val_in_train=False):
    """
    Loads individual or global datasets as PyTorch TensorDatasets, with an option to include validation data in the training set.

    This function dynamically loads training, validation, and test data from CSV files located in a specified directory.
    It can load data for a specific client by ID or global data if the client ID is set to -1. There is an option to merge
    training and validation datasets for scenarios where validation data should be included in training, e.g., for certain
    types of model tuning.

    Parameters:
        client_id : int
            The identifier for the agent's dataset to load. If set to -1, global datasets are loaded.
        include_val_in_train : bool, optional
            Determines whether validation data is included in the training dataset. Default is False.

    Returns:
        tuple
            A tuple containing the training dataset, validation dataset, test dataset, column names of the training features,
            a tensor of test features, and the total exposure calculated from the training (and optionally validation) dataset.

    Examples:
        >>> train_dataset, val_dataset, test_dataset, column_names, test_features, exposure = load_individual_data(-1, True)
        >>> print(f"Training dataset size: {len(train_dataset)}")
    """
    MY_DATA_PATH = DATA_DIR
    suffix = '' if client_id == -1 else f'_{client_id}'

    # Load datasets
    X_train = pd.read_csv(f'{CLIENTS_DATA_DIR}/X_train{suffix}.csv')
    y_train = pd.read_csv(f'{CLIENTS_DATA_DIR}/y_train{suffix}.csv')
    X_val = pd.read_csv(f'{CLIENTS_DATA_DIR}/X_val{suffix}.csv')
    y_val = pd.read_csv(f'{CLIENTS_DATA_DIR}/y_val{suffix}.csv')
    # Assuming test data is the same for all agents
    X_test = pd.read_csv(f'{CLIENTS_DATA_DIR}/X_test.csv')
    y_test = pd.read_csv(f'{CLIENTS_DATA_DIR}/y_test.csv')

    # Merge training and validation datasets if specified
    if include_val_in_train:
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        y_train = pd.concat([y_train, y_val], ignore_index=True)

    # Convert to TensorDatasets
    train_dataset = TensorDataset(torch.tensor(X_train.values).float(), torch.tensor(y_train.values).float())
    val_dataset = TensorDataset(torch.tensor(X_val.values).float(), torch.tensor(y_val.values).float())
    test_dataset = TensorDataset(torch.tensor(X_test.values).float(), torch.tensor(y_test.values).float())

    return (train_dataset, val_dataset, test_dataset, X_train.columns.tolist(), torch.tensor(X_test.values).float())


def uniform_partitions(clients: int = 5, num_features: int = None):
    """
    Splits and saves the dataset into uniform partitions for a specified number of agents.

    This function loads a dataset via a previously defined `upload_dataset` function, then partitions
    the training and validation datasets uniformly across the specified number of agents. Each partition
    is saved to CSV files, containing both features and labels for each agent's training and validation datasets.

    Parameters:
        agents : int, optional
            The number of agents to split the dataset into. Defaults to 10.
        num_features : int, optional
            The number of features in the dataset. Automatically inferred if not specified.

    Notes:
        - Requires `upload_dataset` and `seed_torch` to be defined and accessible within the scope.
        - Saves partitioned data files in the '../data/' directory.

    Example:
        >>> uniform_partitions(agents=3)
        Creates and saves 3 sets of training and validation data for 3 agents, storing them in '../data/clients_data'.

    Raises:
        FileNotFoundError
            If the '../data/clients_data' directory does not exist or cannot be accessed.

    Returns:
        None
            The function does not return a value but saves partitioned datasets to disk.
    """
    # Load the dataset
    X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, X_column_names, _ = upload_dataset()
    num_features = num_features or X_train_sc.shape[1]

    # Define the base path for saving files
    base_path = CLIENTS_DATA_DIR

    # Function to save datasets to CSV
    def save_to_csv(data, filename, column_names=X_column_names):
        pd.DataFrame(data, columns=column_names).to_csv(
            f'{base_path}/{filename}', index=False)

    # Save the global datasets
    save_to_csv(X_train_sc, 'X_train.csv')
    save_to_csv(y_train, 'y_train.csv', ['y'])
    save_to_csv(X_val_sc, 'X_val.csv')
    save_to_csv(y_val, 'y_val.csv', ['y'])
    save_to_csv(X_test_sc[:, :num_features], 'X_test.csv')
    save_to_csv(y_test, 'y_test.csv', ['y'])

    # Prepare and shuffle data
    set_seed()
    train_data = np.hstack((X_train_sc, y_train.reshape(-1, 1)))
    val_data = np.hstack((X_val_sc, y_val.reshape(-1, 1)))
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Split and save partitioned data
    for i in range(clients):
        partition_train = np.array_split(train_data, clients)[i]
        partition_val = np.array_split(val_data, clients)[i]

        save_to_csv(partition_train[:, :num_features], f'X_train_{i}.csv')
        save_to_csv(partition_train[:, num_features:],f'y_train_{i}.csv', ['y'])
        save_to_csv(partition_val[:, :num_features], f'X_val_{i}.csv')
        save_to_csv(partition_val[:, num_features:], f'y_val_{i}.csv', ['y'])
