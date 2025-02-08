import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from typing import List, Callable
from collections import OrderedDict

from src.config import BASE_DATA, BASE_DATA_DOWNSAMPLED, NUM_FEATURES, SEED, TARGET_COLS, NUM_COLS, CAT_COLS
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


# ----------------------------------- OTHER USEFUL STUFF ------------------------------

def set_device() -> str:
    """
    Determine the available computing device (GPU or CPU).

    This function checks if a CUDA-compatible GPU is available. If so, 
    it returns "cuda", otherwise, it defaults to "cpu".

    Returns:
        str: The name of the computing device, either "cuda" (GPU) or "cpu".

    Example:
        >>> set_device()
        'cuda'  # If GPU is available
        'cpu'   # If GPU is not available
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def timer(func: Callable)-> Callable:
    """
    A decorator that measures and prints the execution time of a function.

    Args: 
        func (Callable): The function being decorated.

    Returns:
        Callable: The decorated function that prints execution time.

    Example:
        >>> @timer
        >>> def slow_function():
        >>>     time.sleep(2)
        >>> slow_function()
        slow_function took 2.00 secs to execute
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        stop_time = time.time()
        print(
            f"\n{func.__name__} took {stop_time - start_time:.2f} secs to execute\n")
        return result

    return wrapper


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
    expected_row_count = 30_808
    expected_label_sum = 11_029

    # Load and aggregate datasets
    datasets = {"X_train": [], "X_val": [], "y_train": [], "y_val": []}
    for prefix in datasets.keys():
        for i in range(clients):
            datasets[prefix].append(load_data_frame(prefix, i))
    X_test = pd.read_csv(CLIENTS_DATA_DIR/"X_test.csv")
    y_test = pd.read_csv(CLIENTS_DATA_DIR/"y_test.csv")

    # Calculate totals
    total_row_count = sum(len(df) for prefix in [
                          "X_train", "X_val"] for df in datasets[prefix]) + len(X_test)
    total_label_sum = sum(df["y"].sum() for prefix in ["y_train", "y_val"]
                          for df in datasets[prefix]) + y_test["y"].sum()

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
        temp_df = df.copy()
        temp_df[cols_missing_neg1] = temp_df[cols_missing_neg1].replace(
            -1, np.nan)

        # Calculate the percentage of missing values by feature
        null_X = temp_df.isna().sum() / len(df) * 100

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


# ----------------------- Data Preprocessing --------------------------------

def preprocess_data(df):
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

    # Drop all rows with missing data
    df = df.dropna(axis=0)

    X = df[CAT_COLS + NUM_COLS]
    y = df[TARGET_COLS]

    pipeline = encode_and_scale_data(df)
    X_scaled = pipeline.fit_transform(X)
    encoded_cols = pipeline.named_steps['preprocess'].get_feature_names_out(
    ).tolist()

    df_encoded = pd.DataFrame(X_scaled, columns=encoded_cols)
    df_encoded[TARGET_COLS] = y

    return df_encoded, pipeline


def encode_and_scale_data(df):
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

    # Encode categorical variables

    num_preprocessing = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    cat_preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore")),
    ])

    data_preprocessing = ColumnTransformer(
        transformers=[
            ('numerical', num_preprocessing, NUM_COLS),
            ("categorical", cat_preprocessing, CAT_COLS)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[("preprocess", data_preprocessing)])

    return pipeline


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
    X = df_encoded.iloc[:, :-1].to_numpy()
    y = df_encoded.iloc[:, -1].to_numpy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=SEED)
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

    df = pd.read_csv(BASE_DATA_DOWNSAMPLED)  # Using the downsampled base data
    df_encoded, pipeline = preprocess_data(df)

    # print(f"The shape of the main dataframe {df_encoded.shape}")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_encoded)

    feature_names = df_encoded.columns.tolist()[:-1]

    return (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, pipeline)


def uniform_partitions(clients: int = 5, num_features: int = None):
    """
    Splits and saves the dataset into uniform partitions for a specified number of agents.

    This function loads a dataset via a previously defined `upload_dataset` function, then partitions
    the training and validation datasets uniformly across the specified number of agents. Each partition
    is saved to CSV files, containing both features and labels for each agent's training and validation datasets.

    Parameters:
        agents : int, optional
            The number of agents to split the dataset into. Defaults to 5.
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

        save_to_csv(partition_train[:, :num_features].astype(
            np.float32), f'X_train_{i}.csv')
        save_to_csv(partition_train[:, num_features:].astype(
            np.int64), f'y_train_{i}.csv', ['y'])
        save_to_csv(partition_val[:, :num_features].astype(
            np.float32), f'X_val_{i}.csv')
        save_to_csv(partition_val[:, num_features:].astype(
            np.int64), f'y_val_{i}.csv', ['y'])


class FraudDataset(Dataset):
    """
    A custom PyTorch Dataset for loading fraud detection data.

    This dataset loads feature vectors and corresponding labels from CSV files,
    converts them into PyTorch tensors, and provides standard Dataset methods
    for indexing and length retrieval.

    Args:
        x_file (str): Path to the CSV file containing feature data.
        y_file (str): Path to the CSV file containing labels.

    Attributes:
        x_data (torch.Tensor): Feature data stored as a PyTorch tensor.
        y_data (torch.Tensor): Labels stored as a PyTorch tensor.

    Methods:
        __getitem__(idx): Returns the feature-label pair at the given index.
        __len__(): Returns the number of samples in the dataset.

    Example:
        >>> dataset = FraudDataset("features.csv", "labels.csv")
        >>> len(dataset)  # Number of samples
        1000
        >>> x, y = dataset[0]  # Get first sample
        >>> x.shape, y
        (torch.Size([num_features]), tensor(label))
    """
    def __init__(self, x_file: str, y_file: str):

        # Load the features and Label
        self.x_data = pd.read_csv(x_file).values
        # <-- remove singleton & convert to 1D
        self.y_data = pd.read_csv(y_file).values.squeeze()

        # Convert to tensors
        self.x_data = torch.tensor(self.x_data, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.long)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

    def __len__(self):
        return len(self.x_data)


# Function to load client datasets
def load_client_data(client_id: int):
    """
    Loads training and validation datasets for a specific client.

    This function retrieves the training and validation data for a given client,
    constructs `FraudDataset` instances, and returns them.

    Args:
        client_id (int): The unique identifier for the client.

    Returns:
        Tuple[FraudDataset, FraudDataset]: The training and validation datasets for the specified client.

    Example:
        >>> train_data, val_data = load_client_data(1)
        >>> len(train_data), len(val_data)
        (800, 200)
    """

    x_train = CLIENTS_DATA_DIR / f"x_train_{client_id}.csv"
    y_train = CLIENTS_DATA_DIR / f"y_train_{client_id}.csv"
    x_val = CLIENTS_DATA_DIR / f"x_val_{client_id}.csv"
    y_val = CLIENTS_DATA_DIR / f"y_val_{client_id}.csv"

    train_dataset = FraudDataset(x_train, y_train)
    val_dataset = FraudDataset(x_val, y_val)
    return train_dataset, val_dataset


def load_test_data():
    """
    Loads the global test dataset for evaluating the final model.

    This function retrieves the test data, constructs a `FraudDataset` instance,
    and returns it.

    Returns:
        FraudDataset: The test dataset.

    Example:
        >>> test_data = load_test_data()
        >>> len(test_data)
        1000
    """
    test_dataset = FraudDataset(
        CLIENTS_DATA_DIR/"X_test.csv", CLIENTS_DATA_DIR/"y_test.csv")
    return test_dataset


# ----------------------------- NEURAL-NET (MLP ARCHITECTURE) ----------------------------------

class FraudDetectionModel(nn.Module):
    """
    A simple Multi Layer Perceptron (feedforward neural network) for fraud detection.

    This model consists of:
    - An input layer that takes `num_features` features.
    - Two hidden layers with ReLU activations.
    - An output layer with `num_classes` units.

    Args:
        num_features (int): The number of input features.
        num_classes (int): The number of output classes.

    Attributes:
        all_layers (torch.nn.Sequential): The sequential layers defining the model.

    Methods:
        forward(x): Performs a forward pass through the model.

    Example:
        >>> model = FraudDetectionModel(num_features=30, num_classes=2)
        >>> x = torch.randn(1, 30)  # Sample input tensor
        >>> output = model(x)
        >>> output.shape
        torch.Size([1, 2])  # Output matches the number of classes
    """
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = nn.Sequential(

            # 1st hidden layer
            nn.Linear(num_features, 25),
            nn.ReLU(),

            # 2nd hidden layer
            nn.Linear(25, 15),
            nn.ReLU(),

            # output layer
            nn.Linear(15, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: The output logits of shape (batch_size, num_classes).
        """
        logits = self.all_layers(x)
        return logits


# ---------------------------------- FL HELPER FUNCTIONS ---------------------------------------

def get_weights(model) -> List[np.ndarray]:
    """
    Retrieves model parameters (weights & bias) from local model (client side).
    This function extracts the model's parameters as numpy arrays.

    Returns:
        List[np.ndarray]
            A list of numpy arrays representing the model's parameters. 
            Each numpy array in the list corresponds to parameters of a different layer or component of 
            the model.

    Examples:
        >>> model = YourModelClass()
        >>> parameters = get_parameters(model)
        >>> type(parameters)
        <class 'list'>
        >>> type(parameters[0])
        <class 'numpy.ndarray'>
    """
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return weights


def set_weights(model, parameters):
    """
    Updates the model's parameters with new values provided as a list of NumPy ndarrays.

    This function takes a list of NumPy arrays containing new parameter values and updates the local model's
    parameters accordingly. It's typically used to set model parameters after they have been modified
    or updated elsewhere, possibly after aggregation in a federated learning scenario or after receiving
    updates from an optimization process.

    Parameters:
        parameters : List[np.ndarray]
            A list of NumPy ndarrays where each array corresponds to the parameters for a different layer or
            component of the model. The order of the arrays in the list should match the order of parameters
            in the model's state_dict.

    Returns:
        None

    Examples:
        >>> model = YourModelClass()
        >>> new_parameters = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])]
        >>> set_parameters(model, new_parameters)
        >>> # Model parameters are now updated with `new_parameters`.

    Notes:
        - This method assumes that the provided list of parameters matches the structure and order of the model's parameters. If the order or structure of `parameters` does not match, this may lead to incorrect assignment of parameters or runtime errors.
        - The method converts each NumPy ndarray to a PyTorch tensor before updating the model's state dict. Ensure that the data types and device (CPU/GPU) of the NumPy arrays are compatible with your model's requirements.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
