from src.config import NUM_FEATURES, NUM_CLASSES, SERVER_CONFIG, INPUT_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, DIM_FEEDFORWARD
from src.local_utility import FraudDetectionModel, TransformerModel, set_weights, get_weights, load_test_data, set_device
from src.config import NUM_FEATURES, NUM_CLASSES, SERVER_CONFIG
from src.train_decentralized import evaluate_model, final_test_evaluation
from src.local_utility import FraudDetectionModel, set_weights, get_weights, load_test_data, set_device
from typing import List, Tuple

from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation


DEVICE = set_device()

# ---------------------------------- FLOWER SERVER -----------------------------------


def evaluate(server_round, parameters, config):
    """
    Evaluate the global model on the test set after each round.

    Args:
        server_round (int): The current round of federated learning.
        parameters (List[np.ndarray]): The global model parameters received from clients.
        config (Dict[str, Any]): Configuration settings for evaluation.

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
            - loss (float): The average loss on the test dataset.
            - metrics (Dict[str, float]): A dictionary with evaluation metrics (e.g., accuracy).
    """
    model = FraudDetectionModel(
        num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    set_weights(model, parameters)
    test_set = load_test_data()
    loss, accuracy = evaluate_model(model, test_set)

    # Run detailed evaluation **only after the last round**
    if server_round == SERVER_CONFIG.get("num_rounds"):
        all_metrics = final_test_evaluation(model, test_set)

    return loss, {"accuracy": accuracy}


def evaluate_transformer(server_round, parameters, config):
    """
    Evaluate the global model on the test set after each round.

    Args:
        server_round (int): The current round of federated learning.
        parameters (List[np.ndarray]): The global model parameters received from clients.
        config (Dict[str, Any]): Configuration settings for evaluation.

    Returns:
        Tuple[float, Dict[str, float]]: A tuple containing:
            - loss (float): The average loss on the test dataset.
            - metrics (Dict[str, float]): A dictionary with evaluation metrics (e.g., accuracy).
    """
    model = TransformerModel(INPUT_DIM, NUM_CLASSES,
                             NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT)
    model.to(DEVICE)
    set_weights(model, parameters)
    test_set = load_test_data()
    loss, accuracy = evaluate_model(model, test_set)

    # Run detailed evaluation **only after the last round**
    if server_round == SERVER_CONFIG.get("num_rounds"):
        all_metrics = final_test_evaluation(model, test_set)

    return loss, {"accuracy": accuracy}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute the weighted average accuracy across multiple clients.

    This function calculates the weighted average of accuracy scores 
    from multiple clients, where the weight is the number of examples 
    each client contributes.

    Args:
        metrics (List[Tuple[int, Metrics]]): 
            A list of tuples, where each tuple contains:
            - num_examples (int): The number of examples used by a client.
            - m (Metrics): A dictionary containing accuracy metrics, e.g., {"accuracy": value}.

    Returns:
        Metrics: 
            A dictionary containing the aggregated accuracy metric:
            - "accuracy" (float): The weighted average accuracy across all clients.

    Example:
        >>> metrics = [(100, {"accuracy": 0.85}), (200, {"accuracy": 0.90})]
        >>> weighted_average(metrics)
        {"accuracy": 0.8833}
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
