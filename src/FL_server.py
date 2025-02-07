from typing import List, Tuple 

from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from src.local_utility import FraudDetectionModel, set_weights, get_weights, load_test_data
from src.train_decentralized import evaluate_model, final_test_evaluation
from src.config import NUM_FEATURES, NUM_CLASSES, SERVER_CONFIG


#---------------------------------- FLOWER SERVER -----------------------------------

def evaluate(server_round, parameters, config):
    """
    Evaluate the global model on the test set after each round.

    Args:
        server_round (_type_): _description_
        parameters (_type_): _description_
        config (_type_): _description_
        test_set (_type_): _description_

    Returns:
        _type_: _description_
    """
    model = FraudDetectionModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    set_weights(model, parameters)
    test_set = load_test_data()
    loss, accuracy = evaluate_model(model, test_set)
    
    # Run detailed evaluation **only after the last round**
    if server_round == SERVER_CONFIG.get("num_rounds"):
        all_metrics = final_test_evaluation(model, test_set)
        
    return loss, {"accuracy": accuracy}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

