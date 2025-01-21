import sys 
import argparse
from src import config
from src.local_utility import uniform_partitions, row_check

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for federated learning dataset preparation and validation.
    Returns:
        argparse.Namespace 
        An object containing the parsed command line arguments with the following attributes
            num_clients : int 
                Specifies the number of clients among which the dataset will be partitioned.
    """
    # Check if running in Jupyter
    if "ipykernel" in sys.modules:
        # Return default values for Jupyter
        return argparse.Namespace(
            num_clients=int(config.dataset_config.get("num_clients", 5))
        )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Prepare and validate dataset for federated learning."
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=int(config.dataset_config.get("num_clients", 5)),
        help="Specifies the number of clients to distribute the dataset across. Defaults to the value in config file.",
    )
    return parser.parse_args()

def prepare_and_validate_dataset(num_clients: int):
    """
    Prepares the dataset for federated learning by partitioning it uniformly (by number of records) among the specified number of agents.
    Performs validation checks to ensure the integrity of the dataset partitioning.
    Parameters:
        num_clients : int
            The number of agents among which the dataset will be partitioned.
    """
    # Prepare the dataset by partitioning it uniformly among the specified number of clients
    uniform_partitions(num_clients)

    # Perform a row check to validate the integrity of the dataset partitioning
    row_check(num_clients)

def simulate_dataset():
    """
    Main execution function that orchestrates the dataset preparation and validation for federated learning.
    """
    print(f"Simulating Federated Learning Dataset for {config.server_config.get('num_clients')} Clients ...")
    # Parse command line arguments to determine the configuration for federated learning
    args = parse_arguments()

    # Prepare the dataset and validate it based on the specified number of agents
    prepare_and_validate_dataset(args.num_clients)

    print("Dataset preparation and validation completed successfully.")

if __name__ == "__main__":
    simulate_dataset()
