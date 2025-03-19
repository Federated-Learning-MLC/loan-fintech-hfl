import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from flwr.common import ndarrays_to_parameters, Context, Metrics, Scalar, NDArrays
from flwr.client import Client, ClientApp, NumPyClient

from logging import ERROR

from src.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_FEATURES
from src.paths import CLIENTS_DATA_DIR
from src.train_decentralized import train_model, evaluate_model
from src.local_utility import set_weights, get_weights, set_device


DEVICE = set_device()

# ---------------------------------- FLOWER CLIENT --------------------------------------------


class BankFLClient(NumPyClient):
    """
    A client class for federated learning using Flower framework, designed to handle the 
    training and evaluation of a machine learning model on local data and interact with a 
    federated learning server.

    Attributes:
        - model (torch.nn.Module): The local machine learning model.
        - trainset (torch.utils.data.Dataset): The training dataset.
        - valset (torch.utils.data.Dataset): The validation dataset.
    """

    def __init__(self, model, trainset, valset, optim=None):
        self.device = DEVICE
        self.model = model.to(self.device)
        self.trainset = trainset
        self.valset = valset

        self.optimizer = optim

    # Train model
    def fit(self, parameters, config):
        """
        Trains the model locally using provided parameters and configuration settings.

        This method sets the initial model parameters, trains the model on a local dataset,
        and returns the updated model parameters after training. It encapsulates the process of
        local training within a federated learning framework, including setting initial parameters,
        executing the training loop.
        """
        set_weights(self.model, parameters)

        train_model(self.model, self.trainset, self.optimizer)

        return get_weights(self.model), len(self.trainset), {}

    # Test the model

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        """
        Evaluates the model with the provided global parameters on local test data.

        This method is intended to be used in a federated learning context where global model parameters are evaluated
        on a client's local test dataset. The method sets the model's parameters to the provided global parameters, evaluates
        these parameters on the local test dataset, and returns the evaluation loss, the number of test samples, and a dictionary
        containing evaluation metrics such as accuracy.

        Notes:
            - The accuracy calculation in the returned dictionary is a simplified example. Depending on the model and the task, you might need a more sophisticated method to calculate accuracy or other relevant metrics.\n
            - Ensure that the global `BATCH_SIZE` variable is appropriately set for the evaluation DataLoader to function correctly.
        """
        set_weights(self.model, parameters)
        loss, accuracy = evaluate_model(self.model, self.valset)
        return loss, len(self.valset), {"accuracy": accuracy}
