import os

import flwr as fl
import torch
from flwr.common import Scalar
from model import train, test, set_parameters, get_parameters
from pathlib import Path
import numpy as np
from collections import deque as dq
from torch.utils.data import DataLoader
from typing import Dict, List

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_LOSSES_TO_SAVE = 2


class FedCostWAvgClient(fl.client.NumPyClient):
    """
    The flower client class for Federated Cost Weighted Averaging. Keeps track of the last few losses.
    https://arxiv.org/pdf/2111.08649.pdf

    Results are stored in a directory for each client so that the values are available to it throughout.
    Expects the values of number of epochs and learning rate in the config file from server for fit
    The Fit function returns the loss weight factor in the metrics section

    Parameters
    ---------
    cid: int
        Client Id
    net: torch.nn.module
        ML model
    trainloader: DataLoader
        Training data
    valloader: DataLoader
        Validation data
    client_dir: Path
        Temporary directory to store the loss history
    num_losses_to_save: int
        Number of losses to keep track of; the algorithm only requires the past two losses to compute loss ratio
    """

    def __init__(self,
                 cid: int,
                 net: torch.nn.Module,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 client_dir: Path,
                 num_losses_to_save: int = NUM_LOSSES_TO_SAVE):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_dir = client_dir
        self.num_losses_to_save = num_losses_to_save

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.net)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """
        Receive the aggregated parameters and config file from server
        Compute loss on training data
        Access directory to update the loss history
        Compute loss ratio as the ratio of the previous two losses (equation 3 in https://arxiv.org/pdf/2111.08649.pdf)
        Train on the local data as per instructions in the config file
        Expects the config file to contain the round number, learning rate, and number of epochs
        Send the updated model, number of examples used for training, and loss ratio to the server
        """

        set_parameters(self.net, parameters)
        n_epochs = config['n_epochs']
        lr = config['lr']
        round_num = config['round_num']
        print(f'Client ID is {self.cid} and number of batches in train data is {len(self.trainloader)}')
        print(f'Client ID is {self.cid} and learning rate at round {round_num} is {lr}')

        # train the local model
        train(self.net, self.trainloader, epochs=n_epochs, learning_rate=lr)

        # compute loss on the training set
        initial_loss, _ = test(self.net, self.trainloader)

        # update loss history in the directory
        if not os.path.exists(self.client_dir):
            os.makedirs(self.client_dir)
            loss_collection = [initial_loss]
            np.save(Path(os.path.join(self.client_dir, 'loss_collection')), np.asarray(loss_collection))
        else:
            loss_collection = dq(np.load(Path(os.path.join(self.client_dir, 'loss_collection.npy'))),
                                 maxlen=self.num_losses_to_save)
            loss_collection.append(initial_loss)
            np.save(Path(os.path.join(self.client_dir, 'loss_collection')), np.asarray(loss_collection))

        # compute loss ratio
        loss_ratio = 0
        if len(loss_collection) > 1:
            loss_ratio = loss_collection[-2] / loss_collection[-1]

        return self.get_parameters(config={}), len(self.trainloader), {'loss_ratio': loss_ratio}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def get_fedcostwag_client_fn(model: torch.nn.Module,
                             trainloaders: List[DataLoader],
                             valloaders: List[DataLoader]):
    """
    Helper function to create the function that generates FedCostWAvgClients
    """

    def client_fn(cid) -> FedCostWAvgClient:
        net = model.to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        client_dir = Path(os.path.join(os.getcwd(), 'client_dirs/' + str(cid)))

        return FedCostWAvgClient(cid, net, trainloader, valloader, client_dir)

    return client_fn
