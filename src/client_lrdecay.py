import flwr as fl
from flwr.common import Scalar
import torch
from torch.utils.data import DataLoader
from model import train, test, set_parameters, get_parameters
from typing import Dict, List

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LRDecayClient(fl.client.NumPyClient):
    """
    The flower client class for testing learning rate decay, mostly the same as in flower quickstart-pytorch tutorial

    Expects the values of number of epochs and learning rate in the config file from server for fit
    """

    def __init__(self,
                 cid: int,
                 net: torch.nn.Module,
                 trainloader: DataLoader,
                 valloader: DataLoader) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config: Dict[str, Scalar]):
        return get_parameters(self.net)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """
        Receive the aggregated parameters and config file from server
        Train on the local data as per instructions in the config file
        Expects the config file to contain the round number, learning rate, and number of epochs
        Send the updated model and number of examples used for training to the server
        """
        set_parameters(self.net, parameters)
        round_num = config['round_num']
        lr = config['lr']
        n_epochs = config['n_epochs']
        print(f'Client ID is {self.cid} and number of batches in train data is {len(self.trainloader)}')
        print(f'Client ID is {self.cid} and learning rate at round {round_num} is {lr}')
        train(self.net, self.trainloader, epochs=n_epochs, learning_rate=lr)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def get_lrdecay_client_fn(model: torch.nn.Module,
                  trainloaders: List[DataLoader],
                  valloaders: List[DataLoader]):
    """
    Helper function to create the function that generates the flower clients
    """
    def client_fn(cid) -> LRDecayClient:
        net = model.to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return LRDecayClient(cid, net, trainloader, valloader)

    return client_fn

