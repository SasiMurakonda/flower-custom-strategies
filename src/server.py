from typing import Dict, Optional, Tuple
import flwr as fl
import torch
from model import test, set_parameters

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_fit_config_function_without_lr(n_epochs: int = 1):
    def fit_config(server_round: int):
        """Returns training configuration dict for each round.
        """
        config = {
            "round_num": server_round,
            "n_epochs": n_epochs
        }
        return config

    return fit_config


def get_fit_config_function(n_epochs: int = 1, lr: float = 0.01):
    def fit_config(server_round: int):
        """Returns training configuration dict for each round.
        """
        config = {
            "round_num": server_round,
            "n_epochs": n_epochs,
            "lr": lr
        }
        return config

    return fit_config


def get_evaluate_function(model, testloader):
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar], ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        net = model.to(DEVICE)
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader)

        return loss, {"server_test_accuracy": accuracy}

    return evaluate
