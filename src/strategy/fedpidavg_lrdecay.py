from typing import Callable, Dict, Optional, List, Tuple
from flwr.common import Parameters, Scalar, NDArrays, FitIns

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .fedpidavg import FedPIDAvg
from .lrdecay import LRDecay

ALPHA = 0.45
BETA = 0.45


class FedPIDAVGLRDecay(FedPIDAvg, LRDecay):
    """
    Federated PID Weighted Averaging as implemented in this work:
    https://arxiv.org/pdf/2304.12117.pdf, along with decaying learning rate at half of the clients

    It won the MICCAI Federated Tumor Segmentation Challenge 2022 (FETS)
    https://fets-ai.github.io/Challenge/
    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    alpha: float, optional
        Weight assigned to ratio of number of examples during parameter aggregation
    beta: float, optional
        Weight assigned to loss differentials during parameter aggregation
    initial_learning_rate: float, optional
        Initial learning rate for all the clients
    decay_rate: float, optional
        Decay factor in the learning rate for the select fraction of clients
    """

    def __init__(self,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 evaluate_fn: Optional[
                     Callable[
                         [int, NDArrays, Dict[str, Scalar]],
                         Optional[Tuple[float, Dict[str, Scalar]]],
                     ]
                 ] = None,
                 on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
                 initial_parameters: Optional[Parameters] = None,
                 alpha: float = ALPHA,
                 beta: float = BETA,
                 initial_learning_rate: float = 0.01,
                 decay_rate: float = 0.99
                 ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         initial_parameters=initial_parameters,
                         alpha=alpha,
                         beta=beta)
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __repr__(self) -> str:
        return "FedPIDAvgLRDecay"

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sends fit instructions to clients after every round
        First imports the instructions based on fit_config function
        Then appends the client dependent instructions (learning rate and number of epochs) to config
        """

        fit_configurations = LRDecay.configure_fit(self, server_round, parameters, client_manager)

        return fit_configurations
