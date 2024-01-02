from typing import Callable, Dict, Optional, List, Tuple
from flwr.common import Parameters, Scalar, NDArrays, FitIns
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager


class LRDecay(FedAvg):
    """
    Custom strategy to send varying learning rates and number of epochs to half of the clients.
    Clients with odd value of cid get an exponentially decaying learning rate.

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
                 initial_learning_rate: float = 0.01,
                 decay_rate: float = 0.99
                 ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         initial_parameters=initial_parameters)
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate

    def __repr__(self) -> str:
        return "LRDecay"

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sends fit instructions to clients after every round
        First imports the instructions based on the fit_config function
        Then appends the client dependent instructions to config

        The fit_config function is used to set number of epochs (in general any parameter that's same for all clients)
        The modification then adds client dependent parameters (learning rate) to the config file
        """

        # evaluate fit configurations based on the fit_config function
        # in this setting, receives the values of server round and number of epochs
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)

        new_fit_configurations = []

        # modify the fit_configurations to add client specific learning
        for (client, fit_configuration) in fit_configurations:
            idx = client.cid
            pre_append_config = fit_configuration.config  # config file before changing

            # clients are split into two halves based on cid being even or odd
            # by default the values of cid are integers in the range of number of clients
            # but the flower simulation can also take custom values of cid
            # in such settings a hash function or a pre-stated categorisation can be used to split the clients
            if int(idx) % 2 == 0:
                config_to_append = {'lr': self.initial_learning_rate}
                config = {**pre_append_config, **config_to_append}  # merge the two dictionaries
                new_fit_configurations.append((client, FitIns(parameters, config)))
            else:
                # we use exponential decay of learning rate
                decreased_lr = self.initial_learning_rate * (self.decay_rate ** (server_round - 1))
                config_to_append = {'lr': decreased_lr}
                config = {**pre_append_config, **config_to_append}  # merge the two dictionaries
                new_fit_configurations.append((client, FitIns(parameters, config)))

        return new_fit_configurations
