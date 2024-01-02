from typing import Callable, Dict, Optional, List, Tuple, Union
from flwr.common import Parameters, Scalar, NDArrays, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_proxy import ClientProxy

ALPHA = 0.5


class FedCostWAvg(FedAvg):
    """
    Federated Cost Weighted Averaging as implemented in this work:
    https://arxiv.org/pdf/2111.08649.pdf

    It won the MICCAI Federated Tumor Segmentation Challenge 2021 (FETS)
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
                 alpha: float = ALPHA
                 ) -> None:
        super().__init__(fraction_fit=fraction_fit,
                         fraction_evaluate=fraction_evaluate,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn,
                         initial_parameters=initial_parameters)
        self.alpha = alpha

    def __repr__(self) -> str:
        return "FedCostWAvg"

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        _, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        total_n_examples, sum_loss_ratio = map(sum, zip(*[[fit_res.num_examples,
                                                           fit_res.metrics['loss_ratio']]
                                                          for _, fit_res in results]))

        # alpha/S in equation(2) of https://arxiv.org/pdf/2111.08649.pdf
        n_example_factor = self.alpha / total_n_examples if total_n_examples != 0 else 0

        # (1-alpha)/K in equation(2) of https://arxiv.org/pdf/2111.08649.pdf
        loss_diff_factor = (1 - self.alpha) / sum_loss_ratio if sum_loss_ratio != 0 else 0

        # compute updated weights for parameters from each client
        print("Using FedCostWAvg for aggregation of parameters")
        weight_results = [(parameters_to_ndarrays(fit_res.parameters),
                           n_example_factor * fit_res.num_examples +
                           loss_diff_factor * fit_res.metrics['loss_ratio']
                           )
                          for _, fit_res in results]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weight_results))

        return parameters_aggregated, metrics_aggregated
