import os

# Ray automatically suppresses duplicate looking logs
# Setting the RAY_DEDUP_LOGS flag to 0 turns it off and helps us monitor learning rate at each client
# Helps in verifying that half of the clients are receiving a different config as intended
os.environ['RAY_DEDUP_LOGS'] = '0'  # must be set before importing ray for it to work

import torch
import flwr as fl
from data_loader import load_cifar10
from server import get_evaluate_function, get_fit_config_function_without_lr, get_fit_config_function
import matplotlib.pyplot as plt
import sys
import warnings

from model import Net, get_parameters
from client_lrdecay import get_lrdecay_client_fn
from client_fedpidavg import get_fedpidavgclient_fn
from client_fedcostwavg import get_fedcostwag_client_fn

from strategy import (LRDecay,
                      FedPIDAvg,
                      FedCostWAvg,
                      FedPIDAVGLRDecay,
                      FedCostWAvgLRDecay)
from flwr.server.strategy import FedAvg

warnings.filterwarnings("ignore", category=UserWarning)

NUM_CLIENTS = int(sys.argv[1])
ILR = float(sys.argv[2])  # initial learning rate
DECAY_RATE = float(sys.argv[3])  # we use exponential learning rate decay w_(i+1) = w_i*(DECAY_RATE)^(round_num - 1)
NUM_EPOCHS = int(sys.argv[4])
NUM_ROUNDS = int(sys.argv[5])
STRATEGY = str(sys.argv[6])

# Load the dataset (CIFAR10 imported from flwr_datasets and partitioned with OddEvenPartitioner)
# Clients with odd number ID will have size twice that of the clients with even number ID
trainloaders, valloaders, testloader = load_cifar10(num_clients=NUM_CLIENTS)

# set the evaluation function on the server side.
# this function is called after round of parameter aggregation
# evaluates the accuracy of the global model on data held at the server
evaluate_fn = get_evaluate_function(Net(), testloader)

# initialise the model; these values will be sent from the server to every client
init_parameters = fl.common.ndarrays_to_parameters(get_parameters(Net()))

if STRATEGY == 'FedAvg':
    client_fn = get_lrdecay_client_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function(n_epochs=NUM_EPOCHS, lr=ILR)
    strategy = FedAvg(initial_parameters=init_parameters,
                      evaluate_fn=evaluate_fn,
                      on_fit_config_fn=on_fit_config_fn)
elif STRATEGY == 'LRDecay':
    client_fn = get_lrdecay_client_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function_without_lr(n_epochs=NUM_EPOCHS)
    strategy = LRDecay(initial_parameters=init_parameters,
                       evaluate_fn=evaluate_fn,
                       on_fit_config_fn=on_fit_config_fn,
                       initial_learning_rate=ILR,
                       decay_rate=DECAY_RATE)
elif STRATEGY == 'FedCostWAvg':
    client_fn = get_fedcostwag_client_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function(n_epochs=NUM_EPOCHS, lr=ILR)
    strategy = FedCostWAvg(initial_parameters=init_parameters,
                           evaluate_fn=evaluate_fn,
                           on_fit_config_fn=on_fit_config_fn)
elif STRATEGY == 'FedCostWAvgLRDecay':
    client_fn = get_fedcostwag_client_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function_without_lr(n_epochs=NUM_EPOCHS)
    strategy = FedCostWAvgLRDecay(initial_parameters=init_parameters,
                                  evaluate_fn=evaluate_fn,
                                  on_fit_config_fn=on_fit_config_fn,
                                  initial_learning_rate=ILR,
                                  decay_rate=DECAY_RATE)
elif STRATEGY == 'FedPIDAvg':
    client_fn = get_fedpidavgclient_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function(n_epochs=NUM_EPOCHS, lr=ILR)
    strategy = FedPIDAvg(initial_parameters=init_parameters,
                         evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn)
elif STRATEGY == 'FedPIDAVGLRDecay':
    client_fn = get_fedpidavgclient_fn(Net(), trainloaders, valloaders)
    on_fit_config_fn = get_fit_config_function_without_lr(n_epochs=NUM_EPOCHS)
    strategy = FedPIDAVGLRDecay(initial_parameters=init_parameters,
                                evaluate_fn=evaluate_fn,
                                on_fit_config_fn=on_fit_config_fn,
                                initial_learning_rate=ILR,
                                decay_rate=DECAY_RATE)
else:
    raise Exception("Strategy not in: FedAvg, LRDecay, FedCostWAvg, FedCostWAvgLRDecay, FedPIDAvg, FedPIDAVGLRDecay")

# set the resources available for clients during the simulation
client_resources = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# run the simulation
simulation = fl.simulation.start_simulation(
    num_clients=NUM_CLIENTS,
    client_fn=client_fn,
    client_resources=client_resources,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS))

#  collect test accuracies at the server after different rounds
test_accuracies = simulation.metrics_centralized['server_test_accuracy']

# save the results
path = 'results_{0}_{1}_{2}_{3}_{4}.txt'.format(str(NUM_CLIENTS),
                                                str(ILR),
                                                str(DECAY_RATE),
                                                str(NUM_EPOCHS),
                                                str(NUM_ROUNDS))
results_file = open(path, 'a+')
results_file.write(str({'strategy': STRATEGY, 'acc': test_accuracies}))
results_file.write('\n')
results_file.close()

# plot test accuracies as a function of the round number
# plt.plot(*zip(*test_accuracies))
# plt.xlabel("Number of rounds")
# plt.ylabel("Accuracy")
# plt.title("Accuracy of central model over test data")
# plt.show()
