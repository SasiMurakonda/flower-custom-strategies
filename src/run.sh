#!/bin/bash
# run simulation with parameters in order:
# number of clients, initial learning rate, decay rate, number of epochs, number of rounds, strategy
python3 simulation.py 5 0.01 0.99 1 5 'FedAvg'
rm -rf client_dirs
python3 simulation.py 5 0.01 0.99 1 5 'LRDecay'
rm -rf client_dirs
python3 simulation.py 5 0.01 0.99 1 5 'FedCostWAvg'
rm -rf client_dirs
python3 simulation.py 5 0.01 0.99 1 5 'FedCostWAvgLRDecay'
rm -rf client_dirs
python3 simulation.py 5 0.01 0.99 1 5 'FedPIDAvg'
rm -rf client_dirs
python3 simulation.py 5 0.01 0.99 1 5 'FedPIDAVGLRDecay'
rm -rf client_dirs
python3 plot_results.py 5 0.01 0.99 1 5
