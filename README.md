# Custom training strategies with Flower

This folder contains the code to simulate a federated learning setup for training an image classification model. It implements five custom strategies. There is also a new data partition function, which creates partitions of two different sizes - one set containing twice the number of examples as that of the other set.

## What does it do?

The code provided here simulates the below experiment :

1) Download the CIFAR10 dataset from [Flower Datasets](https://flower.dev/docs/datasets/)
2) Partition the train data with our custom OddEvenPartitioner, where odd-numbered partitions will be twice the size of even-numbered partitions.
3) Learn an image classification model with the partitioned train data at different clients and the test data at the server. 
4) Train the local models and aggregate them according the five custom strategies as well as FedAvg. 
5) Evaluate the accuracy of the aggregated model on the server data after every round.
6) Generate a plot comparing the model accuracy on server data at each round for all the strategies.

## What are the custom strategies?

1) LRDecay - train half of the clients with a decaying learning rate and the other half with a fixed learning rate.
2) [FedCostWAvg](https://arxiv.org/pdf/2111.08649.pdf) - a modified version of FedAvg that takes into account loss improvement at clients for determining the weights during parameter aggregation. 
3) [FedPIDAvg](https://arxiv.org/pdf/2304.12117.pdf) - adaptation of FedCostWAvg inspired by the concept of a PID controller.
4) FedCostWAvgLRDecay - FedCostWAvg with LRDecay i.e., FedCostWAvg for model aggregation and a decaying learning rate at half of the clients for local training.
5) FedPIDAvgLRDecay - FedPIDAvg with LRDecay i.e., FedPIDAvg for model aggregation and a decaying learning rate at half of the clients for local training. 

FedCostWAvg and FedPIDAvg are winning submissions to the [Federated Tumor Segmentation Challenge](https://fets-ai.github.io/Challenge/tasks/) 2021 and 2022 respectively. 

## Running the simulation

Install the requirements from requirements.txt and run the bash scrip run.sh in src

```shell
pip install -r requirements.txt
```

```shell
bash run.sh
```

## Results

Results from sample experiments are in the results directory and their corresponding plots in the plots directory. All files follow the same naming convention: results_{num-clients}_{learning-rate}_{decay-rate}_{num-epochs}_{num-rounds}. For example, results_5_0.01_0.99_1_50.png compares the performance of different strategies when there are 5 clients training with a learning rate of 0.01 for 1 epoch, and a decay factor of 0.99 when applicable, for 50 rounds. 