import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from odd_even_partitioner import OddEvenPartitioner


def load_cifar10(num_clients=10, batch_size=32):
    """
    Returns a partitioned version of the CIFAR10 dataset as per the OddEvenPartitioner
    Odd numbered clients get twice the size of even numbered clients

    Code mostly the same as in the flower tutorial except it uses the newly created OddEvenPartitioner class

    Parameters
    ----------
    num_clients: int
    batch_size: int

    Returns
    ------
    trainloaders: list[DataLoader]
        training data for each client as a list of torch DataLoader objects
    valloaders: list[DataLoader]
        validation data for each client as a list of torch DataLoader objects
    testloader: DataLoader
        test data as a torch DataLoader object, to be used at the server in our setting
    """
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": OddEvenPartitioner(num_partitions=num_clients)})

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=batch_size))
    testset = fds.load_full("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
