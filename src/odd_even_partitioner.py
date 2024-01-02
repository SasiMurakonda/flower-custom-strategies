from flwr_datasets.partitioner.size_partitioner import SizePartitioner


class OddEvenPartitioner(SizePartitioner):
    """
    Creates partitions of size that are correlated with node_id being even or odd.

    Clients with odd number ID have size twice that of the clients with even number ID.

    Parameters
    ----------
    num_partitions: int
        Number of partitions the data should be split into
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__(num_partitions=num_partitions, node_id_to_size_fn=lambda x: x % 2 + 1)
        if num_partitions <= 0:
            raise ValueError("Number of partitions must be greater than zero.")
