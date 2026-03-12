#  Copyright (C) 2026 Samsung Electronics
#
# You may not use this file except in compliance with the License. You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

# Code adapted from https://github.com/adap/flower/blob/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist/dataset.py

"""MNIST dataset utilities for federated learning."""

from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split, TensorDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from distutils.dir_util import copy_tree

from .dataset_utils import create_lda_partitions
from .utils import use_torch_seed


use_torch_seed(42)


def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    dataset_cfg: Dict[str, Any] = None,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    dataset_cfg: Dict[str, Any]
        The dictionary configuration for the dataset partitioning.
        It must contain a 'method' key with the enum values PartitioningMethod
        According to partitioning method other parameters are needed:
        For PartitioningMethod = Dirichlet, the 'alpha' parameter
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    datasets, testset = _partition_data(num_clients, dataset_cfg, seed)
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) * val_ratio)
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


# WORKAROUND to overcome the broken mirror url in torchvision
# fault url: "http://yann.lecun.com/exdb/mnist/"
class TempMNIST(MNIST):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.mirrors = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
        ]
        super().__init__(root, transform=transform, target_transform=target_transform,
                         train=train, download=download)


def _download_data(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    dataset_dir = './dataset'

    if dataset_name == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        trainset = TempMNIST(dataset_dir, train=True, download=True, transform=transform)
        testset = TempMNIST(dataset_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )
        trainset = CIFAR10(dataset_dir, train=True, download=True, transform=transform)
        testset = CIFAR10(dataset_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        )
        trainset = CIFAR100(dataset_dir, train=True, download=True, transform=transform)
        testset = CIFAR100(dataset_dir, train=False, download=True, transform=transform)

    else:
        raise ValueError('Unknown dataset!')

    return trainset, testset


def _partition_data(
    num_clients: int = 10,
    dataset_cfg: Dict[str, Any] = None,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    dataset_cfg: Dict[str, Any]
        The dictionary configuration for the dataset partitioning.
        It must contain a 'method' key with the enum values PartitioningMethod
        According to partitioning method other parameters are needed:
        For PartitioningMethod = Dirichlet, the 'alpha' parameter
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    trainset, testset = _download_data(dataset_cfg.name)
    trainset_len = len(trainset)
    partition_size = int(trainset_len / num_clients)
    if dataset_cfg.method == 1: # IID
        lengths = [partition_size] * num_clients
        lengths_sum = sum(lengths)
        trainset_final = random_split(trainset, [lengths_sum, trainset_len - lengths_sum])
        datasets = random_split(trainset_final[0], lengths, torch.Generator().manual_seed(seed))
    else:
        # Non-IID partinioning using LDA method
        y = trainset.targets
        if dataset_cfg.name == 'cifar10' or dataset_cfg.name == 'cifar100':
            y = torch.IntTensor(y)
        # WORKAROUND to use indices instead of samples,
        # used below in Subsets of original datasets (accessed from disk)
        x = torch.arange(0, len(y), dtype=int)
        lda_parts = create_lda_partitions(dataset=(x,y), num_partitions=num_clients,
                                        concentration=dataset_cfg.alpha,
                                        accept_imbalanced=not dataset_cfg.equal_samples,
                                        seed=seed)
        datasets = []
        for p in lda_parts[0]:
            idxs = p[0].tolist()
            td = Subset(trainset, idxs)
            # extract samples to fresh new dataset (no references to original)
            tmp_dl = DataLoader(td, batch_size=len(td))
            x, y = next(iter(tmp_dl))
            td = TensorDataset(x, y)
            datasets.append(td)


    return datasets, testset


