import os
import pickle
from typing import List

import numpy as np
from scipy.stats import dirichlet
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import settings

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
# testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


def fine_to_coarse_labels(fine_labels: np.ndarray) -> np.ndarray:
    coarse = np.array(
        [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    )
    return coarse[fine_labels]


class CIFAR100Federated:
    def __init__(self, root_dir: str = "./data/splits"):
        """Initialize the splitter
        :param root_dir: Directory to save the split datasets
        """
        self.root_dir = root_dir
        self.splits = {}
        os.makedirs(root_dir, exist_ok=True)

        # Load the full dataset
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(24),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        )
        self.trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=self.transform_train)

        self.transform_test = transforms.Compose(
            [transforms.CenterCrop(24), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        )
        self.testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=self.transform_test)

    def create_splits(self, num_splits: int, balanced: bool, iid: bool) -> None:
        """Create dataset splits based on specified parameters
        :param num_splits: Number of splits to create
        :param balanced: Whether splits should have equal size
        :param iid: Whether splits should be IID
        """
        config_key = f"splits_{num_splits}_bal_{balanced}_iid_{iid}"

        if settings.get("MISSING_LABELS", False):
            indices = self._create_splits_with_missing_labels(num_splits, balanced, iid)
        else:    
            if iid:
                indices = self._create_iid_splits(num_splits, balanced)
            else:
                indices = self._create_non_iid_splits(num_splits, balanced)

        # Save splits
        for i, split_indices in enumerate(indices):
            split_path = os.path.join(self.root_dir, f"{config_key}_split_{i}.pkl")
            with open(split_path, "wb") as f:
                pickle.dump(split_indices, f)

        self.splits[config_key] = indices

    def _create_iid_splits(self, num_splits: int, balanced: bool) -> List[np.ndarray]:
        """Create IID splits of the dataset"""
        indices = np.arange(len(self.trainset))
        np.random.shuffle(indices)

        if balanced:
            # Equal size splits
            split_size = len(indices) // num_splits
            return [indices[i * split_size : (i + 1) * split_size] for i in range(num_splits)]
        else:
            # Random size splits
            split_points = sorted(np.random.choice(len(indices) - 1, num_splits - 1, replace=False))
            return np.split(indices, split_points)

    def _create_non_iid_splits(self, num_splits: int, balanced: bool) -> List[np.ndarray]:
        """Create non-IID splits using Pachinko Allocation Method (PAM)"""
        # Initialize parameters
        alpha = 0.1  # Root Dirichlet parameter
        beta = 10.0  # Coarse-to-fine Dirichlet parameter
        total_examples = len(self.trainset)

        # Calculate examples per split
        if balanced:
            examples_per_split = [total_examples // num_splits] * num_splits
        else:
            # Use Dirichlet to create unbalanced split sizes
            split_ratios = np.random.dirichlet([0.5] * num_splits)  # Lower alpha = more unbalanced
            examples_per_split = np.round(split_ratios * total_examples).astype(int)
            # Ensure we use exactly total_examples
            examples_per_split[-1] = total_examples - examples_per_split[:-1].sum()

        # Get fine labels and map them to coarse labels
        fine_labels = np.array(self.trainset.targets)
        coarse_labels = fine_to_coarse_labels(fine_labels)

        # Initialize DAG structure (track available labels)
        available_coarse = list(range(20))  # 20 coarse labels as list instead of set
        available_fine = {c: set(np.where(coarse_labels == c)[0]) for c in available_coarse}

        indices_per_split = []
        for split_idx in range(num_splits):
            split_indices = []
            N = examples_per_split[split_idx]  # Use the pre-calculated split size

            # Sample root distribution over coarse labels
            coarse_probs = dirichlet.rvs(alpha=[alpha] * len(available_coarse), size=1, random_state=RANDOM_SEED + split_idx)[0]

            # Sample fine label distributions for each available coarse label
            fine_distributions = {}
            for c in available_coarse:
                if len(available_fine[c]) > 0:
                    fine_probs = dirichlet.rvs(alpha=[beta] * len(available_fine[c]), size=1, random_state=RANDOM_SEED + split_idx + c)[0]
                    fine_distributions[c] = fine_probs

            # Sample N examples for this split
            for _ in range(N):
                if len(available_coarse) == 0:
                    break

                # Sample coarse label
                coarse_idx = np.random.choice(available_coarse, p=coarse_probs)

                if len(available_fine[coarse_idx]) == 0:
                    # Remove empty coarse label and renormalize
                    idx_to_remove = available_coarse.index(coarse_idx)
                    available_coarse.remove(coarse_idx)
                    coarse_probs = self._renormalize(coarse_probs, idx_to_remove)
                    continue

                # Sample fine label
                fine_probs = fine_distributions[coarse_idx]
                available_fine_indices = list(available_fine[coarse_idx])
                fine_probs = fine_probs[: len(available_fine_indices)]
                fine_probs = fine_probs / fine_probs.sum()  # Renormalize
                fine_idx = np.random.choice(available_fine_indices, p=fine_probs)

                # Add example to split
                split_indices.append(fine_idx)

                # Remove selected example
                available_fine[coarse_idx].remove(fine_idx)

                # Renormalize if necessary
                if len(available_fine[coarse_idx]) == 0:
                    idx_to_remove = available_coarse.index(coarse_idx)
                    available_coarse.remove(coarse_idx)
                    coarse_probs = self._renormalize(coarse_probs, idx_to_remove)

            indices_per_split.append(np.array(split_indices))

        return indices_per_split
    
    def _create_splits_with_missing_labels(self, num_splits: int, balanced: bool, iid: bool,
                                        missing_label_fraction: float = 0.2, seed: int = 0):
        indices = self._create_iid_splits(num_splits, balanced)
        fine_labels = np.array(self.trainset.targets)
        coarse_labels = fine_to_coarse_labels(fine_labels)
        rng = np.random.default_rng(seed)

        print(f"Creating splits with missing labels, removing {missing_label_fraction*100}% of coarse labels from each split except the first half")

        for i in range(num_splits):
            # keep first half intact, remove labels in second half
            if i < num_splits // 2:
                continue

            split_idx = np.array(indices[i])
            present = np.unique(coarse_labels[split_idx])
            K = len(present)
            # round instead of floor; never remove all classes
            num_to_remove = int(round(K * missing_label_fraction))
            num_to_remove = max(0, min(num_to_remove, K - 1))
            if num_to_remove == 0:
                continue

            remove = set(rng.choice(present, size=num_to_remove, replace=False))
            mask = np.array([coarse_labels[j] not in remove for j in split_idx], dtype=bool)
            indices[i] = split_idx[mask]

        return indices

    def _renormalize(self, probs: np.ndarray, removed_idx: int) -> np.ndarray:
        """Implementation of Algorithm 8 from the paper"""
        # Create a list of valid indices (excluding the removed index)
        valid_indices = [i for i in range(len(probs)) if i != removed_idx]

        # Select only the probabilities for valid indices
        valid_probs = probs[valid_indices]

        # Normalize the remaining probabilities
        return valid_probs / valid_probs.sum()

    def get_split(self, split_id: int, num_splits: int, balanced: bool, iid: bool) -> Dataset:
        """Get a specific split of the dataset
        :param split_id: ID of the split to retrieve
        :param num_splits: Total number of splits
        :param balanced: Whether splits are balanced
        :param iid: Whether splits are IID
        :return: Dataset split
        """
        config_key = f"splits_{num_splits}_bal_{balanced}_iid_{iid}"
        split_path = os.path.join(self.root_dir, f"{config_key}_split_{split_id}.pkl")

        if not os.path.exists(split_path):
            self.create_splits(num_splits, balanced, iid)

        with open(split_path, "rb") as f:
            indices = pickle.load(f)  # noqa: S301

        return Subset(self.trainset, indices)


def get_data_loader(num_splits: int = 5, balanced: bool = True, iid: bool = True, batch_size: int = 100, is_train: bool = True, split_id: int = 0) -> DataLoader:
    """Get a data loader for the CIFAR-100 dataset
    :param num_splits: Number of splits to create
    :param balanced: Whether splits are balanced
    :param iid: Whether splits are IID
    :param batch_size: Batch size
    :param is_train: Whether to get the training or test data loader
    :return: Data loader
    """
    cifar_data = CIFAR100Federated()

    if is_train:
        dataset = cifar_data.get_split(split_id=split_id, num_splits=num_splits, balanced=balanced, iid=iid)
        print(f"Getting data loader for split {split_id} of trainset (size: {len(dataset)})")
    else:
        dataset = cifar_data.testset
        print(f"Getting data loader for testset (size: {len(dataset)})")

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
