import hashlib
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

SAMPLERATE = 16000


class BackgroundNoise(Dataset):
    """Dataset for background noise samples using all *.wav in the given path."""

    def __init__(self, path: str, dataset_split_idx: int, dataset_total_splits: int) -> "BackgroundNoise":
        """Initialize the dataset."""
        super().__init__()
        self._path = path
        self._walker = sorted(str(p) for p in Path(self._path).glob("*.wav"))

        self._dataset_split_idx = dataset_split_idx
        self._dataset_total_splits = dataset_total_splits

        self._start_idx = int(self._dataset_split_idx * len(self._walker) / self._dataset_total_splits)
        self._end_idx = int((self._dataset_split_idx + 1) * len(self._walker) / self._dataset_total_splits)

        self._loudness_meter = pyln.Meter(SAMPLERATE)

        self._loudness = [self._loudness_meter.integrated_loudness(self._load_audio(file)[0].numpy()) for file in self._walker]

    def _load_audio(self, filename: str) -> tuple[torch.Tensor, int]:
        data, sr = torchaudio.load(filename)
        data.squeeze_()
        return data, sr

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get the audio sample at the given index."""
        index = index + self._start_idx
        filename = self._walker[index]
        audio, sr = self._load_audio(filename)
        loudness = self._loudness[index]

        audio_np = audio.numpy()
        audio_np = pyln.normalize.loudness(audio_np, loudness, -27)
        audio = torch.from_numpy(audio_np)

        if sr != SAMPLERATE:
            raise ValueError(f"sample rate should be {SAMPLERATE}, but got {sr}")
        return audio

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self._end_idx - self._start_idx


class FedSCDataset(Dataset):
    """Dataset for the Federated Speech Commands dataset."""

    NEGATIVE_KEYWORD = "<negative>"
    SEED = 1

    def __init__(  # noqa: PLR0913
        self, path: str, keywords: list[str], subset: str, dataset_split_idx: int, dataset_total_splits: int, data_augmentation: bool = False
    ) -> "FedSCDataset":
        """Initialize the dataset.

        Args:
            path (str): Path to the dataset.
            keywords (list[str]): List of keywords to detect.
            subset (str): Subset of the dataset to use. One of "training", "validation", or "testing".
            dataset_split_idx (int): Index of the dataset split to use.
            dataset_total_splits (int): Total number of dataset splits.
            data_augmentation (bool): Whether to apply data augmentation.

        """
        super(FedSCDataset, self).__init__()
        self._path = path
        self._subset = subset
        self._labels = keywords + [self.NEGATIVE_KEYWORD]

        self._dataset_split_idx = dataset_split_idx
        self._dataset_total_splits = dataset_total_splits

        Path(path).mkdir(parents=True, exist_ok=True)
        self._dataset = torchaudio.datasets.SPEECHCOMMANDS(path, subset=subset, download=True)
        self._start_idx = int(dataset_split_idx * len(self._dataset) / self._dataset_total_splits)
        self._end_idx = int((dataset_split_idx + 1) * len(self._dataset) / self._dataset_total_splits)

        if data_augmentation:
            self._noise_prob = 0.5
            self._noise_mag = 0.9
            self._noise_dataset = BackgroundNoise(
                Path(self._dataset._path).joinpath("_background_noise_").as_posix(),
                dataset_split_idx=self._dataset_split_idx,
                dataset_total_splits=self._dataset_total_splits,
            )
        else:
            self._noise_prob = 0.0
            self._noise_mag = 0.0
            self._noise_dataset = None

        # All splits use the same rng to shuffle the dataset
        self._rng = np.random.RandomState(self.SEED)
        self._shuffle_order = self._rng.permutation(len(self._dataset))

        self._n_mels = 64
        self._hop_length = 160
        self._white_noise_mag = 0.0015
        self._transform = self._get_spectogram_transform(self._n_mels, self._hop_length, SAMPLERATE, data_augmentation)

        self._spectrogram_size = (self._n_mels, SAMPLERATE // self._hop_length)

        # Reinitialize rng with different seeds fot the different splits
        self._rng = np.random.RandomState(self.SEED + self._dataset_split_idx)

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def n_mels(self) -> int:
        return self._n_mels

    @property
    def n_labels(self) -> int:
        return len(self._labels)

    @property
    def spectrogram_size(self) -> tuple[int, int]:
        return (self.n_mels, 100)

    def __getitem__(self, index: int) -> tuple[int, str, torch.Tensor, torch.Tensor]:
        shuffled_index = self._shuffle_order[index]
        sample, sr, label, _, _ = self._dataset[shuffled_index]
        sample.squeeze_()
        if sample.shape[-1] < SAMPLERATE:
            sample = torch.nn.functional.pad(sample, (0, SAMPLERATE - sample.shape[-1]))

        if sr != SAMPLERATE:
            raise Exception("Samplerate from sample: " + str(shuffled_index) + " is not equal to: " + str(SAMPLERATE))
        if sample.shape[-1] != SAMPLERATE:
            raise Exception("Sample: " + str(shuffled_index) + " is not one second " + str(sample.shape))

        if self._noise_dataset and len(self._noise_dataset) and self._noise_prob > self._rng.rand():
            noise_idx = self._rng.randint(len(self._noise_dataset))
            waveform = self._noise_dataset[noise_idx]
            sub_start_idx = self._rng.randint(waveform.shape[-1] - SAMPLERATE)
            noise = waveform[sub_start_idx : sub_start_idx + SAMPLERATE]
            sample += self._noise_mag * noise * self._rng.rand()

        sample += self._rng.normal(scale=self._white_noise_mag, size=sample.shape).astype(np.float32)

        y = self.get_label_from_text(label)
        spectrogram = self.get_spectrogram(sample)

        return y, label, spectrogram, sample

    def __len__(self) -> int:
        return self._end_idx - self._start_idx

    def get_label_from_text(self, text_label: str) -> int:
        if text_label in self._labels:
            y = self._labels.index(text_label)
        else:
            y = len(self._labels) - 1
        return y

    def get_spectrogram(self, sample: int) -> torch.Tensor:
        start_idx = self._rng.randint(0, self._hop_length)
        length = sample.shape[0] - self._hop_length
        return self._transform(sample[start_idx : start_idx + length])

    def get_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the mean and std of the training data. If the stats are already calculated, they are loaded from disk."""
        sha1 = hashlib.sha1()  # noqa:S324
        for word in self.labels:
            sha1.update(str.encode(word))
        sha1.update(str.encode(str(self._n_mels)))
        sha1.update(str.encode(str(self._dataset_split_idx)))
        sha1.update(str.encode(str(self._dataset_total_splits)))
        sha1.update(str.encode(str(self._hop_length)))
        sha1.update(str.encode(str(self.SEED)))
        guid = sha1.hexdigest()
        filepath = Path(self._path)
        filepath = filepath.joinpath(guid + ".stats")
        if filepath.exists():
            with open(filepath, "r") as file:
                data = json.load(file)
                if (
                    data["n_mels"] == self._n_mels
                    and data["labels"] == self.labels
                    and data["split_index"] == self._dataset_split_idx
                    and data["dataset_total_splits"] == self._dataset_total_splits
                    and data["hop_length"] == self._hop_length
                    and data["white_noise_mag"] == self._white_noise_mag
                    and data["SEED"] == self.SEED
                ):
                    return torch.tensor(data["label_mean"]), torch.tensor(data["spectrogram_mean"])[:, None], torch.tensor(data["spectrogram_std"])[:, None]

        dataset = FedSCDataset(self._path, [], subset="training", dataset_split_idx=self._dataset_split_idx, dataset_total_splits=self._dataset_total_splits)
        label_count = np.zeros(len(self.labels))
        spectrogram_sum = torch.zeros(self._n_mels)
        spectrogram_square_sum = torch.zeros(self._n_mels)

        print("Calculating training data statistics...")  # noqa:T201
        n_samples = len(dataset)
        n_spectrogram_cols = 0
        for i in range(n_samples):
            _, label, spectrogram, _ = dataset[i]
            spectrogram_sum += spectrogram.sum(-1)
            spectrogram_square_sum += spectrogram.square().sum(-1)

            n_spectrogram_cols += spectrogram.shape[-1]

            if label in self.labels:
                idx = self.labels.index(label)
                label_count[idx] += 1
            else:
                label_count[-1] += 1

        label_mean = label_count / n_samples
        spectrogram_mean = spectrogram_sum / n_spectrogram_cols
        spectrogram_std = (spectrogram_square_sum - spectrogram_mean.square()) / (n_spectrogram_cols - 1)
        spectrogram_std.sqrt_()
        with open(filepath, "w") as file:
            d = {
                "labels": self.labels,
                "n_mels": self._n_mels,
                "white_noise_mag": self._white_noise_mag,
                "SEED": self.SEED,
                "split_index": self._dataset_split_idx,
                "dataset_total_splits": self._dataset_total_splits,
                "hop_length": self._hop_length,
                "label_mean": label_mean.tolist(),
                "spectrogram_mean": spectrogram_mean.numpy().tolist(),
                "spectrogram_std": spectrogram_std.numpy().tolist(),
            }
            json.dump(d, file)
        return torch.tensor(label_mean), spectrogram_mean[:, None], spectrogram_std[:, None]

    def get_collate_fn(self) -> callable:
        def collate_fn(batch: tuple[int, str, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            ys, _, spectrogram, _ = zip(*batch)
            return torch.tensor(ys, dtype=torch.long), torch.stack(spectrogram)

        return collate_fn

    def _get_spectogram_transform(self, n_mels: int, hop_length: int, sr: int, data_augmentation: bool = False) -> torch.nn.Sequential:
        if data_augmentation:
            return torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=320, hop_length=hop_length, n_mels=n_mels),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_mels * 0.2)),
                torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000 / 160)),
                torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
            )
        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=320, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80),
        )


def get_dataloaders(
    path: str, keywords: list[str], dataset_split_idx: int, dataset_total_splits: int, batchsize_train: int, batchsize_valid: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get the dataloaders for the training, validation, and testing datasets."""
    dataset_train = FedSCDataset(path, keywords, "training", dataset_split_idx, dataset_total_splits, data_augmentation=True)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batchsize_train, collate_fn=dataset_train.get_collate_fn(), shuffle=True, drop_last=True)

    dataset_valid = FedSCDataset(path, keywords, "validation", dataset_split_idx, dataset_total_splits)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=batchsize_valid, collate_fn=dataset_valid.get_collate_fn(), shuffle=False, drop_last=False)

    dataset_test = FedSCDataset(path, keywords, "testing", dataset_split_idx, dataset_total_splits)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batchsize_valid, collate_fn=dataset_test.get_collate_fn(), shuffle=False, drop_last=False)

    return dataloader_train, dataloader_valid, dataloader_test
