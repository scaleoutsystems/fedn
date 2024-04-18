import os
from math import floor

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)


# Function to save data in chunks
def save_chunks(dataset, chunk_size, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    num_chunks = len(dataset) // chunk_size + (1 if len(dataset) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(dataset))
        images = []
        labels = []
        # Extract images and labels
        for j in range(start, end):
            image, label = dataset[j]
            images.append(image)
            labels.append(label)

        # Stack images into a single tensor and convert labels to a tensor
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        # Save the tuple of images and labels
        torch.save((images, labels), os.path.join(folder, f'chunk_{i}.pt'))

class SingleChunkDataset(Dataset):
    def __init__(self, chunk_file):
        """
        Initialize the dataset with the path to a specific chunk file.
        """
        # Load the data from the specified chunk file
        self.data = torch.load(chunk_file)
        self.images = self.data[0]  # Images tensor
        self.labels = self.data[1]  # Labels tensor


    def __len__(self):
        """
        Return the total number of samples in the chunk.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieve the image and label at the specified index.
        """
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
def get_data(out_dir='data'):
    # Make dir if necessary
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Only download if not already downloaded
    if not os.path.exists(f'{out_dir}/train'):
        torchvision.datasets.MNIST(
            root=f'{out_dir}/train', transform=torchvision.transforms.ToTensor, train=True, download=True)
    if not os.path.exists(f'{out_dir}/test'):
        torchvision.datasets.MNIST(
            root=f'{out_dir}/test', transform=torchvision.transforms.ToTensor, train=False, download=True)


def load_data(data_path, is_train=True):
    """ Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: dataset.
    :rtype: dataset
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH")

    data_chunk = os.environ.get("FEDN_DATA_CHUNK")


    if is_train:
        # Create an instance of the dataset for a specific chunk
        data_path += '/data/train_chunks/chunk_' + data_chunk + '.pt'

        chunk_dataset = SingleChunkDataset(data_path)
    else:
        data_path += '/data/test_chunks/chunk_' + data_chunk + '.pt'
        chunk_dataset = SingleChunkDataset(data_path)



    return chunk_dataset




def split(out_dir='data'):

    n_splits = int(os.environ.get("FEDN_NUM_DATA_SPLITS", 2))

    # Make dir
    if not os.path.exists(f'{out_dir}/clients'):
        os.mkdir(f'{out_dir}/clients')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist_trainset = torchvision.datasets.MNIST(
        root=f'{out_dir}/train', transform=transform, train=True)
    mnist_testset = torchvision.datasets.MNIST(
        root=f'{out_dir}/test', transform=transform, train=False)

    # Define chunk size and folders
    splits = 2
    chunk_size = int(floor(len(mnist_trainset) / n_splits))  # Define the size of each chunk
    train_folder = 'data/train_chunks'
    test_folder = 'data/test_chunks'

    # Save chunks
    save_chunks(mnist_trainset, chunk_size, train_folder)
    chunk_size = int(floor(len(mnist_testset) / n_splits))  # Define the size of each chunk

    save_chunks(mnist_testset, chunk_size, test_folder)





if __name__ == '__main__':
    # Prepare data if not already done
    if not os.path.exists(abs_path+'/data/clients/1'):
        get_data()
        split()
