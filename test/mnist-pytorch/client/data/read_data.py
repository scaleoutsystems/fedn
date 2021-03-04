import torchvision
from torch.utils.data import Subset
import numpy as np
import torch
import os
import subprocess as sp
from torchvision.datasets.mnist import MNIST, read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive

def patched_download(self):
    """wget patched download method.
    """
    if self._check_exists():
        return

    os.makedirs(self.raw_folder, exist_ok=True)
    os.makedirs(self.processed_folder, exist_ok=True)

    # download files
    for url, md5 in self.resources:
        filename = url.rpartition('/')[2]
        download_root = os.path.expanduser(self.raw_folder)
        extract_root = None
        remove_finished = False

        if extract_root is None:
            extract_root = download_root
        if not filename:
            filename = os.path.basename(url)

        # Use wget to download archives
        sp.run(["wget", url, "-P", download_root])

        archive = os.path.join(download_root, filename)
        print("Extracting {} to {}".format(archive, extract_root))
        extract_archive(archive, extract_root, remove_finished)

    # process and save as torch files
    print('Processing...')

    training_set = (
        read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
    )
    with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
        torch.save(test_set, f)

def read_data(trainset=True, nr_examples=1000,bias=0.7):
    """ Helper function to read and preprocess data for training with Keras. """
    torchvision.datasets.MNIST.download = patched_download

    dataset = torchvision.datasets.MNIST('../app/data', train=trainset, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    sample = np.random.choice(np.arange(len(dataset)),nr_examples,replace=False)
    dataset = Subset(dataset=dataset, indices=sample)

    return dataset



