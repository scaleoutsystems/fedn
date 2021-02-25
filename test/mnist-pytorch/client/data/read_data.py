import torchvision
from torch.utils.data import Subset
import numpy as np

def read_data(trainset=True, nr_examples=1000,bias=0.7):
    """ Helper function to read and preprocess data for training with Keras. """

    dataset = torchvision.datasets.MNIST('../app/data', train=trainset, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    sample = np.random.choice(np.arange(len(dataset)),nr_examples,replace=False)
    dataset = Subset(dataset=dataset, indices=sample)

    return dataset



