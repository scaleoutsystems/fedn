import torchvision

if __name__ == '__main__':

    for trainset in [True, False]:
        dataset = torchvision.datasets.MNIST('./data', train=trainset, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))