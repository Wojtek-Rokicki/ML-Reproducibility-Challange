from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms


def get_mnist_dataset(train=True):
    return MNIST("./data", train=train, download=True,
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.7,), (0.7,))]))


def get_fashion_mnist_dataset(train=True):
    return FashionMNIST("./data", train=train, download=True,
                        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.7,), (0.7,))]))
