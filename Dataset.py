from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from Data.mnist_m import mnist_m
from Data.synthetic_digits import synthetic_digits
import torch


class MNIST_M_DATA:
    def __init__(self, val_ratio=0.125, random_seed=42):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_val_dataset = mnist_m(root="./Data/MNIST_M", train=True, download=True, transform=self.transform)
        self.test_dataset = mnist_m(root="./Data/MNIST_M", train=False, download=True, transform=self.transform)

        val_size = int(len(train_val_dataset) * val_ratio)
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [len(train_val_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}


class SVHN_DATA:
    def __init__(self, val_ratio=0.125, random_seed=42):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_val_dataset = datasets.SVHN(root="./Data/SVHN", split="train", download=True, transform=self.transform)
        self.test_dataset = datasets.SVHN(root="./Data/SVHN", split="test", download=True, transform=self.transform)

        val_size = int(len(train_val_dataset) * val_ratio)
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [len(train_val_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}


class SYN_DIGITS_DATA:
    def __init__(self, val_ratio=0.125, random_seed=42):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_val_dataset = synthetic_digits(root="./Data/SynthDigits", train=True, download=True, transform=self.transform)
        self.test_dataset = synthetic_digits(root="./Data/SynthDigits", train=False, download=True, transform=self.transform)

        val_size = int(len(train_val_dataset) * val_ratio)
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [len(train_val_dataset) - val_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}
