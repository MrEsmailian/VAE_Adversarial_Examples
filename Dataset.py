from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
from Data.mnist_m import mnist_m
from Data.synthetic_digits import synthetic_digits
import torch

TRAIN_SIZE = 54000
TEST_SIZE = 10000
VALIDATION_SIZE = 6000


def _labels(dataset):
    for attr in ("labels", "targets"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if values is not None:
                return torch.as_tensor(values).reshape(-1).tolist()
    return [int(dataset[i][1]) for i in range(len(dataset))]

def _sample(dataset, limit, seed):
    if len(dataset) <= limit:
        return dataset
    labels = _labels(dataset)
    indices, _ = train_test_split(
        range(len(dataset)), train_size=limit, stratify=labels,
        random_state=seed
    )
    return Subset(dataset, indices)

def _split_train_val(dataset, seed):
    dataset = _sample(dataset, TRAIN_SIZE + VALIDATION_SIZE, seed)
    if isinstance(dataset, Subset):
        base_labels = _labels(dataset.dataset)
        labels = [base_labels[i] for i in dataset.indices]
    else:
        labels = _labels(dataset)
    indices_train, indices_val = train_test_split(
        range(len(dataset)), test_size=min(VALIDATION_SIZE, len(dataset) - 1),
        stratify=labels, random_state=seed + 1
    )
    return Subset(dataset, indices_train), Subset(dataset, indices_val)


class MNIST_M_DATA:
    def __init__(self, random_seed=42):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        train_val_dataset = mnist_m(root="./Data/MNIST_M", train=True, download=True, transform=self.transform)
        self.test_dataset = mnist_m(root="./Data/MNIST_M", train=False, download=True, transform=self.transform)

        self.train_dataset, self.val_dataset = _split_train_val(train_val_dataset, random_seed)
        self.test_dataset = _sample(self.test_dataset, TEST_SIZE, seed=random_seed)

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def info(self):
        image, _ = self.train_dataset[0]

        print(f"Train sample count: {len(self.train_dataset)}")
        print(f"Test sample count: {len(self.test_dataset)}")
        print(f"Validation sample count: {len(self.val_dataset)}")
        print(f"Image shape: {tuple(image.shape)}")

        print("Label counts for Train data:")
        labels = torch.tensor([int(self.train_dataset[i][1]) for i in range(len(self.train_dataset))])
        counts = torch.bincount(labels, minlength=10)

        for label, count in enumerate(counts.tolist()):
            print(f"\t#{label}: {count}", end=',')
        print()

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}


class SVHN_DATA:
    def __init__(self, random_seed=42):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        train_val_dataset = datasets.SVHN(root="./Data/SVHN", split="train", download=True, transform=self.transform)
        self.test_dataset = datasets.SVHN(root="./Data/SVHN", split="test", download=True, transform=self.transform)

        self.train_dataset, self.val_dataset = _split_train_val(train_val_dataset, random_seed)
        self.test_dataset = _sample(self.test_dataset, TEST_SIZE, seed=random_seed)

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def info(self):
        image, _ = self.train_dataset[0]

        print(f"Train sample count: {len(self.train_dataset)}")
        print(f"Test sample count: {len(self.test_dataset)}")
        print(f"Validation sample count: {len(self.val_dataset)}")
        print(f"Image shape: {tuple(image.shape)}")

        print("Label counts for Train data:")
        labels = torch.tensor([int(self.train_dataset[i][1]) for i in range(len(self.train_dataset))])
        counts = torch.bincount(labels, minlength=10)

        for label, count in enumerate(counts.tolist()):
            print(f"\t#{label}: {count}", end=',')
        print()

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}


class SYN_DIGITS_DATA:
    def __init__(self, random_seed=42):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        train_val_dataset = synthetic_digits(root="./Data/SynthDigits", train=True, download=True, transform=self.transform)
        self.test_dataset = synthetic_digits(root="./Data/SynthDigits", train=False, download=True, transform=self.transform)

        self.train_dataset, self.val_dataset = _split_train_val(train_val_dataset, random_seed)
        self.test_dataset = _sample(self.test_dataset, TEST_SIZE, seed=random_seed)

        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def info(self):
        image, _ = self.train_dataset[0]

        print(f"Train sample count: {len(self.train_dataset)}")
        print(f"Test sample count: {len(self.test_dataset)}")
        print(f"Validation sample count: {len(self.val_dataset)}")
        print(f"Image shape: {tuple(image.shape)}")

        print("Label counts for Train data:")
        labels = torch.tensor([int(self.train_dataset[i][1]) for i in range(len(self.train_dataset))])
        counts = torch.bincount(labels, minlength=10)

        for label, count in enumerate(counts.tolist()):
            print(f"\t#{label}: {count}", end=',')
        print()

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}


class COMBINED_DATA:
    def __init__(self, *datasets):
        if len(datasets) < 2:
            raise ValueError("At least two datasets are required.")

        self.train_dataset = ConcatDataset([d.train_dataset for d in datasets])
        self.val_dataset = ConcatDataset([d.val_dataset for d in datasets])
        self.test_dataset = ConcatDataset([d.test_dataset for d in datasets])

        self.class_names = datasets[0].class_names

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def info(self):
        image, _ = self.train_dataset[0]

        print(f"Train sample count: {len(self.train_dataset)}")
        print(f"Test sample count: {len(self.test_dataset)}")
        print(f"Validation sample count: {len(self.val_dataset)}")
        print(f"Image shape: {tuple(image.shape)}")
        print(f"Pixel values range: [{image.min().item():.2f}, {image.max().item():.2f}]")

        print("Label counts for Train data:")
        labels = torch.tensor([int(self.train_dataset[i][1]) for i in range(len(self.train_dataset))])
        counts = torch.bincount(labels, minlength=10)

        for label, count in enumerate(counts.tolist()):
            print(f"\t#{label}: {count}", end=',')
        print()

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}