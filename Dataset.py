from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset

class MNIST_DATA:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = datasets.MNIST(root="./Data", train=True, download=True, transform=self.transform)
        self.val_dataset = datasets.MNIST(root="./Data", train=False, download=True, transform=self.transform)
        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}
    
class SVHN_DATA:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
            transforms.Resize((64,64))
        ])

        self.train_dataset = datasets.SVHN(root="./Data/SVHN", split="train", download=True, transform=self.transform)
        self.val_dataset = datasets.SVHN(root="./Data/SVHN", split="test", download=True, transform=self.transform)
        self.class_names = [str(i) for i in range(10)]

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}
    
class CELEBA_DATA:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        train_full_dataset = datasets.CelebA(root="./Data", split="train", target_type="attr", download=True, transform=self.transform)
        self.val_dataset = datasets.CelebA(root="./Data", split="valid", target_type="attr", download=True, transform=self.transform)
        self.train_dataset, _ = random_split(train_full_dataset, [100000, 62770])
        self.class_names = train_full_dataset.attr_names

    def get_loaders(self, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_class_names(self):
        return {i: name for i, name in enumerate(self.class_names)}