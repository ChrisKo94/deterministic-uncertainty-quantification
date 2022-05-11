import math
import os
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms

from scipy.io import loadmat
from PIL import Image


def get_MNIST(root="./"):
    input_size = 28
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root + "data/", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root + "data/", train=False, download=True, transform=transform
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_FashionMNIST(root="./"):
    input_size = 28
    num_classes = 10

    transform_list = [transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(
        root + "data/", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root + "data/", train=False, download=True, transform=transform
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_SVHN(root="./"):
    input_size = 32
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.SVHN(
        root + "data/SVHN", split="train", transform=transform, download=True
    )
    test_dataset = datasets.SVHN(
        root + "data/SVHN", split="test", transform=transform, download=True
    )
    return input_size, num_classes, train_dataset, test_dataset


def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset


def get_notMNIST(root="./"):
    input_size = 28
    num_classes = 10

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4254,), (0.4586,))]
    )

    test_dataset = NotMNIST(root + "data/", transform=transform)

    return input_size, num_classes, None, test_dataset

def get_LCZ42(root="E:/Dateien/LCZ_Votes/"):
    input_size = 32
    num_classes = 10

    train_dataset = LCZ42_train(root)
    val_dataset = LCZ42_val(root)
    test_dataset = LCZ42_test(root)

    return input_size, num_classes, train_dataset, val_dataset, test_dataset


all_datasets = {
    "MNIST": get_MNIST,
    "notMNIST": get_notMNIST,
    "FashionMNIST": get_FashionMNIST,
    "SVHN": get_SVHN,
    "CIFAR10": get_CIFAR10,
    "LCZ42": get_LCZ42,
}


class NotMNIST(Dataset):
    def __init__(self, root, transform=None):
        root = os.path.expanduser(root)

        self.transform = transform

        data_dict = loadmat(os.path.join(root, "notMNIST_small.mat"))

        self.data = torch.tensor(
            data_dict["images"].transpose(2, 0, 1), dtype=torch.uint8
        ).unsqueeze(1)

        self.targets = torch.tensor(data_dict["labels"], dtype=torch.int64)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            img = Image.fromarray(img.squeeze().numpy(), mode="L")
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class FastFashionMNIST(datasets.FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = self.data.unsqueeze(1).float().div(255)
        self.data = self.data.sub_(0.2861).div_(0.3530)

        self.data, self.targets = self.data.to("cuda"), self.targets.to("cuda")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

class LCZ42_train(Dataset):
    def __init__(self, root):

        data_h5 = h5py.File(os.path.join(root, "train_data.h5"))
        targets_h5 = h5py.File(os.path.join(root, "train_label_distributions_data.h5"))
        one_hot_labels = np.array(data_h5["y"])
        indices_in = np.where(np.where(one_hot_labels == np.amax(one_hot_labels, 0))[1] + 1 < 11)[0]

        self.data = torch.tensor(np.array(data_h5["x"][indices_in, :, :, :]),
                                 dtype=torch.float32).permute(0,3,1,2)

        #self.targets = torch.tensor(np.array(targets_h5["train_label_distributions"][:,:10]),
        #                            dtype=torch.float32)
        self.targets = torch.tensor(np.argmax(one_hot_labels[indices_in,:],1), dtype=torch.int64)

        # Subset 50% of data:
        np.random.seed(424242)
        indices_train = np.random.choice(len(self.targets), math.ceil(0.5*len(self.targets)), False)
        self.data = self.data[indices_train, :, :, :]
        self.targets = self.targets[indices_train]

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        return img, target

    def __len__(self):
        return len(self.data)

class LCZ42_val(Dataset):
    def __init__(self, root):

        data_h5 = h5py.File(os.path.join(root, "validation_data.h5"))
        targets_h5 = h5py.File(os.path.join(root, "val_label_distributions_data.h5"))
        one_hot_labels = np.array(data_h5["y"])
        indices_in = np.where(np.where(one_hot_labels == np.amax(one_hot_labels, 0))[1] + 1 < 11)[0]

        self.data = torch.tensor(np.array(data_h5["x"][indices_in, :, :, :]),
                                 dtype=torch.float32).permute(0,3,1,2)

        #self.targets = torch.tensor(np.array(targets_h5["val_label_distributions"][:, :10]),
        #                            dtype=torch.float32)
        self.targets = torch.tensor(np.argmax(one_hot_labels[indices_in,:],1), dtype=torch.int64)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        return img, target

    def __len__(self):
        return len(self.data)

class LCZ42_test(Dataset):
    def __init__(self, root):

        data_h5 = h5py.File(os.path.join(root, "test_data.h5"))
        targets_h5 = h5py.File(os.path.join(root, "test_label_distributions_data.h5"))
        one_hot_labels = np.array(data_h5["y"])
        indices_in = np.where(np.where(one_hot_labels == np.amax(one_hot_labels, 0))[1] + 1 < 11)[0]
        self.data = torch.tensor(np.array(data_h5["x"][indices_in, :, :, :]),
                                 dtype=torch.float32).permute(0,3,1,2)

        #self.targets = torch.tensor(np.array(targets_h5["test_label_distributions"][:, :10]),
        #                            dtype=torch.float32)
        self.targets = torch.tensor(np.argmax(one_hot_labels[indices_in,:],1), dtype=torch.int64)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        return img, target

    def __len__(self):
        return len(self.data)
