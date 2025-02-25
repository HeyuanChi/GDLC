import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils.utils import get_device


class MissingDataset(Dataset):
    def __init__(self, bscan_dir='./data/Training_Bscan'):
        self.bscan_dir = bscan_dir
        self.num_samples = 4400

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load data
        file_path = os.path.join(self.bscan_dir, f'Bscan_{idx}.npy')
        bscan_full = np.load(file_path).astype(np.float32)
        bscan_full /= np.max(np.abs(bscan_full))  # Convert to [-1, 1]

        # Simulate missing data:
        #   5% of the columns are randomly selected
        #   set a contiguous block (random length(1-6) columns) to zero（missing)
        bscan_missing = bscan_full.copy()
        missing_rate = 0.05
        n = bscan_full.shape[1]
        missing_start = np.random.choice(n, size=int(np.ceil(n*missing_rate)), replace=False)
        for s in missing_start:
            bscan_missing[:, s: min(s + np.random.randint(6) + 1, n)] = 0.0

        # Convert to torch.Tensor
        bscan_missing = torch.from_numpy(bscan_missing).unsqueeze(0)
        bscan_full    = torch.from_numpy(bscan_full).unsqueeze(0)
        return bscan_missing, bscan_full


class FWIDataset(Dataset):
    def __init__(self, bscan_dir='./data/Training_Bscan', labels_dir='./data/Training_Labels'):
        self.bscan_dir = bscan_dir
        self.labels_dir = labels_dir
        self.num_samples = 4400

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Bscan data
        bscan_path = os.path.join(self.bscan_dir, f'Bscan_{idx}.npy')
        bscan = np.load(bscan_path).astype(np.float32)
        bscan /= np.max(np.abs(bscan))  # Convert to [-1, 1]
        
        # Labels data
        labels_path = os.path.join(self.labels_path, f'Model_{idx}.npy')
        labels = np.load(labels_path).astype(np.float32)
        labels = labels / 10 - 0.5 # Convert to [-0.5, 0.5]

        # Convert to torch.Tensor
        bscan = torch.from_numpy(bscan).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0)
        return bscan, labels


def dataloader(dataset, test_size, batch_size, shuffle):
    # use GPU
    device, kwargs = get_device()

    # Split dataset
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])

    # Get train_loader and test_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def dataloader_missing(test_size=0.1, batch_size=8, shuffle=True):
    dataset = MissingDataset()
    return dataloader(dataset, test_size, batch_size, shuffle)


def dataloader_fwi(test_size=0.1, batch_size=8, shuffle=True):
    dataset = FWIDataset()
    return dataloader(dataset, test_size, batch_size, shuffle)

