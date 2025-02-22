import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from utils.utils import get_device


class MissingDataset(Dataset):
    def __init__(self, root_dir='./data/Training_Bscan'):
        self.root_dir = root_dir
        self.num_samples = 4400  # Bscan_0.npy ~ Bscan_4399.npy

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load data
        file_path = os.path.join(self.root_dir, f'Bscan_{idx}.npy')
        bscan_full = np.load(file_path).astype(np.float32)
        bscan_full /= np.max(np.abs(bscan_full))

        # Generate missing data
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


def dataloader_missing(test_size=0.1, batch_size=8, shuffle=True):
    # use GPU
    device, kwargs = get_device()

    # Load dataset
    dataset = MissingDataset()

    # Split dataset
    total_size = len(dataset)
    test_num   = int(total_size * test_size)
    train_num  = total_size - test_num
    train_dataset, test_dataset = random_split(dataset, [1-test_size, test_size])

    # Get train_loader and test_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

