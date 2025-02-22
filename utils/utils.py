import numpy as np

import torch


def get_device():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda or use_mps else {}
    return device, kwargs