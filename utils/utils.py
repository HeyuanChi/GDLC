import numpy as np

import torch


def get_device():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda or use_mps else {}
    return device, kwargs

def predict(model, data_loader):
    device, kwargs = get_device()

    model.to(device)
    model.eval()

    all_inputs = []
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            outputs = model(inputs)

            all_inputs.append(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_inputs, all_outputs, all_targets


def calc_travel_time_2d(eps_map, dz=0.1, c0=3e8):
    H, W = eps_map.shape
    T_out = np.zeros((W,), dtype=np.float32)
    for x in range(W):
        col_eps = eps_map[:, x]
        col_sqrt = np.sqrt(np.clip(col_eps, 1e-6, None))
        val = col_sqrt.sum() * (dz/c0)
        T_out[x] = val
    return T_out


def predict_bagging(models, data_loader):
    all_outputs = []
    for model in models:
        inputs, outputs, targets = predict(model, data_loader)
        all_outputs.append(outputs.cpu().numpy())
    return inputs, all_outputs, targets