import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.stopper import EarlyStopper
from utils.utils import get_device

# use GPU
device, kwargs = get_device()


def train_unet_fill(model, train_loader, test_loader, lr=1e-4, num_epochs=100, patience=30, tol=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    early_stopper = EarlyStopper(patience=patience, min_delta=tol)

    if not os.path.exists('./results/unet_fill'):
        os.mkdir('./results/unet_fill')
    loss_log = open("./results/unet_fill/loss.csv", "w")
    loss_log.write("epoch,train_loss,val_loss\n")
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs', leave=True)
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        
        # Update tqdm bar with current losses
        epoch_bar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Val Loss': f'{val_loss:.8f}'})
        
        # Log the losses
        loss_log.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        loss_log.flush()

        # Check for early stopping.
        stop = early_stopper.early_stop(val_loss, model)
        # Save the best model parameters
        torch.save(early_stopper.best_model_state, f"./results/unet_fill/model.pt")
        if stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {early_stopper.min_validation_loss:.8f}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    loss_log.close()


def train_unet_fwi(model, train_loader, test_loader, lr=5e-4, num_epochs=100, patience=30, tol=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda epoch: max(0.97 ** epoch, 0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.L1Loss()
    
    early_stopper = EarlyStopper(patience=patience, min_delta=tol)

    if not os.path.exists('./results/unet_fwi'):
        os.mkdir('./results/unet_fwi')
    loss_log = open("./results/unet_fwi/loss.csv", "w")
    loss_log.write("epoch,train_loss,val_loss\n")
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs', leave=True)
    for i, epoch in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        
        # Update tqdm bar with current losses
        epoch_bar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Val Loss': f'{val_loss:.8f}'})
        
        # Log the losses
        loss_log.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        loss_log.flush()

        # Check for early stopping.
        stop = early_stopper.early_stop(val_loss, model)
        # Save the best model parameters
        torch.save(early_stopper.best_model_state, f"./results/unet_fwi/model.pt")
        if stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {early_stopper.min_validation_loss:.8f}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    loss_log.close()


def calc_travel_time_tensor(eps_pred, dz=0.1, c0=3e8):
    """
    eps_pred: shape [B,1,224,224], scale=[-0.5,0.5]
    Convert to [0,10]? => eps= (pred+0.5)*10
    Then do vertical sum => shape [B,224]
    """
    # 1) map from [-0.5,0.5] to [0,10]
    eps_mapped = (eps_pred + 0.5)*10.0
    # 2) clamp >0
    eps_mapped = torch.clamp(eps_mapped, min=1e-6)
    # 3) sqrt & sum
    sqrt_ = torch.sqrt(eps_mapped)
    # sum over dim=2 => depth
    # shape => [B,1,W=224]
    sum_ = sqrt_.sum(dim=2, keepdim=True)  # => [B,1,1,W]
    T_pred = sum_.squeeze(2)* (dz/c0)      # => [B,W=224]
    return T_pred


def train_unet_fwi(model, train_loader, test_loader, 
                   lr=5e-4, num_epochs=100, 
                   patience=30, tol=1e-4,
                   gamma=0.01,  # new param: weight for travel-time loss
                   dz=0.1, c0=3e8):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_lambda = lambda epoch: max(0.97 ** epoch, 0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.L1Loss()
    
    early_stopper = EarlyStopper(patience=patience, min_delta=tol)

    if not os.path.exists('./results/unet_fwi'):
        os.mkdir('./results/unet_fwi')
    loss_log = open("./results/unet_fwi/loss.csv", "w")
    loss_log.write("epoch,train_loss,val_loss\n")
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs', leave=True)
    for i, epoch in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0

        for batch_data in train_loader:
            # batch_data could be (inputs, targets) or (inputs, targets, Tobs)
            if len(batch_data) == 2:
                inputs, targets = batch_data
                Tobs = None
            else:
                inputs, targets, Tobs = batch_data
            
            inputs  = inputs.to(device).float()
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_sup = criterion(outputs, targets)  # original supervised L1

            # If Tobs present => do travel-time loss
            if Tobs is not None:
                Tobs = Tobs.to(device).float()  # shape [B,224]
                # compute T_pred from outputs
                T_pred = calc_travel_time_tensor(outputs, dz=dz, c0=c0)  # => [B,224]
                loss_tt = criterion(T_pred, Tobs)
                loss = loss_sup + gamma*loss_tt
            else:
                loss = loss_sup

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data)==2:
                    inputs, targets = batch_data
                    Tobs = None
                else:
                    inputs, targets, Tobs = batch_data

                inputs  = inputs.to(device).float()
                targets = targets.to(device).float()
                outputs = model(inputs)
                loss_sup = criterion(outputs, targets)

                if Tobs is not None:
                    Tobs = Tobs.to(device).float()
                    T_pred = calc_travel_time_tensor(outputs, dz=dz, c0=c0)
                    loss_tt = criterion(T_pred, Tobs)
                    loss_val = loss_sup + gamma*loss_tt
                else:
                    loss_val = loss_sup

                val_loss += loss_val.item() * inputs.size(0)
        val_loss /= len(test_loader.dataset)
        
        epoch_bar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Val Loss': f'{val_loss:.8f}'})
        loss_log.write(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        loss_log.flush()

        # early stopping
        stop = early_stopper.early_stop(val_loss, model)
        torch.save(early_stopper.best_model_state, f"./results/unet_fwi/model.pt")
        if stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {early_stopper.min_validation_loss:.8f}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    loss_log.close()
