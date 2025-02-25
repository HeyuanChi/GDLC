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
    
    loss_log = open("./results/unet_fwi/loss.csv", "w")
    loss_log.write("epoch,train_loss,val_loss\n")
    
    epoch_bar = tqdm(range(1, num_epochs + 1), desc='Epochs', leave=True)
    for i, epoch in enumerate(epoch_bar):
        model.train()
        train_loss = 0.0

        if i == 55:
            optimizer.load_state_dict
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