import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Tuple, Optional, Any


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device,
                epoch: int = 1, total_epochs: int = 1,
                log_interval: int = 50) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(output, 1)
        correct += (preds == target).sum().item()
        total += target.size(0)

        # 🔹 Print every `log_interval` batches
        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            batch_acc = correct / total * 100
            print(f"[Epoch {epoch}/{total_epochs}] "
                  f"Batch {batch_idx}/{len(train_loader)} "
                  f"Loss: {running_loss/batch_idx:.4f}, Acc: {batch_acc:.2f}%")

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy



def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✅ Model saved to {filepath}")


def load_checkpoint(model: nn.Module, filepath: str, device: torch.device):
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"✅ Model loaded from {filepath}")
    return model


def get_optimizer(model: nn.Module, optimizer_name: str = 'adam', lr: float = 0.001,
                  weight_decay: float = 1e-4) -> optim.Optimizer:
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
