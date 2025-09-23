import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                      average: str = 'weighted') -> Dict[str, float]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def calculate_per_class_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                                class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(prec))]

    return {
        name: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
        for i, name in enumerate(class_names)
    }


def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor,
                          class_names: List[str],
                          normalize: bool = True,
                          save_path: Optional[str] = None):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        if y_pred.dim() > 1:
            y_pred = torch.argmax(y_pred, dim=1)
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, None]
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        fmt, title = "d", "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class MetricsTracker:
    def __init__(self):
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def update_train(self, loss: float, acc: float):
        self.train_losses.append(loss)
        self.train_accs.append(acc)

    def update_val(self, loss: float, acc: float):
        self.val_losses.append(loss)
        self.val_accs.append(acc)

    def plot_curves(self, save_path: Optional[str] = None):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, "b-", label="Train Loss")
        plt.plot(epochs, self.val_losses, "r-", label="Val Loss")
        plt.legend(); plt.title("Loss"); plt.xlabel("Epoch")

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, "b-", label="Train Acc")
        plt.plot(epochs, self.val_accs, "r-", label="Val Acc")
        plt.legend(); plt.title("Accuracy"); plt.xlabel("Epoch")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()