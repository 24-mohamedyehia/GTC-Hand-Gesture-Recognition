from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.mobilenet import MobileNetV2Gesture
from preprocessing.dataloader import get_dataloaders
from utils.train_utils import train_epoch, validate_epoch, get_optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Config
# -------------------------
EPOCHS = 5
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 1e-3
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Data
# -------------------------
print("📂 Loading data...")

train_loader, val_loader, class_names = get_dataloaders(
    data_dir="data",
    batch_size=BATCH_SIZE,
    num_workers=4,
    img_size=IMG_SIZE,
    augment=True,
)
print("Classes:", class_names)

# -------------------------
# Model
# -------------------------
model = MobileNetV2Gesture(num_classes=len(class_names), pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(model, optimizer_name="adam", lr=LR)

print("🚀 Training in progress...")

# -------------------------
# Training loop
# -------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
checkpoint_path = OUTPUT_DIR / "mobilenetv2_gesture.pth"

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion ,optimizer, device,epoch=epoch, total_epochs=EPOCHS,log_interval=100)
    val_loss, val_acc     = validate_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train: {train_loss:.4f}/{train_acc:.3f} | "
          f"Val: {val_loss:.4f}/{val_acc:.3f}")

    # Save history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_names": class_names,
        }, checkpoint_path)
        print(f"💾 Saved best model so far → {checkpoint_path}")

# -------------------------
# Save training results
# -------------------------
# Save history as .pt and .csv
torch.save(history, OUTPUT_DIR / "training_history.pt")

import pandas as pd
pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

# Plot curves
plt.figure(figsize=(10,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(OUTPUT_DIR / "loss_curve.png"); plt.close()

plt.figure(figsize=(10,5))
plt.plot(history["train_acc"], label="Train Acc")
plt.plot(history["val_acc"], label="Val Acc")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.savefig(OUTPUT_DIR / "accuracy_curve.png"); plt.close()

print(f"📊 Training results saved in: {OUTPUT_DIR}")
