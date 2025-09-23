import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from models.mobilenet import MobileNetV2Gesture
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from preprocessing.transforms import val_transforms

# Config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "data/val"
IMG_SIZE = 224
BATCH = 32
MODEL_PATH = "outputs/mobilenetv2_gesture.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# Data
print("📂 Loading dataset...")
test_ds = datasets.ImageFolder(DATA_DIR, transform=val_transforms(IMG_SIZE))
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
class_names = test_ds.classes

print(f"Dataset ready: {len(test_ds)} samples, {len(class_names)} classes")

# Model
print(f"📦 Loading model from {MODEL_PATH} ...")
model = MobileNetV2Gesture(num_classes=len(class_names), pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded and ready for evaluation")

# Evaluation
print("Starting evaluation...")
criterion = nn.CrossEntropyLoss()
test_loss, correct, total = 0, 0, 0
all_preds, all_labels = [], []


with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

test_loss /= len(test_loader)
test_acc = correct / total

print(f"\nEvaluation Results:")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")

# Classification report
print("\n📝 Detailed Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# save classification report
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("outputs/classification_report.csv", index=True)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n🔎 Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True,
    xticklabels=class_names, yticklabels=class_names,
    linewidths=.5, square=True
)

plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.tight_layout()

# Save confusion matrix
save_path = "outputs/confusion_matrix.png"
plt.savefig(save_path)
plt.close()


