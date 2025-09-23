import cv2
import torch
from PIL import Image
import json
from models.mobilenet import MobileNetV2Gesture
from preprocessing.transforms import infer_transforms  # <-- import once

# -------------------------
# Config
# -------------------------
IMG_SIZE = 224
MODEL_PATH = "outputs/mobilenetv2_gesture.pth"
LABELS_PATH = "labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load class labels
# -------------------------
with open(LABELS_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# -------------------------
# Load model
# -------------------------
model = MobileNetV2Gesture(num_classes=num_classes, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# -------------------------
# Preprocessing (imported)
# -------------------------
tfms = infer_transforms(IMG_SIZE)   # define transform once

# -------------------------
# Real-time loop
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("✅ Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame → PIL → tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    tensor = tfms(pil_img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        gesture = idx_to_class[pred.item()]
        confidence = conf.item()

    # Overlay prediction
    cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show window
    cv2.imshow("Hand Gesture Recognition", frame)

    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
