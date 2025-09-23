import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import json
from preprocessing.transforms import infer_transforms

from models.mobilenet import MobileNetV2Gesture

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_SIZE = 224
MODEL_PATH = "outputs/mobilenetv2_gesture.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names from labels.json
with open("labels.json", "r") as f:
    class_to_idx = json.load(f)
    
# Create idx to class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# -------------------------
# Data preprocessing
# -------------------------

tfms = infer_transforms(img_size=224)  # use same normalization as training

# -------------------------
# Model
# -------------------------
model = MobileNetV2Gesture(num_classes=num_classes, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def predict_image(image_path):

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = tfms(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        predicted_class = idx_to_class[predicted.item()]
        confidence = probabilities[0][predicted].item()
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        print(f"Image: {image_path}")
        print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
        print("Top 3 predictions:")
        for i in range(3):
            class_name = idx_to_class[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item()
            print(f"  {i+1}. {class_name}: {prob:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predict_image(image_path)