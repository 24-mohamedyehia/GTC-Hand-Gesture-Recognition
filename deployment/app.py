import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.mobilenet import MobileNetV2Gesture
from preprocessing.transforms import infer_transforms

# Configuration
IMG_SIZE = 224
MODEL_PATH = "outputs/mobilenetv2_gesture.pth"
LABELS_PATH = "labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# Initialize model
@st.cache_resource
def load_model():
    model = MobileNetV2Gesture(num_classes=num_classes, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Preprocessing
tfms = infer_transforms(IMG_SIZE)

def process_image(image):
    """Process PIL Image and return predictions"""
    tensor = tfms(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probs, 3)
        
        results = []
        for i in range(3):
            class_name = idx_to_class[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item()
            results.append((class_name, prob))
            
        return results

# Streamlit UI
st.title("✋ Hand Gesture Recognition")
st.write("Choose between uploading an image or using your webcam for real-time gesture recognition!")

# Sidebar for mode selection
mode = st.sidebar.radio("Select Mode:", ["Image Upload", "Webcam"])

# Load the model
model = load_model()

if mode == "Image Upload":
    st.subheader("📸 Image Upload Mode")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        results = process_image(image)
        
        # Display results
        st.subheader("Predictions:")
        for idx, (gesture, confidence) in enumerate(results, 1):
            st.write(f"{idx}. {gesture}: {confidence:.2%}")

else:  # Webcam mode
    st.subheader("🎥 Webcam Mode")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    if run:
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam!")
                break
                
            # Convert frame to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Get predictions
            results = process_image(pil_img)
            
            # Draw predictions on frame
            for idx, (gesture, conf) in enumerate(results):
                cv2.putText(frame_rgb, 
                           f"{gesture}: {conf:.2%}",
                           (10, 30 + idx * 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0,
                           (0, 255, 0),
                           2)
            
            # Display the frame
            FRAME_WINDOW.image(frame_rgb)
        
        # Release resources when stopped
        cap.release()