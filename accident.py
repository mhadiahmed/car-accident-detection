import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models import resnet50
from PIL import Image
from ultralytics import YOLO

# Load YOLOv5u model
yolo_model = YOLO("yolov5su.pt")  # Use YOLOv5u for improved performance

# Load a pre-trained classification model (e.g., ResNet)
# Updated to use the 'weights' parameter
model = resnet50(weights='DEFAULT')
model.eval()

# Define the transformation for the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_accidents(image_path):
    img = Image.open(image_path)
    results = yolo_model(img)

    detected_accidents = []

    for result in results:
        # Iterate through detected objects
        for detection in result.boxes:
            # Extract bounding box coordinates and class information
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            detected_accidents.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return detected_accidents

def classify_accident(image_path):
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    category = torch.argmax(probabilities).item()
    
    return category

def process_image(image_path):
    detected_accidents = detect_accidents(image_path)
    
    for i, (x, y, w, h) in enumerate(detected_accidents):
        # Crop and save the detected accident region for classification
        img = Image.open(image_path)
        accident_region = img.crop((x, y, x + w, y + h))
        accident_image_path = f"accident_{i}.jpg"
        accident_region.save(accident_image_path)
        
        # Classify the cropped accident region
        category = classify_accident(accident_image_path)
        if category == 0:
            print("Serious accident detected")
            # Notify fire department, ambulance, and police
        elif category == 1:
            print("Moderate accident detected")
            # Notify ambulance and police
        else:
            print("Minor accident detected")
            # Notify police only

# Example usage
process_image("dataset-card.jpg")
