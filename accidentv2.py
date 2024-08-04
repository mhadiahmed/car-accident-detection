import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from ultralytics import YOLO

# Load YOLOv5u model
yolo_model = YOLO("yolov5su.pt")  # Ensure the correct YOLO model is used

# Load a pre-trained classification model (e.g., ResNet)
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
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            detected_accidents.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return detected_accidents

def classify_accident(image):
    # Convert the PIL image to a tensor for classification
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    category = torch.argmax(probabilities).item()

    return category

def process_image(image_path):
    img = cv2.imread(image_path)  # Open the image with OpenCV
    detected_accidents = detect_accidents(image_path)
    
    for x, y, w, h in detected_accidents:
        # Draw rectangle around the detected accident
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Crop and classify the detected region
        accident_region = img[y:y+h, x:x+w]
        accident_image_pil = Image.fromarray(cv2.cvtColor(accident_region, cv2.COLOR_BGR2RGB))
        category = classify_accident(accident_image_pil)
        
        # Apply the logic for notification based on the classification
        if category == 0:
            print("Serious accident detected")
            # Notify fire department, ambulance, and police
        elif category == 1:
            print("Moderate accident detected")
            # Notify ambulance and police
        elif category == 2:
            print("Minor accident detected")
            # Notify police only
        else:
            print("Unknown category detected")

    # Display the image with rectangles
    cv2.imshow("Detected Accidents", img)
    # Wait for 5 seconds and then close the window
    cv2.waitKey(5000)  # Adjust the timeout as needed
    cv2.destroyAllWindows()

# Example usage
process_image("img.jpg")
