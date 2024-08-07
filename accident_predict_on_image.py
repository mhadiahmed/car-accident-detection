import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model with your trained weights
yolo_model = YOLO("car-accident.v1i.yolov8/runs/detect/train5/weights/best.pt")

# Define the class names as per your training
class_names = ['car-minor-accident', 'car-moderate-accident', 'car-serious-accident']

def detect_accidents(image_path, confidence_threshold=0.5):
    img = Image.open(image_path)
    results = yolo_model(img)

    detected_accidents = []

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2, conf, cls = detection.xyxy[0].tolist() + [detection.conf[0].item(), int(detection.cls[0].item())]
            if conf > confidence_threshold:
                detected_accidents.append({
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'confidence': conf,
                    'class_id': cls
                })

    return detected_accidents

def process_image(image_path):
    img = cv2.imread(image_path)  # Open the image with OpenCV
    detected_accidents = detect_accidents(image_path)
    
    for detection in detected_accidents:
        x, y, w, h = detection['bbox']
        # Draw rectangle around the detected accident
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get the class name from class ID
        class_id = detection['class_id']
        class_name = class_names[class_id]
        
        # Put the class name on the image
        cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Apply the logic for notification based on the classification
        if class_name == 'car-serious-accident':
            print("Serious accident detected")
            # Notify fire department, ambulance, and police
        elif class_name == 'car-moderate-accident':
            print("Moderate accident detected")
            # Notify ambulance and police
        elif class_name == 'car-minor-accident':
            print("Minor accident detected")
            # Notify police only
        else:
            print("Unknown category detected")

    # Display the image with rectangles and text
    cv2.imshow("Detected Accidents", img)
    # Wait for 5 seconds and then close the window
    cv2.waitKey(5000)  # Adjust the timeout as needed
    cv2.destroyAllWindows()

# Example usage
process_image("dataset-card.jpg")
