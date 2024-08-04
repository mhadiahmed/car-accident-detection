import cv2
import numpy as np
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Load YOLOv5u model
yolo_model = YOLO("yolov5su.pt")  # Use YOLOv5u for improved performance

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

def detect_accidents(image):
    results = yolo_model(image)

    detected_accidents = []

    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()
            detected_accidents.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return detected_accidents

def classify_accident(image):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    category = torch.argmax(probabilities).item()
    
    return category

def process_frame(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detected_accidents = detect_accidents(img_pil)
    
    for x, y, w, h in detected_accidents:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        accident_region = frame[y:y+h, x:x+w]
        accident_image_pil = Image.fromarray(cv2.cvtColor(accident_region, cv2.COLOR_BGR2RGB))
        category = classify_accident(accident_image_pil)
        
        if category == 0:
            print("Serious accident detected")
        elif category == 1:
            print("Moderate accident detected")
        else:
            print("Minor accident detected")

    cv2.imshow("Detected Accidents", frame)

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam or replace with video file path

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        process_frame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
