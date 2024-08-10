from django.shortcuts import render, HttpResponse
from django.conf import settings
from .forms import UploadFileForm
from .models import UploadedFile, ProcessedImage
import cv2
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from django.http import StreamingHttpResponse

# Load YOLOv8 model with your trained weights
yolo_model = YOLO(os.path.join(settings.BASE_DIR, "media_root", "best.pt"))
class_names = ['car-minor-accident', 'car-moderate-accident', 'car-serious-accident']

def detect_accidents(image, confidence_threshold=0.5):
    results = yolo_model(image)
    detected_accidents = []
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2, conf, cls = detection.xyxy[0].tolist() + [detection.conf[0].item(), int(detection.cls[0].item())]
            if conf > confidence_threshold:
                detected_accidents.append({'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 'confidence': conf, 'class_id': cls})
    return detected_accidents

def process_image(image_path, uploaded_file=None):
    img = cv2.imread(image_path)
    detected_accidents = detect_accidents(Image.open(image_path))
    accident_type = None
    for detection in detected_accidents:
        x, y, w, h = detection['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        class_id = detection['class_id']
        class_name = class_names[class_id]
        cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Determine accident type
        if class_name == 'car-serious-accident':
            accident_type = "Serious accident detected. calling fire department, ambulance, and police...."
        elif class_name == 'car-moderate-accident':
            accident_type = "Moderate accident detected. calling ambulance and police..."
        elif class_name == 'car-minor-accident':
            accident_type = "Minor accident detected. calling police ..."
    
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', 'result.jpg')
    cv2.imwrite(processed_image_path, img)

    if uploaded_file:
        # Save processed image to the database
        processed_image = ProcessedImage.objects.create(
            image='processed_images/result.jpg',
            uploaded_file=uploaded_file,
            accident_type=accident_type  # Save accident type to the database
        )
        return processed_image, accident_type
    return processed_image_path, accident_type


def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB and process it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        detected_accidents = detect_accidents(pil_image)

        # Draw bounding boxes and labels
        for detection in detected_accidents:
            x, y, w, h = detection['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            class_id = detection['class_id']
            class_name = class_names[class_id]
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in the MJPEG format
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            file_path = uploaded_file.file.path
            file_extension = Path(file_path).suffix.lower()

            if file_extension in ['.jpg', '.jpeg', '.png']:
                # Process image and get processed image and accident type
                processed_image, accident_type = process_image(file_path, uploaded_file)
                # Use the URL from the database
                result_image_url = processed_image.image.url if isinstance(processed_image, ProcessedImage) else os.path.relpath(processed_image, settings.MEDIA_ROOT)
                return render(request, 'result.html', {'result_image_url': result_image_url, 'accident_type': accident_type})
            elif file_extension in ['.mp4', '.mov', '.avi']:
                # Save the video path to session
                request.session['video_path'] = file_path
                return render(request, 'video_stream.html')
            else:
                return HttpResponse("Unsupported file type.", status=400)
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


def stream_video(request):
    video_path = request.session.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return HttpResponse("No video uploaded or file not found.", status=400)
    return StreamingHttpResponse(generate_frames(video_path),
                                content_type='multipart/x-mixed-replace; boundary=frame')
