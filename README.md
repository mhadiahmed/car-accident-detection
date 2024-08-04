
### 1. **Dataset Preparation**

#### A. **Collecting Images**

- **Capture or Download Images:** Gather a variety of images that represent the different types of accidents you want to classify (e.g., serious, moderate, minor).

#### B. **Annotating Images**

- **Annotation Tools:**
  - **LabelImg:** A graphical image annotation tool.
    - Install with:
      ```bash
      pip install labelImg
      ```
    - Run with:
      ```bash
      labelImg
      ```
    - Use it to draw bounding boxes around accidents and label them.

  - **Roboflow:** An online tool for labeling and dataset management.
    - Create a project, upload images, and label them directly on the website.

- **Annotation Format:**
  - YOLO format requires a `.txt` file per image with the same name, containing bounding box annotations in the format:
    ```
    class_id x_center y_center width height
    ```
    - `class_id`: Numeric ID of the class (0 for serious, 1 for moderate, 2 for minor).
    - `x_center, y_center`: Center of the bounding box (normalized to [0, 1]).
    - `width, height`: Size of the bounding box (normalized to [0, 1]).

#### C. **Organizing Your Dataset**

- **Directory Structure:**
  ```
  /dataset
    /images
      /train
        img1.jpg
        img2.jpg
        ...
      /val
        img1.jpg
        img2.jpg
        ...
    /labels
      /train
        img1.txt
        img2.txt
        ...
      /val
        img1.txt
        img2.txt
        ...
  ```

### 2. **Configuring YOLO Training**

#### A. **Creating a YAML Configuration File**

- Create a YAML file (`config.yaml`) to specify your dataset paths and class names:
  ```yaml
  train: ./dataset/images/train
  val: ./dataset/images/val

  nc: 3  # Number of classes
  names: ['serious', 'moderate', 'minor']  # Class names
  ```

#### B. **Choosing a Pre-trained Model**

- **Pre-trained Models:**
  - You can use a pre-trained YOLO model for transfer learning. For example, `yolov5s.pt` is a small YOLOv5 model. The model name can be changed to `yolov5su.pt` if you are using a YOLOv5u model.
  
### 3. **Training Your Model**

#### A. **Install Ultralytics YOLO**

- Ensure you have the Ultralytics YOLO package installed:
  ```bash
  pip install ultralytics
  ```

#### B. **Run Training Command**

- Execute the training command in your terminal:
  ```bash
  yolo train data=config.yaml model=yolov5s.pt epochs=50 imgsz=640
  ```
  - `data=config.yaml`: Path to your YAML configuration file.
  - `model=yolov5s.pt`: Path to a pre-trained model. Replace with your specific model if necessary.
  - `epochs=50`: Number of epochs for training. Adjust based on your needs.
  - `imgsz=640`: Size of images used for training. Choose based on your GPU memory.

### 4. **Evaluating and Using Your Model**

#### A. **Evaluate Your Model**

- **Check Training Output:**
  - After training, check the results and logs. The model's performance is typically saved in a `runs` directory within the project folder.
  - Evaluate metrics like Precision, Recall, and mAP (mean Average Precision) to assess the model’s performance.

#### B. **Inference with Your Trained Model**

- **Run Inference:**
  - Use the trained model to make predictions on new images:
    ```python
    from ultralytics import YOLO

    # Load the trained model
    model = YOLO("path/to/your/trained/model.pt")

    # Perform inference
    results = model("path/to/your/image.jpg")

    # Print or visualize results
    results.show()  # Display results
    results.save()  # Save results to file
    ```

#### C. **Adjust and Fine-Tune**

- If the results are not satisfactory, consider:
  - **More Data:** Collect more images or better-quality annotations.
  - **Hyperparameter Tuning:** Adjust learning rates, batch sizes, etc.
  - **Model Selection:** Try different YOLO architectures or pretrained models.

### Additional Tips

1. **Hardware Requirements:**
   - Ensure you have a suitable GPU for training. YOLO training can be resource-intensive.

2. **Documentation:**
   - Check the [Ultralytics Documentation](https://docs.ultralytics.com/) for additional details on training parameters and advanced configuration.

3. **Community Support:**
   - Engage with the YOLO community or forums if you encounter specific issues.

how to run this project on your local:

make sure you have python installed the latest version 3.10 if not download it from here python.org
make sure to make a virtual environment using this command 

```
  python -m venv venv
```

active the virtual env on your terminal or cmd using the following command

```
# for windows
venv\Scripts\activate

# for linux & mac
source venv/bin/activate
```

then install the requirements:

```
pip install -r requirements.txt
```

you are good to go run one of the script's by this command


```
python accidentv2.py
```