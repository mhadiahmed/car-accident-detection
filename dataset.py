!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="kgZ5INeazeQDKYq0YdGq")
project = rf.workspace("mhadi-ahmed").project("car-accident-opikb")
version = project.version(1)
dataset = version.download("yolov8")

