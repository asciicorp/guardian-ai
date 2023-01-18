import os
import sys

import numpy as np
from openvino.runtime import Core, PartialShape

SCRIPT_DIR = os.path.dirname(os.path.abspath("models/object_detection/yolo/v7"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class YoloV7:
  def __init__(self, device=None, model="yolo-v7"):
    ie = Core()
    model_xml = f"{SCRIPT_DIR}/v7/yolov7.xml"
    self.model = ie.read_model(model=model_xml)
    self.compiled_model = ie.compile_model(model=self.model, device_name="CPU")

    self.input_layer = self.compiled_model.input(0)
    self.output_layer = self.compiled_model.output(0)
    self.N, self.C, self.H, self.W = self.input_layer.shape

  def detect_batch(self, images, threshold=0.5):
    inputs = self._resize_images(images)
    # Convert to NCHW format
    input_data = [np.expand_dims(np.transpose(input, (2, 0, 1)), 0).astype(np.float32) for input in inputs]

    results = np.array([self.compiled_model(input)[self.output_layer] for input in input_data]).squeeze()

    batch_detections = []
    for i, image in enumerate(input_data):
      detections = []
      for score, label, box in zip(
          results[i]["scores"], results[i]["labels"], results[i]["boxes"]
      ):
          box = [round(i, 2) for i in box.tolist()]
          labelname = self.model.config.id2label[label.item()]
          score = round(score.item(), 3)
          detections.append({"label": labelname, "score": score, "box": box})
      batch_detections.append(detections)

    del inputs, outputs, results
    return batch_detections

    #formatted = (output * 255 / np.max(output)).astype("uint8")
    #return formatted

  def _resize_images(self, images):
    inputs = [np.asarray(image.resize((self.W, self.H))) for image in images]
    return inputs

  def get_labels(self):
    labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
       ]
    return labels