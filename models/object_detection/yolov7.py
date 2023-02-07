import os
import sys
import torch
import numpy as np
from openvino.runtime import Core, PartialShape
from utils.plots import plot_one_box
from utils.general import scale_coords, non_max_suppression
from PIL import Image
import cv2

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

  def preprocess_image(self, img0:np.ndarray):
    img = self.letterbox(img0, auto=False)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

  def letterbox(self, img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]
    w, h = size


    r = min(h / shape[0], w / shape[1])
    if not scaleup: 
        r = min(r, 1.0)


    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  
    if auto:  
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill: 
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0] 

    dw /= 2 
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

  def prepare_input_tensor(self, image: np.ndarray):
    input_tensor = image.astype(np.float32)  # uint8 to fp16/32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

  def get_labels(self):
    return ["person", "car"]

  def detect_batch(self, img, threshold):
    output_blob = self.compiled_model.output(0)
    images = np.array([np.array(img[i]) for i in range(len(img))] )
    print(images.shape)
    preprocessed_img, orig_img = self.preprocess_image(images)
    input_tensor = self.prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(self.model(input_tensor)[output_blob])
    pred = non_max_suppression(predictions, threshold)
    return pred, orig_img, input_tensor.shape  
