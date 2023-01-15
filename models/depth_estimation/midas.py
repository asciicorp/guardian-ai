import os
import sys

import numpy as np
from openvino.runtime import Core, PartialShape

SCRIPT_DIR = os.path.dirname(os.path.abspath("models/depth_estimation/midas"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class MiDaS:
  def __init__(self, device=None, model="midas"):
    ie = Core()
    model_xml = f"{SCRIPT_DIR}/midas/openvino_midas_v21_small_256.xml"
    model = ie.read_model(model=model_xml)
    self.compiled_model = ie.compile_model(model=model, device_name="CPU")

    self.input_layer = self.compiled_model.input(0)
    self.output_layer = self.compiled_model.output(0)
    self.N, self.C, self.H, self.W = self.input_layer.shape

  def estimate_batch(self, images):
    inputs = self._resize_images(images)
    # Convert to NCHW format
    input_data = [np.expand_dims(np.transpose(input, (2, 0, 1)), 0).astype(np.float32) for input in inputs]

    output = np.array([self.compiled_model(input)[self.output_layer] for input in input_data]).squeeze()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return formatted

  def _resize_images(self, images):
    inputs = [np.asarray(image.resize((self.W, self.H))) for image in images]
    return inputs