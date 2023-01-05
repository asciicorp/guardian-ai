import time

from utils import draw_bboxes, get_output_video, get_device, get_object_detector
model = "DETR"
device = "cpu"
object_detector = get_object_detector(model, device)

if object_detector is not None:
  batch_size = 4
  labels = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
  threshold = 1.0
  fps = 1
  video_start_time = time.time()
  get_output_video(
      video="video.mp4",
      detector=object_detector,
      labels=labels,
      threshold=threshold,
      fps=fps,
      batch_size=batch_size,
  )
  video_end_time = time.time()