from utils.object_detection import get_output_video_od
from utils.depth_estimation import get_output_video_de
from utils.video_anomaly_detection import get_output_video_vad


def get_output_video(video, model_type, detector, params):
    if model_type == "Object Detection":
        return get_output_video_od(video, detector, params)
    if model_type == "Depth Estimation":
        return get_output_video_de(video, detector, params)
    if model_type == "Video Anomaly Detection":
        return get_output_video_vad(video, detector, params)
