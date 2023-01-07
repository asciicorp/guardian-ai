from utils.object_detection import get_output_video_od

def get_output_video(video, model_type, detector, params):
    time_elapsed = 0
    if model_type == "Object Detection":
        time_elapsed = get_output_video_od(video, detector, params)
    return time_elapsed