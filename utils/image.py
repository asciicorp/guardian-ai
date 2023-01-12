from utils.object_detection import get_output_image_od
from utils.depth_estimation import get_output_image_de


def get_output_image(image, model_type, detector, params):
    time_elapsed = 0
    if model_type == "Object Detection":
        image, time_elapsed = get_output_image_od(image, detector, params)
    elif model_type == "Depth Estimation":
        image, time_elapsed = get_output_image_de(image, detector, params)
    return image, time_elapsed
