MODELS = {
    "Object Detection": [
        {
            "name": "detr-resnet-50",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DETR ResNet-50",
            "model_class": "DetrDetector",
            "args": {
                "model": "detr-resnet-50",
            },
        },
        {
            "name": "detr-resnet-101",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DETR ResNet-101",
            "model_class": "DetrDetector",
            "args": {
                "model": "detr-resnet-101",
            },
        },
        {
            "name": "detr-resnet-50 (CPU Optimized)",
            "supported_devices": ["CPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DETR ResNet-50 (CPU Optimized)",
            "args": {},
        },
        {
            "name": "yolos-model",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "Yolos Model",
            "model_class": "YolosDetector",
            "args": {
                "model": "hustvl/yolos-tiny",
            },
        },
        {
            "name": "yolo-v7",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "YoloV7 Model",
            "model_class": "YoloV7",
            "args": {
                "model": "yolo-v7",
            },
        },
    ],
    "Depth Estimation": [
        {
            "name": "dpt-large",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DPT Large",
            "model_class": "DPTLarge",
            "args": {
                "model": "dpt-large",
            },
        },
        {
            "name": "midas",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image"],
            "description": "MiDaS",
            "model_class": "MiDaS",
            "args": {
                "model": "midas",
            },
        },
    ],
    "Video Anomaly Detection": [
        {
            "name": "RFTM",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Video"],
            "description": "RFTM",
            "model_class": "RFTM",
            "args": {},
        }
    ],
}

MODEL_TYPES = [None, *MODELS.keys()]
DEVICE_TYPES = [None, "CPU", "GPU"]

OBJECT_DETECTION_MODELS = [
    None,
    *[model["name"] for model in MODELS["Object Detection"]],
]

DEPTH_ESTIMATION_MODELS = [
    None,
    *[model["name"] for model in MODELS["Depth Estimation"]],
]

VIDEO_ANOMALY_DETECTION_MODELS = [
    None,
    *[model["name"] for model in MODELS["Video Anomaly Detection"]],
]


def get_model_name(model_type):
    if model_type is None:
        return [None]
    model_names = [None, *[model["name"] for model in MODELS[model_type]]]
    return model_names


APP_STYLES = """
    <style>
        .css-1lsmgbg {
            visibility: hidden;
        }
        .css-1rs6os {
            visibility: hidden;
        }
    </style>
"""

SAMPLE_IMAGES = [
    None,
    "samples/burglary_1.jpg",
    "samples/dancing_outside_cctv.webp",
    "samples/indoor_cctv.png",
    "samples/person_street_walking.webp",
    "samples/small_image_person.jpg",
]

SAMPLE_VIDEOS = [
    None,
    "samples/Arrest001_x264.mp4",
    "samples/Abuse001_x264.mp4",
    "samples/Arson001_x264.mp4",
    "samples/Assault001_x264.mp4",
    "samples/Burglary001_x264.mp4",
]
