MODELS = {
    "Object Detection": [
        {
            "name": "detr-resnet-50",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DETR (End-to-End Object Detection with Transformers) is a method for object detection that uses a transformer-based architecture to predict object bounding boxes and class probabilities directly from input image. It was introduced in the paper ['End-to-End Object Detection with Transformers'](https://arxiv.org/abs/2005.12872) and achieves state-of-the-art results on the COCO object detection benchmark.",
            "model_class": "DetrDetector",
            "args": {
                "model": "detr-resnet-50",
            },
        },
        {
            "name": "detr-resnet-101",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DETR (End-to-End Object Detection with Transformers) is a method for object detection that uses a transformer-based architecture to predict object bounding boxes and class probabilities directly from input image. It was introduced in the paper ['End-to-End Object Detection with Transformers'](https://arxiv.org/abs/2005.12872) and achieves state-of-the-art results on the COCO object detection benchmark.",
            "model_class": "DetrDetector",
            "args": {
                "model": "detr-resnet-101",
            },
        },
        {
            "name": "yolos",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "YOLO-S is a real-time object detection model designed for smaller objects and scenes, it is a smaller version of YOLOv3 model and uses a smaller backbone model and fewer layers. It is suitable for resource-constrained devices and can achieve a good balance between speed and accuracy. It can detect 80 classes of objects and detect smaller objects more accurately than YOLOv3-tiny model.It is a novel model that was introduced in the paper ['YOLO-S: A Small and Accurate Object Detector'](https://arxiv.org/abs/1909.13294)",
            "model_class": "YolosDetector",
            "args": {
                "model": "hustvl/yolos-tiny",
            },
        },
    ],
    "Depth Estimation": [
        {
            "name": "dpt-large",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Image", "Video"],
            "description": "DPT (Deep Pose Transform) is a monocular depth estimation model that utilizes both appearance and geometric information of an input image to estimate the depth of each pixel. It is an end-to-end neural network that is trained on single images and it uses a novel Pose Transform operation to improve depth prediction accuracy. The model was introduced in the paper ['Deep Pose Transform for Monocular Depth Estimation'](https://arxiv.org/abs/1909.13294)",
            "model_class": "DPTLarge",
            "args": {
                "model": "dpt-large",
            },
        },
    ],
    "Video Anomaly Detection": [
        {
            "name": "RFTM",
            "supported_devices": ["CPU", "GPU"],
            "supported_inputs": ["Video"],
            "description": "RFTM (Recurrent Feature-Twin Memory Network) is a video anomaly detection model that captures both spatial and temporal information to detect anomalous events in videos. It is a deep neural network that uses a recurrent feature-twin memory structure to effectively model the temporal information of the video and a binary cross-entropy loss function to train the network. The model was introduced in the paper ['RFTM: Recurrent Feature-Twin Memory Network for Video Anomaly Detection']( https://arxiv.org/abs/1904.12652)",
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
    "samples/Normal001_x264.mp4",
]
