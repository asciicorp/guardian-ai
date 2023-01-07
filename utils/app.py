import torch
import streamlit as st
import importlib

from utils.constants import MODELS, SAMPLE_IMAGES, SAMPLE_VIDEOS


def get_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "CPU":
        return torch.device("cpu")
    elif device == "GPU" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        st.sidebar.error("GPU not available")
        return torch.device("cpu")


def get_model(model_type, model_name, device_type):
    models = MODELS[model_type]
    model_info = [model for model in models if model["name"] == model_name][0]
    if device_type not in model_info["supported_devices"]:
        st.sidebar.error("Model not supported on selected device")
        return None, None
    device = get_device(device_type)
    if device.type == "cpu":
        st.warning("Running on CPU")
    else:
        st.success("Running on GPU")
    model = getattr(
        importlib.import_module("models.object_detection"), model_info["model_class"]
    )(device=device, **model_info["args"])
    return model, model_info


def get_controls(model, input_mode):
    labels = st.multiselect(
        "Select labels",
        model.get_labels() if model is not None else ["person"],
        ["person"],
    )
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.1)
    if input_mode == "Video":
        batch_size = st.slider("Batch size", 1, 16, 4, 1)
        fps = st.slider("FPS", 1, 30, 1, 1)
    else:
        batch_size = 1
        fps = 1
    return {
        "labels": labels,
        "threshold": threshold,
        "batch_size": batch_size,
        "fps": fps,
    }

def get_image_inputs():
    uploaded_image = st.sidebar.file_uploader(
            "Upload an image", type=["jpg", "png", "jpeg"]
        )
    image = st.sidebar.selectbox("Select an image", SAMPLE_IMAGES)
    return uploaded_image, image

def get_video_inputs():
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    if uploaded_video is not None:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
    video = st.sidebar.selectbox("Select a video", SAMPLE_VIDEOS)
    return uploaded_video, video