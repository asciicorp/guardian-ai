# create a streamlit app where there is a sidebar where we can select the tranformer detr model and select a video to run the model on
# the video will be displayed on the main page and the output will be displayed on the main page
import streamlit as st
from PIL import Image
import time
import torch

from models.object_detection import DetrDetector
from utils import draw_bboxes, get_output_video

# create a sidebar
st.sidebar.title("Guardian AI")
# device selection
device = st.sidebar.selectbox("Select the device", [None, "CPU", "GPU"])
if device == "CPU":
    device = torch.device("cpu")
elif device == "GPU" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    st.sidebar.error("GPU not available")
    device = torch.device("cpu")

# model selection
model = st.sidebar.selectbox("Select a Object Detection model", [None, "DETR"])
object_detector = None
if model == "DETR":
    object_detector = DetrDetector(device=device)

labels = st.sidebar.multiselect(
    "Select labels",
    ["person", "car", "bicycle", "motorcycle", "bus", "truck"],
    ["person"],
)
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.1)

# load an image
image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
# display the image in the sidebar
if image is not None:
    image = Image.open(image)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    if object_detector is not None:
        st.subheader("Object Detection Output Image")
        with st.spinner("Detecting objects..."):
            image_start_time = time.time()
            outputs = object_detector.detect(image, threshold)
            outputs = [out for out in outputs if out["label"] in labels]
            output_image = draw_bboxes(image, outputs)
            image_end_time = time.time()
        st.image(output_image, caption="Output Image", use_column_width=True)
        st.write(f"**Inference time:** {image_end_time - image_start_time:.3f} seconds")

# #load a video
uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())
video = st.sidebar.selectbox(
    "Select a video",
    [
        None,
        "samples/Arrest001_x264.mp4",
        "samples/Abuse001_x264.mp4",
        "samples/Arson001_x264.mp4",
        "samples/Assault001_x264.mp4",
        "samples/Burglary001_x264.mp4",
    ],
)
if video is not None or uploaded_video is not None:
    current_video = video
    if uploaded_video is not None:
        current_video = "uploaded_video.mp4"
    st.sidebar.video(current_video)
    batch_size = st.sidebar.slider("Batch size", 1, 16, 4, 1)
    fps = st.sidebar.slider("FPS", 1, 30, 1, 1)
    if object_detector is not None:
        st.subheader("Object Detection Output Video")
        with st.spinner("Detecting objects..."):
            video_start_time = time.time()
            get_output_video(
                video=current_video,
                detector=object_detector,
                labels=labels,
                threshold=threshold,
                fps=fps,
                batch_size=batch_size,
            )
            video_end_time = time.time()
        st.video("output.mp4")
        st.write(f"**Inference time:** {video_end_time - video_start_time:.3f} seconds")
