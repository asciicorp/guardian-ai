# create a streamlit app where there is a sidebar where we can select the tranformer detr model and select a video to run the model on
# the video will be displayed on the main page and the output will be displayed on the main page
import streamlit as st
from PIL import Image

from models.object_detection import DetrDetector
from utils import draw_bboxes, get_output_video

# create a sidebar
st.sidebar.title("Guardian AI")
st.sidebar.subheader("Select a model")
# model selection
model = st.sidebar.selectbox("Select a model", [None, "DETR"])
# load the model
detector = None
if model == "DETR":
    detector = DetrDetector()

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
if image is not None and detector is not None:
    st.subheader("Model Output")
    with st.spinner("Detecting objects..."):
        outputs = detector.detect(image, threshold)
        outputs = [out for out in outputs if out["label"] in labels]
        output_image = draw_bboxes(image, outputs)
    st.image(output_image, caption="Output Image", use_column_width=True)

# #load a video
# video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
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
# display the video in the sidebar
if video is not None:
    st.sidebar.video(video)
    # with open("video.mp4", "wb") as f:
    #     f.write(video.getbuffer())
if video is not None and detector is not None:
    st.subheader("Model Output")
    # loading
    with st.spinner("Detecting objects..."):
        get_output_video(video, detector, labels=labels, threshold=threshold)
    st.video("output.mp4")
