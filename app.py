# create a streamlit app where there is a sidebar where we can select the tranformer detr model and select a video to run the model on
# the video will be displayed on the main page and the output will be displayed on the main page
import streamlit as st
from PIL import Image
import time

from utils import draw_bboxes, get_output_video, get_device, get_object_detector

st.set_page_config(page_title="GuardianAI", page_icon=":eyes:", layout="wide")
st.sidebar.info(
    "This is a demo app for the [GuardianAI]() project."
)
# logo
st.sidebar.image("logo.png", use_column_width=True)
# ai service selection
ai_service = st.sidebar.selectbox("Select an AI Service", [None, "Object Detection"])
if ai_service == "Object Detection":
    # device selection
    device = st.sidebar.selectbox("Select the device", [None, "CPU", "GPU"])
    device = get_device(device)
    # model selection
    model = st.sidebar.selectbox("Select a Object Detection model", [None, "DETR"])
    object_detector = get_object_detector(model, device)

    labels = st.sidebar.multiselect(
        "Select labels",
        ["person", "car", "bicycle", "motorcycle", "bus", "truck"],
        ["person"],
    )  # select the labels to detect. default is person
    threshold = st.sidebar.slider(
        "Threshold", 0.0, 1.0, 0.5, 0.1
    )  # select the threshold for the model. default is 0.5

    uploaded_image = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"]
    )
    image = st.sidebar.selectbox(
        "Select an image",
        [
            None,
            "samples/burglary_1.jpg",
            "samples/dancing_outside_cctv.webp",
            "samples/indoor_cctv.png",
            "samples/person_street_walking.webp",
            "samples/small_image_person.jpg",
        ],
    )
    if uploaded_image is not None or image is not None:
        if uploaded_image is not None:
            current_image = Image.open(uploaded_image)
        else:
            current_image = Image.open(image)  # type: ignore
        st.sidebar.image(current_image, caption="Uploaded Image", use_column_width=True)
        if object_detector is not None:
            st.subheader("Object Detection Output Image")
            with st.spinner("Detecting objects..."):
                image_start_time = time.time()
                outputs = object_detector.detect(current_image, threshold)
                outputs = [out for out in outputs if out["label"] in labels]
                output_image = draw_bboxes(current_image, outputs)
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
else:
    st.sidebar.info("Please select an AI Service")
    