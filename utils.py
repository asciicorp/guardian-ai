from PIL import Image, ImageDraw
import os
import glob
import torch
import streamlit as st
from models.object_detection import DetrDetector

LABELS = ["person", "car"]


def draw_bboxes(img, outputs):
    # outputs: list of {'label': 'person', 'score': 0.993, 'box': [185.9, 81.54, 214.87, 124.32]}
    img0 = img.copy()
    draw = ImageDraw.Draw(img0)
    for output in outputs:
        box = output["box"]
        # draw a rectangle with the label and score
        draw.rectangle(box, outline="red", width=3)
        draw.text(
            (box[0], box[1]),
            f"{output['label']} {output['score']}",
            fill="red",
            font=None,
        )
    return img0


def get_output_video(
    video, detector, batch_size=4, labels=LABELS, threshold=0.5, fps=1
):
    os.system("rm -rf temp")
    os.makedirs("temp", exist_ok=True)
    os.system(f"ffmpeg -i {video} -r {fps} temp/%d.jpg")

    frames = [Image.open(frame) for frame in sorted(glob.glob("temp/*.jpg"))]
    outputs = []
    for i in range(0, len(frames), batch_size):
        filtered_outputs = filter_outputs(
            detector.detect_batch(frames[i : i + batch_size], threshold), labels
        )
        outputs.extend(filtered_outputs)

    for i, frame in enumerate(sorted(glob.glob("temp/*.jpg"))):
        frame0 = draw_bboxes(frames[i], outputs[i])
        frame0.save(frame)

    os.system(
        f"ffmpeg -r {fps} -i temp/%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -y output.mp4"
    )
    os.system("rm -rf temp")


def filter_outputs(outputs, labels):
    filtered_outputs = []
    for output in outputs:
        filtered_outputs.append([out for out in output if out["label"] in labels])
    return filtered_outputs


def get_device(device):
    if device is None:
        return None
    elif device == "CPU":
        return torch.device("cpu")
    elif device == "GPU" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        st.sidebar.error("GPU not available")
        return torch.device("cpu")


def get_object_detector(model, device):
    object_detector = None
    if model == "DETR":
        object_detector = DetrDetector(device=device)
    return object_detector
