from PIL import ImageDraw, ImageFont, Image
import os
import glob
import time
import random


def draw_bboxes(img, outputs, colors):
    width, height = img.size
    img0 = img.copy()
    img0 = img0.resize((int(width * (1080 / height)), 1080))
    draw = ImageDraw.Draw(img0)
    for output in outputs:
        box = [x * (1080 / height) for x in output["box"]]
        box_h, box_w = box[2] - box[0], box[3] - box[1]
        label: str = output["label"]
        score: float = output["score"]
        color = colors[label] if len(colors.keys()) > 1 else "white"
        draw.rounded_rectangle(
            (box[0], box[1], box[2], box[3]),
            outline=color,
            width=5,
            radius=5,
        )
        draw.text(
            (box[0], box[3] + 5),
            f"{label.upper()} {int(score*100)}%"
            if box_w > 100
            else f"{label.upper()}"
            if box_w > 50
            else "",
            fill=color,
            font=ImageFont.truetype("public/oswald.ttf", 20),
        )
    return img0


def get_output_video_od(video, detector, model_name, params):
    video_start_time = time.time()
    os.system("rm -rf temp")
    os.makedirs("temp", exist_ok=True)
    os.system(f"ffmpeg -i {video} -r {params['fps']} temp/%d.jpg")

    frames = [Image.open(frame) for frame in sorted(glob.glob("temp/*.jpg"))]
    outputs = []
    colors = {}
    if model_name == "yolov8n":
        for i in range(0, len(frames), params["batch_size"]):
            filtered_outputs = detector.detect_batch(frames[i : i + params["batch_size"]])
            outputs.extend(filtered_outputs)

        for i, frame in enumerate(sorted(glob.glob("temp/*.jpg"))):
            frame0 = outputs[i]
            frame0 = Image.fromarray(frame0)
            frame0.save(frame)

    else:
        for label in params["labels"]:
            colors[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        for i in range(0, len(frames), params["batch_size"]):
            filtered_outputs = filter_outputs(
                detector.detect_batch(
                    frames[i : i + params["batch_size"]], params["threshold"]
                ),
                params["labels"],
            )
            outputs.extend(filtered_outputs)

        for i, frame in enumerate(sorted(glob.glob("temp/*.jpg"))):
            frame0 = draw_bboxes(frames[i], outputs[i], colors)
            frame0.save(frame)

    os.system(
        f"ffmpeg -r {params['fps']} -i temp/%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -y output.mp4"
    )
    os.system("rm -rf temp")
    video_end_time = time.time()
    return {"video" : "output.mp4"}, video_end_time - video_start_time

def get_output_image_od(img, detector, model_name, params):
    img_start_time = time.time()
    outputs = detector.detect_batch([img], params["threshold"])
    if model_name == "yolov8n":
        img_end_time = time.time()
        return outputs, img_end_time - img_start_time
    else:
        filtered_outputs = filter_outputs(outputs, params["labels"])
        colors = {}
        for label in params["labels"]:
            colors[label] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
        img0 = draw_bboxes(img, filtered_outputs[0], colors)
        img_end_time = time.time()
        return img0, img_end_time - img_start_time

def filter_outputs(outputs, labels):
    filtered_outputs = []
    for output in outputs:
        filtered_outputs.append([out for out in output if out["label"] in labels])
    return filtered_outputs
