import os
import time
import glob
from PIL import Image


def get_output_video_de(video, detector, params):
    video_start_time = time.time()
    os.system("rm -rf temp")
    os.makedirs("temp", exist_ok=True)
    os.system(f"ffmpeg -i {video} -r {params['fps']} temp/%d.jpg")

    frames = [Image.open(frame) for frame in sorted(glob.glob("temp/*.jpg"))]
    outputs = []
    for i in range(0, len(frames), params["batch_size"]):
        filtered_outputs = detector.estimate_batch(frames[i : i + params["batch_size"]])
        outputs.extend(filtered_outputs)

    for i, frame in enumerate(sorted(glob.glob("temp/*.jpg"))):
        frame0 = outputs[i]
        frame0 = Image.fromarray(frame0)
        frame0.save(frame)

    os.system(
        f"ffmpeg -r {params['fps']} -i temp/%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -y output.mp4"
    )
    os.system("rm -rf temp")
    video_end_time = time.time()
    return video_end_time - video_start_time


def get_output_image_de(img, detector, params):
    img_start_time = time.time()
    outputs = detector.estimate_batch([img])
    img_end_time = time.time()
    return outputs, img_end_time - img_start_time
