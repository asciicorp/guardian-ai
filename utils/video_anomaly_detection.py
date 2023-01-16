import torch
import glob
import time
from PIL import Image
import os
import torch

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np


def get_output_video_vad(video, detector, params):
    video_start_time = time.time()

    os.system("rm -rf temp")
    os.makedirs("temp", exist_ok=True)
    os.system(
        f"ffmpeg -i {video} -r {params['fps']} -q:v 2 -vf scale=320:240 temp/%5d.jpg"
    )

    frames = glob.glob("temp/*.jpg")
    inputs = torch.Tensor(1, 3, 16, 240, 320).to(detector.device)
    y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for num, i in enumerate(frames):
        if num < 16:
            inputs[:, :, num, :, :] = ToTensor(1)(Image.open(i))
        else:
            inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :]
            inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(i))
            with torch.no_grad():
                out = detector.predict(inputs)
            y_pred.append(out.item())
    os.system(
        f"ffmpeg -r {params['fps']} -i temp/%5d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p -y output.mp4"
    )
    video_end_time = time.time()
    os.system("rm -rf temp")
    return {"video": "output.mp4", "chart": y_pred}, video_end_time - video_start_time


class ToTensor(object):

    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass
