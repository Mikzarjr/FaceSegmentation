import _io
import os
from functools import singledispatch

import PIL.JpegImagePlugin
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from FaceSegmentation.src.helpers import get_image_format
from FaceSegmentation.src.utils import colored_log, colored_string, show_error


@singledispatch
def ImageConverter(value):
    valid_types = ", ".join(("Pathname", "Numpy Array", "PIL image", "Tensor", "In-memory Binary Streams", "Raw Bytes",
                             "Matplotlib Figure"))
    error = (f"Input image type {colored_string(f"'{value}'", 'BRIGHT_RED')} not supported\n"
             f"Please input image in one of the following supported types: "
             f"{colored_string(f"{valid_types}", 'BRIGHT_RED')}")

    show_error(error)
    # colored_log("ERROR",
    #             f"Input image type {colored_string(f"'{value}'", 'BRIGHT_RED')} not supported\n"
    #             f"Please input image in one of the following supported types: "
    #             f"{colored_string(f"{valid_types}", 'BRIGHT_RED')}")


@ImageConverter.register(str)
def _(value):
    try:
        if not os.path.exists(value):
            raise FileNotFoundError(f"File not found: {value}")
        image_format = get_image_format(value)
        if image_format is None:
            colored_log("WARNING", f"File at {value} is not recognized as a valid image.")
        else:
            colored_log("INFO", f"Processing {image_format} image from path: {value}")
    except Exception as e:
        colored_log("ERROR", f"Error processing file path: {e}")


@ImageConverter.register(np.ndarray)
def _(value):
    if len(value.shape) == 2:
        colored_log("INFO", f"Image processed as Numpy Array: {value.shape}")
    else:
        colored_log("INFO", f"Image processed as Numpy Array: {value.shape}")


@ImageConverter.register(PIL.Image.Image)
def _(value):
    print(f"Image processed as PIL image with format: {value.format}")


@ImageConverter.register(torch.Tensor)
def _(value):
    print(f"Image processed as Tensor: {value}")


@ImageConverter.register(_io.BytesIO)
def _(value):
    print(f"Image processed as In-memory Binary Streams: {value}")


@ImageConverter.register(bytes)
def _(value):
    print(f"Image processed as Raw Bytes: {value}")


@ImageConverter.register(plt.Figure)
def _(value):
    print(f"Image processed as Matplotlib Figure: {value}")


# image_path = "/Users/mike/Documents/GitHub/Face-Segmentation/constant/Assets/TestImages/img1.jpeg"
image_path = "/Users/mike/Documents/GitHub/Face-Segmentation/FaceSegmentation/src/image_converter.py"
# image_path = "/Users/mike/Documents/GitHub/Face-Segmentation/constant/Assets/TestImages/1.sm.webp"


# cv2_image = cv2.imread(image_path)
# PIL_image = Image.open(image_path)
#
# transform = transforms.ToTensor()
# tensor_img = transform(PIL_image)
#
# binary_image = io.BytesIO()
# PIL_image.save(binary_image, format="JPEG")
#
# with open(image_path, "rb") as f:
#     raw_bytes_image = f.read()

ImageConverter(False)

# colors = {
#     "RESET": "30",
#     "RED": "31",
#     "GREEN": "32",
#     "YELLOW": "33",
#     "BLUE": "34",
#     "MAGENTA": "35",
#     "CYAN": "36",
#     "WHITE": "37",
#     "BRIGHT_BLACK": "1;90",
#     "BRIGHT_RED": "1;91",
#     "BRIGHT_GREEN": "1;92",
#     "BRIGHT_YELLOW": "1;93",
#     "BRIGHT_BLUE": "1;94",
#     "BRIGHT_MAGENTA": "1;95",
#     "BRIGHT_CYAN": "1;96",
#     "BRIGHT_WHITE": "1;97"
# }
# for i in colors:
#     print(colored_string(f"Hello World!", i))
