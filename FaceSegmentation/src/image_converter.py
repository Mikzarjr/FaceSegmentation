import _io
from functools import singledispatch

import PIL.JpegImagePlugin
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from FaceSegmentation.src.helpers import get_image_format
from FaceSegmentation.src.utils import colored_log, colored_string, show_error


@singledispatch
def ImageConverter(value):
    valid_types = ", ".join(("Pathname", "Numpy Array", "PIL image", "Tensor", "In-memory Binary Streams", "Raw Bytes",
                             "Matplotlib figure"))
    error = (f"Input image type {colored_string(f"'{value}'", 'BRIGHT_MAGENTA')} not supported\n"
             f"Please input image in one of the following supported types: "
             f"{colored_string(f"{valid_types}", 'BRIGHT_YELLOW')}")

    show_error(error)


@ImageConverter.register(str)
def _(value):
    try:
        image_format = get_image_format(value)
        if image_format is None:
            show_error(f"File at {value} is not recognized as a valid image.")
        else:
            colored_log("INFO", f"Processing {image_format} image from path: {colored_string(value, "CYAN")}")
    except Exception as e:
        show_error(f"Error processing file path: {e}")


@ImageConverter.register(np.ndarray)
def _(value):
    colored_log("INFO", f"Image processed as Numpy Array with dimensions: {colored_string(value.shape, "CYAN")}")


@ImageConverter.register(PIL.Image.Image)
def _(value):
    print(f"Image processed as PIL image with format: {colored_string(value.format, "CYAN")}")


@ImageConverter.register(torch.Tensor)
def _(value):
    print(f"Image processed as Tensor: {colored_string(value, "CYAN")}")


@ImageConverter.register(_io.BytesIO)
def _(value):
    print(f"Image processed as In-memory Binary Streams: {colored_string(value, "CYAN")}")


@ImageConverter.register(bytes)
def _(value):
    print(f"Image processed as Raw Bytes: {colored_string(value, "CYAN")}")


@ImageConverter.register(plt.Figure)
def _(value):
    print(f"Image processed as Matplotlib Figure: {colored_string(value, "CYAN")}")


image_path = "/Users/mike/Documents/GitHub/Face-Segmentation/constant/Assets/TestImages/img1.jpeg"
ImageConverter(0)
