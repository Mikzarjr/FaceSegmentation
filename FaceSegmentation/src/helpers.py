import os

import cv2
import numpy as np
from PIL import Image


def get_image_format(image_path: str) -> object:
    """
    :Description:
    Function {get_image_format} ...

    :param image_path:
    :return:
    :rtype: object
    """
    try:
        with Image.open(image_path) as img:
            return img.format
    except IOError:
        return None


def get_image_name(image_path):
    image_name = os.path.basename(image_path)
    name, _ = os.path.splitext(image_name)
    return name


def get_image_dir(image_path):
    directory = os.path.dirname(image_path)
    return directory


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def ConvertImageToGRAY(image: np.ndarray) -> np.ndarray:
    """
    :Description:
    Function {ConvertImageToGRAY} converts the image in cv2.GRAY format.

    :param image: Original image
    :type image: np.ndarray
    :rtype: np.ndarray
    :return: Image in cv2.GRAY
    """
    if image is None:
        raise ValueError("The image is None.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Invalid image format.")


def ConvertImageToBGR(image: np.ndarray) -> np.ndarray:
    """
    :Description:
    Function {ConvertImageToBGR} converts the image in cv2.BGR format.

    :param image: Original image
    :type image: np.ndarray
    :rtype: np.ndarray
    :return: Image in cv2.BGR
    """
    if image is None:
        raise ValueError("The image is None.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        return image
    else:
        raise ValueError("Invalid image format.")
