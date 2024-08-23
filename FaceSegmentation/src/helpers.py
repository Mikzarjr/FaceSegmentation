import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from FaceSegmentation.src.utils import colored_log


def get_image_format(image_path: str) -> Optional[str]:
    """
    :Description:
    Function {get_image_format} determines the format of the image at the given path.

    :param image_path: The file path to the image.
    :type image_path: str
    :return: The format of the image (e.g., 'JPEG', 'PNG'), or None if the image cannot be opened or is invalid.
    :rtype: Optional[str]
    """
    try:
        with Image.open(image_path) as img:
            return img.format
    except IOError:
        return None


def get_image_name(image_path: str) -> str:
    """
    :Description:
    Function {get_image_name} extracts the name of the image file without its extension.

    :param image_path: The file path to the image.
    :return: The name of the image file without the extension.
    :rtype: str
    """
    image_name = os.path.basename(image_path)
    name, _ = os.path.splitext(image_name)
    return name


def get_image_dir(image_path: str) -> str:
    """
    :Description:
    Function {get_image_dir} gets the directory path where the image is located.

    :param image_path: The file path to the image.
    :return: The directory containing the image.
    :rtype: str
    """
    directory = os.path.dirname(image_path)
    return directory


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """
    :Description:
    Function {get_image_dimensions} retrieves the dimensions (width and height) of the image.

    :param image_path: The file path to the image.
    :return: A tuple containing the width and height of the image in pixels.
    :rtype: Optional[Tuple[int, int]]
    :raises IOError: If the image cannot be opened or read.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except IOError as e:
        colored_log('ERROR', f"Failed to open image '{image_path}'. Error: {e}")
        raise


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
