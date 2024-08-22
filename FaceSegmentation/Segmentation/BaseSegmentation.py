import numpy as np

from FaceSegmentation.src.helpers import ConvertImageToBGR


def Paint(mask: np.ndarray, color: list) -> np.ndarray:
    """
    :Description:
    Function {Paint} paints mask of desired class to desired color

    :param mask: Mask in cv2.GRAY format
    :type mask: np.ndarray
    :param color: RGB color
    :type color: list
    :rtype: np.ndarray
    :return: Colored mask
    """
    image = ConvertImageToBGR(mask)

    mask = np.any(image >= 50, axis=2)
    blacked = np.any(image < 50, axis=2)

    image[mask] = color
    image[blacked] = [0, 0, 0]

    return image
