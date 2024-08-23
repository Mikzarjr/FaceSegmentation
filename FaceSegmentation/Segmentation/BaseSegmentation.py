import os

import cv2
import numpy as np
from PIL import Image

from FaceSegmentation.src.helpers import ConvertImageToBGR
from FaceSegmentation.src.helpers import ConvertImageToGRAY
from FaceSegmentation.src.helpers import get_image_name
from FaceSegmentation.src.utils import WORK_DIR


class RemoveIntersections:
    def __init__(self, CLASSES: list, MASKS: dict):
        """
        :param CLASSES: All classes (face parts from FaceSeg self.CLASSES)
        :type CLASSES: list
        :param MASKS: Dictionary with all masks class-wise
        :type MASKS: dict
        """
        self.CLASSES = CLASSES
        self.MASKS = MASKS

    @property
    def MainRemoveIntersections(self) -> dict:
        """
        :Description:
        Property {RemoveIntersections} removes all intersections of all masks and
        converts new masks to cv2.GRAY saving them to self.MASKS

        :return: Dictionary with all cleaned masks class-wise
        :rtype: dict
        """
        intersecting_classes: dict[str, list[str]] = \
            {
                "eyes": ["eyebrows", "nose"],
                "face": ["ears", "eyebrows", "eyes", "glasses", "mouth", "neck", "nose"],
                "glasses": ["eyebrows", "eyes", "nose"],
                "hair": ["ears", "eyebrows", "eyes", "face", "glasses", "mouth", "neck", "nose"],
                "nose": ["eyebrows", "mouth"]
            }

        for curr_class in self.CLASSES:
            if curr_class in intersecting_classes:
                self.MASKS[curr_class] = self.PerClassMain(curr_class, intersecting_classes[curr_class])
            else:
                self.MASKS[curr_class] = self.AllRest(curr_class)

        return self.MASKS

    def PerClassBase(self, main_part: np.ndarray, sub_part: str) -> np.ndarray:
        """
        :Description:
        Method {PerClassBase} removes intersection of main_part with <sub_part> from <main_part>

        :Inheritance: Method {PerClassBase} is sub-method of {PerClassMain}
        :param main_part: Mask that is being modified
        :type main_part: np.ndarray
        :param sub_part: Name of class that is intersecting with <main_part>
        :type sub_part: np.ndarray
        :return: main_part class without intersection with sub_part
        :rtype: np.ndarray
        """
        instance = self.MASKS[sub_part]
        instance = ConvertImageToGRAY(instance)

        intersection = cv2.bitwise_and(main_part, instance)
        main_part_wi = cv2.bitwise_and(main_part, cv2.bitwise_not(intersection))
        return main_part_wi

    def PerClassMain(self, current_class: str, other_classes: list) -> np.ndarray:
        """
        :Description:
        Method {PerClassMain} removes all intersections of all masks from
        <other_classes> with <current_class>, converts new masks to cv2.GRAY and saves
        as SPLIT_MASK_DIR/current_class.jpg

        :keyword class: Face part (from FaceSeg self.CLASSES)

        :param current_class: Name of class that is being modified
        :type current_class: str
        :param other_classes: Names of all classes that are intersecting with <current_class>
        :type other_classes: list
        :return: Current mask in cv2.GRAY without intersections
        :rtype: np.ndarray
        """
        curr_mask = self.MASKS[current_class]
        curr_mask = ConvertImageToGRAY(curr_mask)

        for class_name in other_classes:
            curr_mask = self.PerClassBase(curr_mask, class_name)

        return curr_mask

    def AllRest(self, current_class: str) -> np.ndarray:
        """
        :Description:
        Method {AllRest} converts <current_class> to cv2.GRAY format

        :param current_class: Name of class that is converted to grayscale
        :type current_class: str
        :return: current_class in cv2.GRAY format
        :rtype: np.ndarray
        """
        curr_mask = self.MASKS[current_class]
        curr_mask = ConvertImageToGRAY(curr_mask)

        return curr_mask


class SaveSegmentationMasks:
    def __init__(self, image_path, MASKS):
        self.MASKS = MASKS
        self.image_path = image_path
        self.image_name = get_image_name(image_path)

        self.SEG_DIR = f'{WORK_DIR}/segmentation'
        self.CURR_IMAGE_DIR = f'{self.SEG_DIR}/{self.image_name}'
        self.SPLIT_MASK_DIR = f'{self.CURR_IMAGE_DIR}/split_masks'
        self.COMBINED_MASK_DIR = f'{self.CURR_IMAGE_DIR}/combined_masks'

        self.COLORS = {
            'face': [255, 0, 0],
            'eyebrows': [0, 255, 0],
            'eyes': [0, 0, 255],
            'hair': [255, 255, 0],
            'mouth': [0, 255, 255],
            'neck': [255, 0, 255],
            'ears': [255, 165, 0],
            'nose': [128, 0, 128],
            'glasses': [165, 42, 42]
        }

    def SaveMasks(self) -> None:
        """
        :Description:
        Method {SaveMasks} runs the baseline and returns binary segmentation masks

        :rtype: None
        """
        self.MakeDirs()
        self.SaveOriginalImage()
        self.SaveSplitMasks()
        self.SaveCombinedMask()

    def MakeDirs(self) -> None:
        """
        :Description:
        Method {MakeDirs} creates all needed directories

        :rtype: None
        """
        os.makedirs(self.SEG_DIR, exist_ok=True)
        os.makedirs(self.CURR_IMAGE_DIR, exist_ok=True)
        os.makedirs(self.SPLIT_MASK_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_MASK_DIR, exist_ok=True)

    def SaveSplitMasks(self) -> None:
        """
        :Description:
        Method {SaveSplitMasks} saves all masks for each class in {self.SPLIT_MASK_DIR}

        :rtype: None
        """
        for i in self.MASKS:
            Image.fromarray(self.MASKS[i]).save(f"{self.SPLIT_MASK_DIR}/{i}.jpg")

    def SaveOriginalImage(self) -> None:
        """
        :Description:
        Method {SaveOriginalImage} saves original image in {self.WORK_DIR}

        :rtype: None
        """
        img = cv2.imread(self.image_path)
        Image.fromarray(img).save(f"{self.CURR_IMAGE_DIR}/{self.image_name}.jpg")

    def SaveCombinedMask(self) -> None:
        """
        :Description:
        Method {SaveCombinedMask} saves combined mask in {self.COMBINED_MASK_DIR}

        :rtype: None
        """
        img = self.MASKS['face']
        img = ConvertImageToBGR(img)
        combined_mask = np.zeros_like(img)

        for i in self.MASKS:
            mask = Paint(self.MASKS[i], self.COLORS[i])
            combined_mask += mask

        annotated_frame_pil = Image.fromarray(combined_mask)
        annotated_frame_pil.save(f"{self.COMBINED_MASK_DIR}/{self.image_name}.jpg")


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
