import cv2
import numpy as np
import supervision as sv
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill.detection import CaptionOntology
from autodistill_clip import CLIP
from autodistill_grounded_sam import GroundedSAM

from FaceSegmentation.Segmentation.BaseSegmentation import RemoveIntersections
from FaceSegmentation.Segmentation.BaseSegmentation import SaveSegmentationMasks
from FaceSegmentation.src.helpers import ConvertImageToGRAY


class FaceSeg:
    def __init__(self, image_path: str):
        """
        :param image_path: Path to image desired for segmentation
        :type image_path: str
        """
        self.image_path = image_path
        self.CLASSES = ['face', 'eyebrows', 'eyes', 'hair', 'mouth', 'neck', 'ears', 'nose', 'glasses']
        self.MASKS: dict[str, np.ndarray] = {class_name: np.ndarray([], dtype=np.float64) for class_name in
                                             self.CLASSES}

    @property
    def Segment(self) -> dict:
        """
        :Description:
        Property {Segment} runs the baseline and returns binary segmentation masks

        :rtype: dict
        :return: Returns <self.MASKS> dictionary dict[str, np.ndarray] where keys are classes (face
        areas) and values are respective binary masks
        """
        self.SegmentImage()
        self.RemoveIntersections()
        # self.DeleteNoize()
        return self.MASKS

    def SaveMasks(self) -> None:
        """
        :Description:
        Method {SaveMasks} runs the baseline and returns binary segmentation masks

        :rtype: None
        """
        SM = SaveSegmentationMasks(self.image_path, self.MASKS)
        SM.SaveMasks()

    def RemoveIntersections(self) -> None:
        """
        :Description:
        Method {RemoveIntersections} removes all intersections of all masks,
        converts new masks to grayscale and saves them in SPLIT_MASK_DIR

        :rtype: None
        """
        RI = RemoveIntersections(self.CLASSES, self.MASKS)
        self.MASKS = RI.MainRemoveIntersections

    def SegmentImage(self) -> None:
        """
        :Description: Method {SegmentImage} runs base models (CLIP grounding DINO and grounding SAM) for face
        segmentation

        :rtype: None
        """

        image = cv2.imread(self.image_path)

        for i, class_name in enumerate(self.CLASSES):
            classes_dict = {class_name: class_name}
            MODEL = ComposedDetectionModel(
                detection_model=GroundedSAM(
                    CaptionOntology(classes_dict)
                ),
                classification_model=CLIP(
                    CaptionOntology({class_name: class_name})
                )
            )

            results = MODEL.predict(self.image_path)

            annotator = sv.MaskAnnotator()
            mask = annotator.annotate(scene=np.zeros_like(image), detections=results)

            self.MASKS[class_name] = mask

    def DeleteNoize(self):
        for image in self.MASKS:
            img = self.MASKS[image]
            img = ConvertImageToGRAY(img)

            _, binary_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

            kernel = np.ones((2, 2), np.uint8)
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

            min_size = 100
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    cleaned_mask[labels == i] = 0

            self.MASKS[image] = cleaned_mask
