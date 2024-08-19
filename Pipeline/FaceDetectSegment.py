import logging
import os
import supervision as sv
import cv2
import numpy as np
from retinaface import RetinaFace
from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill.detection import CaptionOntology
from autodistill_clip import CLIP
from autodistill_grounded_sam import GroundedSAM


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


def RetrieveFrame(video_path: str, frame_index: int) -> np.ndarray:
    """
    :Description:
    Property {RetrieveFrame} retrieves a specific frame from a video file.

    :param video_path: Path to video
    :type video_path: str
    :param frame_index: Index of frame
    :type frame_index: int
    :rtype: np.ndarray
    :return: Frame specified by <frame_index>
    """

    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise IOError(f"Error: Could not open video file {video_path}")

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_capture.read()
        if not ret:
            raise ValueError(f"Error: Could not read frame at index {frame_index}")

    except Exception as e:
        logging.error(f"Failed to retrieve frame: {e}")
        raise
    finally:
        video_capture.release()

    return frame


class FaceDetector:
    def __init__(self, video_path: str, frame_index: int):
        """
        :param video_path: Path to video
        :type video_path: str
        :param frame_index: Index of frame
        :type frame_index: int
        :rtype: None
        """
        self.video_path = video_path
        self.frame_index = frame_index

    @staticmethod
    def ExpandBBox(bbox: tuple, img_width: float, img_height: float, scale: float = 1.5) -> tuple:
        """
        :Description: Staticmethod {ExpandBBox} expands a bounding box by a given scale factor while ensuring it
        remains within the image boundaries.

        :param bbox: Tuple representing the original bounding box (x, y, width, height).
        :type bbox: tuple
        :param img_width: The width of the image in which the bounding box is situated.
        :type img_width: float
        :param img_height: The height of the image in which the bounding box is situated.
        :type img_height: float
        :param scale: Const factor by which the bounding box dimensions are to be scaled.
        :type scale: float
        :rtype: tuple
        :return: Tuple representing the expanded bounding box, adjusted for image boundaries.
        """
        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2
        scaled_w, scaled_h = w * scale, h * scale
        new_x = max(0, int(center_x - scaled_w / 2))
        new_y = max(0, int(center_y - scaled_h / 2))
        new_w = int(min(scaled_w, img_width - new_x))
        new_h = int(min(scaled_h, img_height - new_y))

        return new_x, new_y, new_w, new_h

    @property
    def Detect(self) -> list:
        """
        :Description: Property {DetectFaces} detects faces in the video frame, expands their bounding boxes,
        and returns the faces as a list of RGB images.

        :rtype: list
        :return: A list of np.ndarray's each containing an RGB image of a detected face.
        """
        images = []

        img = RetrieveFrame(self.video_path, self.frame_index)
        img_height, img_width = img.shape[:2]
        detections = RetinaFace.detect_faces(img)

        for _, key in enumerate(detections.keys()):
            identity = detections[key]
            facial_area = identity["facial_area"]

            x, y, w, h = facial_area[0], facial_area[1], facial_area[2] - facial_area[0], facial_area[3] - facial_area[
                1]
            new_x, new_y, new_w, new_h = self.ExpandBBox((x, y, w, h), img_width, img_height)
            expanded_face = img[new_y:new_y + new_h, new_x:new_x + new_w]
            expanded_face_rgb = cv2.cvtColor(expanded_face, cv2.COLOR_BGR2RGB)
            images.append(expanded_face_rgb)

        return images


class FaceOnlySeg:
    def __init__(self, video_path: str):
        """
        :param video_path: Path to video
        :type video_path: str
        """
        self.video_path = video_path
        self.frame_index = 0
        self.CLASSES = ['face']
        self.COLORS = {
            'face': [255, 0, 0],
        }
        self.MASKS: dict[str, np.ndarray] = {class_name: np.ndarray([], dtype=np.float64) for class_name in
                                             self.CLASSES}
        self.image = RetrieveFrame(self.video_path, self.frame_index)

    def Segment(self):
        MODEL = ComposedDetectionModel(
            detection_model=GroundedSAM(
                CaptionOntology({self.CLASSES[0]: self.CLASSES[0]})
            ),
            classification_model=CLIP(
                CaptionOntology({k: k for k in self.CLASSES})
            )
        )
        Image.fromarray(self.image).save("face_temp.jpg")
        results = MODEL.predict("face_temp.jpg")
        os.remove("face_temp.jpg")

        annotator = sv.MaskAnnotator()
        mask = annotator.annotate(scene=np.zeros_like(self.image), detections=results)

        return mask

    def SegmentFace(self):
        mask = ConvertImageToGRAY(self.Segment())

        mask = mask / 255.0
        mask = cv2.merge([mask, mask, mask])
        output = self.image * mask
        output = np.clip(output, 0, 255).astype(np.uint8)

        return output


def DetectFaces(video_path, frame):
    FD = FaceDetector(video_path, frame)
    FaceDetector_output = FD.Detect
    return FaceDetector_output


def SegmentFaces(video_path):
    FOS = FaceOnlySeg(video_path)
    FaceOnlySeg_output = FOS.SegmentFace()
    return FaceOnlySeg_output


path_to_video = "../ video.mp4"
frame_index = 0

Faces = DetectFaces(path_to_video, frame_index)
Faces2 = SegmentFaces(path_to_video)
