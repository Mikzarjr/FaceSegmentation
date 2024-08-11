import logging

import cv2
import numpy as np
from retinaface import RetinaFace


class CoTracker:
    def __init__(self, video_path):
        self.video_path = video_path

    def load_video(self):
        video = self.video_path
        return video

    def process_video(self):
        processed_video = self.video_path
        return processed_video


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

    @property
    def RetrieveFrame(self) -> np.ndarray:
        """
        :Description:
        Property {RetrieveFrame} retrieves a specific frame from a video file.

        :rtype: np.ndarray
        :return: Frame specified by <self.frame_index>
        """

        try:
            video_capture = cv2.VideoCapture(self.video_path)
            if not video_capture.isOpened():
                raise IOError(f"Error: Could not open video file {self.video_path}")

            video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = video_capture.read()
            if not ret:
                raise ValueError(f"Error: Could not read frame at index {self.frame_index}")

        except Exception as e:
            logging.error(f"Failed to retrieve frame: {e}")
            raise
        finally:
            video_capture.release()

        return frame

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
    def DetectFaces(self) -> list:
        """
        :Description: Property {DetectFaces} detects faces in the video frame, expands their bounding boxes,
        and returns the faces as a list of RGB images.

        :rtype: list
        :return: A list of np.ndarray's each containing an RGB image of a detected face.
        """
        images = []

        img = self.RetrieveFrame
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


def CoTracker_and_Faces(video_path, frame):
    """
    TODO: 1 Rework class :CoTracker: output logic
    TODO: 2 Rework {CoTracker_and_Faces}
    """
    CT = CoTracker(video_path)
    CoTracker_output = CT.process_video()

    FD = FaceDetector(video_path, frame)
    FaceDetector_output = FD.DetectFaces

    return CoTracker_output, FaceDetector_output


path_to_video = "../ video.mp4"
frame_index = 0

CoTracker_video, Faces = CoTracker_and_Faces(path_to_video, frame_index)
