from autodistill.core.composed_detection_model import ComposedDetectionModel
from autodistill.detection import CaptionOntology
from autodistill_clip import CLIP
from autodistill_grounded_sam import GroundedSAM
import cv2
import numpy as np
from PIL import Image
import os
import supervision as sv
from tqdm import tqdm
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import json


MAIN_DIR = os.getcwd()
IMGS_DIR = os.path.join(MAIN_DIR, "FaceSegmentation/docks/TestImages")
