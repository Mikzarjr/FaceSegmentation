from FaceSegmentation.Segmentation.Segmentation import FaceSeg
from FaceSegmentation.src.utils import IMGS_DIR

image_path = f"{IMGS_DIR}/img1.jpeg"

S = FaceSeg(image_path)
Masks = S.Segment
S.SaveMasks()
