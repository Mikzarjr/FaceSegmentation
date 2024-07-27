from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Segmentation import single_image_segmentation

image_path = f"{IMGS_DIR}/img1.jpeg"

S = single_image_segmentation(image_path)
S.Segment()
