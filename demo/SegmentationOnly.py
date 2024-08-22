from FaceSegmentation.Segmentation.Segmentation import FaceSeg

image_path = f"/docks/TestImages/img1.jpeg"

S = FaceSeg(image_path)
Masks = S.Segment
S.SaveMasks()
