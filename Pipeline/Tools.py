from FaceSegmentation.Pipeline.Config import *


def GetImageName(image_path):
    image_name = os.path.basename(image_path)
    name, _ = os.path.splitext(image_name)
    return name
