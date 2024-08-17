from Pipeline.Config import *


def GetImageName(image_path):
    image_name = os.path.basename(image_path)
    name, _ = os.path.splitext(image_name)
    return name


def GetImageDir(image_path):
    directory = os.path.dirname(image_path)
    return directory


def GetImageDimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height
