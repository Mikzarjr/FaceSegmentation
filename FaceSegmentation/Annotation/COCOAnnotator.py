import json
import os
from typing import Optional

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

from FaceSegmentation.Annotation.BaseAnnotation.Bboxes import bboxes
from FaceSegmentation.Annotation.BaseAnnotation.Polygon import polygons
from FaceSegmentation.src.helpers import get_image_dir, get_image_name, get_image_dimensions
from FaceSegmentation.src.utils import MAIN_DIR


class CreateJson:
    def __init__(self, original_image_path):
        self.original_image_path = original_image_path
        self.original_image_dir = get_image_dir(original_image_path)
        self.original_image_name = get_image_name(original_image_path)
        print(self.original_image_dir)
        self.mask_dir = f"{self.original_image_dir}/split_masks"

        self.Json = None
        self.ImageHeight, self.ImageWidth = get_image_dimensions(original_image_path)

    def BaseDrawAnnotatoins(self):
        coco = COCO(f'{MAIN_DIR}/{self.Json}.json')
        img_dir = self.original_image_dir
        image_id = 0
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
        cat_ids = coco.getCatIds()
        return coco, img, image, cat_ids

    def DrawAnnotations(self, task, coco, img, image, cat_ids: Optional[object], cat_id: Optional[object]):
        """

        """
        if task == "all":
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            category_name = "Annotations"
        elif task == "byclass":
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=[cat_id], iscrowd=None)
            category_name = coco.loadCats(cat_id)[0]['name']
        anns = coco.loadAnns(anns_ids)
        plt.figure(figsize=(10, 10))
        plt.imshow(image, interpolation='nearest')
        coco.showAnns(anns)
        plt.title(f"{category_name}")
        plt.axis('off')
        plt.show()

    def VisualizeAll(self):
        if self.Json is None:
            print(f"There is no json annotation for '{self.original_image_name}'")
            print("Run 'CreateJsonAnnotation' to create one")
        else:
            coco, img, image, cat_ids = self.BaseDrawAnnotatoins()
            self.DrawAnnotations("all", coco, img, image, cat_ids, None)

    def VisualizeByClass(self):
        if self.Json is None:
            print(f"There is no json annotation for '{self.original_image_name}'")
            print("Run 'CreateJsonAnnotation' to create one")
        else:
            coco, img, image, cat_ids = self.BaseDrawAnnotatoins()
            for cat_id in cat_ids:
                self.DrawAnnotations("byclass", coco, img, image, None, cat_id)

    def CreateJsonAnnotation(self):
        with open(f'{MAIN_DIR}/FaceSegmentation/docks/Constant/Formats/JSON/ConstantData.json', 'r') as f:
            coco_data = json.load(f)

        image_info = {
            "id": 0,
            "license": 1,
            "file_name": self.original_image_path,
            "height": self.ImageHeight,
            "width": self.ImageWidth
        }
        coco_data["images"] = [image_info]

        folder = os.listdir(self.mask_dir)
        ccc = 0
        for counter in range(len(folder)):
            file_path = folder[counter]
            ClassName = get_image_name(file_path)
            annotations, l = self.Annotate(ClassName, ccc)
            ccc += l
            for i in range(len(annotations)):
                coco_data["annotations"] += [annotations[i]]

        self.Json = "coco_annotations"
        with open(f"{self.Json}.json", "w") as json_file:
            json.dump(coco_data, json_file, indent=4)

        self.Json = self.Json

    def Annotate(self, ClassName, counter):
        Annotations = []

        id = counter
        image_id = 0
        CLASSES = ['face', 'ears', 'eyebrows', 'eyes', 'glasses', 'nose', 'hair', 'mouth', 'neck']
        category_id = CLASSES.index(ClassName)

        mask_path = f"/content/segmentation/{self.original_image_name}/split_masks/{ClassName}.jpg"
        bbox, area = self.bbox(mask_path)
        plgns = self.polygon(mask_path)
        l = len(bbox)
        for i in range(l):
            annotation = {
                "id": id + i,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox[i],
                "area": area[i],
                "segmentation": [plgns[i]],
                "iscrowd": 0
            }
            Annotations.append(annotation)

        return Annotations, l

    def bbox(self, mask_path):
        BB = bboxes(mask_path)
        bbs = BB.GetBboxCoords()
        areas = BB.Area()
        return bbs, areas

    def polygon(self, mask_path):
        P = polygons(mask_path, self.original_image_path)
        plgns = P.binary_mask_to_polygon()
        return plgns

    def DemoJson(self):
        self.Json = 'qwe'
