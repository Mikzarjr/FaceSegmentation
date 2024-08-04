from FaceSegmentation.Pipeline.Annotation.BaseAnnotation.bboxes import *
from FaceSegmentation.Pipeline.Annotation.BaseAnnotation.polygon import *
from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


class CreateJson:
    def __init__(self, original_image_path):
        self.original_image_path = original_image_path
        self.original_image_dir = GetImageDir(original_image_path)
        self.original_image_name = GetImageName(original_image_path)

        self.mask_dir = f"{self.original_image_dir}/split_masks"

        self.Json = None
        self.ImageHeight, self.ImageWidth = GetImageDimensions(original_image_path)

    def CheckJson(self):
        if self.Json is None:
            print(f"There is no json annotation for '{self.original_image_name}'")
            print("Run 'CreateJsonAnnotation' to create one")
        else:
            coco = COCO(f'{MAIN_DIR}/{self.Json}.json')
            img_dir = self.original_image_dir
            image_id = 0
            img = coco.imgs[image_id]
            image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
            plt.imshow(image, interpolation='nearest')
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            coco.showAnns(anns)

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
        for counter in range(len(folder)):
            file_path = folder[counter]
            ClassName = GetImageName(file_path)
            annotations = self.Annotate(ClassName, counter)
            if len(annotations) == 1:
                coco_data["annotations"] += annotations
            elif len(annotations) > 1:
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

        mask_path = f"/content/segmentation/img1/split_masks/{ClassName}.jpg"
        bbox, area = self.bbox(mask_path)
        polygons = self.polygon(mask_path)
        for i in range(len(bbox)):
            annotation = {
                "id": id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox[i],
                "area": area[i],
                "segmentation": polygons[i],
                "iscrowd": 0
            }
            Annotations.append(annotation)

        return Annotations

    def bbox(self, mask_path):
        BB = bboxes(mask_path)
        bbs = BB.GetBboxCoords()
        areas = BB.Area()
        return bbs, areas

    def polygon(self, mask_path):
        P = polygons(mask_path, self.original_image_path)
        polygons = P.binary_mask_to_polygon()
        return polygons

    def DemoJson(self):
        self.Json = 'qwe'

