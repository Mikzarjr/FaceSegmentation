from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


class CreateJson:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_dir = GetImageDir(image_path)
        self.image_name = GetImageName(image_path)
        self.Json = None
        self.ImageHeight, self.ImageWidth = GetImageDimensions(image_path)

    def CheckJson(self):
        if self.Json is None:
            print(f"There is no json annotation for '{self.image_name}'")
            print("Run 'CreateJsonAnnotation' to create one")
        else:
            coco = COCO(f'{MAIN_DIR}/{self.Json}.json')
            img_dir = self.image_dir
            image_id = 0
            img = coco.imgs[image_id]
            image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
            plt.imshow(image, interpolation='nearest')
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            coco.showAnns(anns)

    def CreateJsonAnnotation(self):
        with open('/content/FaceSegmentation/Pipeline/ConstantData.json', 'r') as f:
            coco_data = json.load(f)

        image_info = {
            "id": 0,
            "license": 1,
            "file_name": self.image_path,
            "height": self.ImageHeight,
            "width": self.ImageWidth
        }

        coco_data["images"] = [image_info]

        self.Json = "coco_annotations"
        with open(f"{self.Json}.json", "w") as json_file:
            json.dump(coco_data, json_file, indent=4)

        self.Json = self.Json

    def DemoJson(self):
        self.Json = 'qwe'

