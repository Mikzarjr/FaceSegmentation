from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


class CreateJson:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_dir = GetImageDir(image_path)
        self.image_name = GetImageName(image_path)
        self.Json = None

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
        with open('ConstantData.json', 'r') as f:
            coco_data = json.load(f)

        self.Json = "coco_annotations"
        with open(f"{self.Json}.json", "w") as json_file:
            json.dump(coco_data, json_file, indent=4)

        self.Json = self.Json
