from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Tools import *

C
class CreateJson:
    def __init__(self, image_path, json_name):
        self.image_path = image_path
        self.image_dir = GetImageDir(image_path)
        self.image_name = GetImageName(image_path)
        self.json = None
        self.json_name = json_name

    def CheckJson(self):
        if self.json is None:
            print(f"There is no json annotation for {self.image_name}")
            print(f"Run CreateJsonAnnotation to create one")
        else:
            coco = COCO(f'{MAIN_DIR}/{self.json}.json')
            img_dir = {self.image_dir}
            image_id = 0
            img = coco.imgs[image_id]
            image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
            plt.imshow(image, interpolation='nearest')
            cat_ids = coco.getCatIds()
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            coco.showAnns(anns)

    def CreateJsonAnnotation(self):
        print("CREATE JSON METHOD EXECUTED")
        self.json = self.json_name
