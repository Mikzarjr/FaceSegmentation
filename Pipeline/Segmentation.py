from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


class FaceSeg:
    def __init__(self, image_path):
        self.image_path = image_path
        self.SEG_DIR = f'{MAIN_DIR}/segmentation'
        self.image_name = GetImageName(image_path)
        self.WORK_DIR = f'{self.SEG_DIR}/{self.image_name}'
        self.SPLIT_MASK_DIR = f'{self.WORK_DIR}/split_masks'
        self.COMBINED_MASK_DIR = f'{self.WORK_DIR}/combined_masks'
        self.CLASSES = ['face', 'eyebrows', 'eyes', 'hair', 'mouth', 'neck', 'ears', 'nose', 'glasses']
        self.COLORS = [[255, 0, 0],
                       [0, 255, 0],
                       [0, 0, 255],
                       [255, 255, 0],
                       [0, 255, 255],
                       [255, 0, 255],
                       [255, 165, 0],
                       [128, 0, 128],
                       [165, 42, 42],
                       [191, 255, 0]]

    def MakeDirs(self):
        os.makedirs(self.SEG_DIR, exist_ok=True)
        os.makedirs(self.WORK_DIR, exist_ok=True)
        os.makedirs(self.SPLIT_MASK_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_MASK_DIR, exist_ok=True)

    def Segment(self):
        self.MakeDirs()
        self.SegmentImage()
        self.SaveMask()
        self.DeleteOtherMasks()

    def SegmentImage(self):
        all_results = []
        image = cv2.imread(self.image_path)

        for i, class_name in enumerate(self.CLASSES):
            classes_dict = {class_name: class_name}
            MODEL = ComposedDetectionModel(
                detection_model=GroundedSAM(
                    CaptionOntology(classes_dict)
                ),
                classification_model=CLIP(
                    CaptionOntology({class_name: class_name})
                )
            )

            results = MODEL.predict(self.image_path)
            all_results.append(results)

            annotator = sv.MaskAnnotator()
            mask = annotator.annotate(scene=np.zeros_like(image), detections=results)
            Image.fromarray(mask).save(f"{self.SPLIT_MASK_DIR}/{class_name}.jpg")

        for i in self.CLASSES:
            if i == 'face':
                FACE = self.FACE_WithoutIntersections()
                Image.fromarray(FACE).save(f"{self.SPLIT_MASK_DIR}/face_WI.jpg")
            elif i == 'eyes':
                EYES = self.EYES_WithoutIntersections()
                Image.fromarray(EYES).save(f"{self.SPLIT_MASK_DIR}/eyes_WI.jpg")
            elif i == 'nose':
                NOSE = self.NOSE_WithoutIntersections()
                Image.fromarray(NOSE).save(f"{self.SPLIT_MASK_DIR}/nose_WI.jpg")
            elif i == 'glasses':
                GLASSES = self.GLASSES_WithoutIntersections()
                Image.fromarray(GLASSES).save(f"{self.SPLIT_MASK_DIR}/glasses_WI.jpg")
            elif i == 'hair':
                HAIR = self.HAIR_WithoutIntersections()
                Image.fromarray(HAIR).save(f"{self.SPLIT_MASK_DIR}/hair_WI.jpg")
            else:
                CLASS = self.REST_WithoutIntersection(i)
                Image.fromarray(CLASS).save(f"{self.SPLIT_MASK_DIR}/{i}_WI.jpg")

    def FACE_WithoutIntersections(self):
        FACE = cv2.imread(f"{self.SPLIT_MASK_DIR}/face.jpg", cv2.IMREAD_GRAYSCALE)

        for i in self.CLASSES:
            if i != 'face' and i != 'hair':
                instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/{i}.jpg", cv2.IMREAD_GRAYSCALE)
                intersection = cv2.bitwise_and(FACE, instance)
                FACE = cv2.bitwise_and(FACE, cv2.bitwise_not(intersection))
        return FACE

    def HAIR_WithoutIntersections(self):
        HAIR = cv2.imread(f"{self.SPLIT_MASK_DIR}/hair.jpg", cv2.IMREAD_GRAYSCALE)

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/face.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(HAIR, instance)
        HAIR = cv2.bitwise_and(HAIR, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/neck.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(HAIR, instance)
        HAIR = cv2.bitwise_and(HAIR, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/ears.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(HAIR, instance)
        HAIR = cv2.bitwise_and(HAIR, cv2.bitwise_not(intersection))
        return HAIR

    def EYES_WithoutIntersections(self):
        EYES = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyes.jpg", cv2.IMREAD_GRAYSCALE)

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/nose.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(EYES, instance)
        EYES = cv2.bitwise_and(EYES, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyebrows.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(EYES, instance)
        EYES = cv2.bitwise_and(EYES, cv2.bitwise_not(intersection))
        return EYES

    def GLASSES_WithoutIntersections(self):
        GLASSES = cv2.imread(f"{self.SPLIT_MASK_DIR}/glasses.jpg", cv2.IMREAD_GRAYSCALE)

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyes.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(GLASSES, instance)
        GLASSES = cv2.bitwise_and(GLASSES, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyebrows.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(GLASSES, instance)
        GLASSES = cv2.bitwise_and(GLASSES, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/nose.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(GLASSES, instance)
        GLASSES = cv2.bitwise_and(GLASSES, cv2.bitwise_not(intersection))
        return GLASSES

    def NOSE_WithoutIntersections(self):
        NOSE = cv2.imread(f"{self.SPLIT_MASK_DIR}/nose.jpg", cv2.IMREAD_GRAYSCALE)

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/mouth.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(NOSE, instance)
        NOSE = cv2.bitwise_and(NOSE, cv2.bitwise_not(intersection))

        instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyebrows.jpg", cv2.IMREAD_GRAYSCALE)
        intersection = cv2.bitwise_and(NOSE, instance)
        NOSE = cv2.bitwise_and(NOSE, cv2.bitwise_not(intersection))
        return NOSE

    def REST_WithoutIntersection(self, CLASS):
        CLASS = cv2.imread(f"{self.SPLIT_MASK_DIR}/{CLASS}.jpg", cv2.IMREAD_GRAYSCALE)
        return CLASS

    @staticmethod
    def Paint(image_path, color):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        mask = np.any(image >= 50, axis=2)
        blacked = np.any(image < 50, axis=2)

        image[mask] = color
        image[blacked] = [0, 0, 0]

        return image

    # def GetImageName(self):
    #     image_name = os.path.basename(self.image_path)
    #     name, _ = os.path.splitext(image_name)
    #     return name

    def SaveMask(self):
        img = cv2.imread(f'{self.SPLIT_MASK_DIR}/face_WI.jpg')
        combined_mask = np.zeros_like(img)

        for i in range(len(self.CLASSES)):
            path = f"{self.SPLIT_MASK_DIR}/{self.CLASSES[i]}_WI.jpg"
            mask = self.Paint(path, self.COLORS[i])
            combined_mask += mask

        annotated_frame_pil = Image.fromarray(combined_mask)
        annotated_frame_pil.save(f"{self.COMBINED_MASK_DIR}/{self.image_name}.jpg")

    def DeleteOtherMasks(self):
        """
        Deletes all files in the SPLIT_MASK_DIR that do not have "WI" in their name.
        Renames files containing "WI" by removing "WI" from their names.
        """
        for file in os.listdir(self.SPLIT_MASK_DIR):
            if "WI" not in file:
                file_path = os.path.join(self.SPLIT_MASK_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        for file in os.listdir(self.SPLIT_MASK_DIR):
            if "WI" in file:
                new_file_name = file.replace("_WI", "")
                old_file_path = os.path.join(self.SPLIT_MASK_DIR, file)
                new_file_path = os.path.join(self.SPLIT_MASK_DIR, new_file_name)
                if os.path.isfile(old_file_path):
                    os.rename(old_file_path, new_file_path)
