from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


def ConvertImageToGRAY(image: np.ndarray) -> np.ndarray:
    """
    :Description:
    Function {ConvertImageToGRAY} converts the image in cv2.GRAY format.

    :param image: Original image
    :type image: np.ndarray
    :rtype: np.ndarray
    :return: Image in cv2.GRAY
    """
    if image is None:
        raise ValueError("The image is None.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        return image
    else:
        raise ValueError("Invalid image format.")


def ConvertImageToBGR(image: np.ndarray) -> np.ndarray:
    """
    :Description:
    Function {ConvertImageToBGR} converts the image in cv2.BGR format.

    :param image: Original image
    :type image: np.ndarray
    :rtype: np.ndarray
    :return: Image in cv2.BGR
    """
    if image is None:
        raise ValueError("The image is None.")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        return image
    else:
        raise ValueError("Invalid image format.")


class FaceSeg:
    def __init__(self, image_path: str):
        """
        TODO: 1 Finish all dock-strings
        TODO: 2 Naming

        :param image_path: Path to image desired for segmentation
        :type image_path: str
        """
        self.image_path = image_path
        self.SEG_DIR = f'{MAIN_DIR}/segmentation'
        self.image_name = GetImageName(image_path)
        self.WORK_DIR = f'{self.SEG_DIR}/{self.image_name}'
        self.SPLIT_MASK_DIR = f'{self.WORK_DIR}/split_masks'
        self.COMBINED_MASK_DIR = f'{self.WORK_DIR}/combined_masks'
        self.CLASSES = ['face', 'eyebrows', 'eyes', 'hair', 'mouth', 'neck', 'ears', 'nose', 'glasses']

        self.COLORS = {
            'face': [255, 0, 0],
            'eyebrows': [0, 255, 0],
            'eyes': [0, 0, 255],
            'hair': [255, 255, 0],
            'mouth': [0, 255, 255],
            'neck': [255, 0, 255],
            'ears': [255, 165, 0],
            'nose': [128, 0, 128],
            'glasses': [165, 42, 42]
        }

        self.MASKS: dict[str, np.ndarray] = {class_name: np.ndarray([], dtype=np.float64) for class_name in
                                             self.CLASSES}

    @property
    def Segment(self) -> dict:
        """
        :Description:
        Property {Segment} runs the baseline and returns binary segmentation masks

        :rtype: dict
        :return: Returns <self.MASKS> dictionary dict[str, np.ndarray] where keys are classes (face
        areas) and values are respective binary masks
        """
        self.SegmentImage()
        self.RemoveIntersections()
        self.DeleteNoize()
        return self.MASKS

    def SaveMasks(self) -> None:
        """
        :Description:
        Method {SaveMasks} runs the baseline and returns binary segmentation masks

        :rtype: None
        """
        self.Prepare()
        self.SaveOriginalImage()
        self.SaveSplitMasks()
        self.SaveCombinedMask()

    def Prepare(self) -> None:
        """
        :Description:
        Method {Prepare} creates all needed directories

        :rtype: None
        """
        os.makedirs(self.SEG_DIR, exist_ok=True)
        os.makedirs(self.WORK_DIR, exist_ok=True)
        os.makedirs(self.SPLIT_MASK_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_MASK_DIR, exist_ok=True)

    def SegmentImage(self) -> None:
        """
        :Description:
        Method {SegmentImage} creates all needed directories

        :rtype: None
        """

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

            annotator = sv.MaskAnnotator()
            mask = annotator.annotate(scene=np.zeros_like(image), detections=results)

            self.MASKS[class_name] = mask

    def RemoveIntersections(self) -> None:
        """
        :Description:
        Method {RemoveIntersections} removes all intersections of all masks,
        converts new masks to grayscale and saves them in SPLIT_MASK_DIR

        :rtype: None
        """
        RI = RemoveIntersections(self.CLASSES, self.MASKS)
        self.MASKS = RI.MainRemoveIntersections

    def SaveSplitMasks(self) -> object:
        """

        :rtype: object
        """
        for i in self.MASKS:
            Image.fromarray(self.MASKS[i]).save(f"{self.SPLIT_MASK_DIR}/{i}.jpg")

    def SaveOriginalImage(self):
        img = cv2.imread(self.image_path)
        Image.fromarray(img).save(f"{self.WORK_DIR}/{self.image_name}.jpg")

    def SaveCombinedMask(self):
        img = self.MASKS['face']
        img = ConvertImageToBGR(img)
        combined_mask = np.zeros_like(img)

        for i in self.MASKS:
            mask = self.MASKS[i]
            mask = self.Paint(mask, self.COLORS[i])
            combined_mask += mask

        annotated_frame_pil = Image.fromarray(combined_mask)
        annotated_frame_pil.save(f"{self.COMBINED_MASK_DIR}/{self.image_name}.jpg")

    @staticmethod
    def Paint(image_arr: np.ndarray, color: list):
        """
        :Description:
        Method {RemoveIntersections} removes all intersections of all masks,
        converts new masks to grayscale and saves them in SPLIT_MASK_DIR

        :param image_arr:
        :param color:
        :rtype: ???
        :return: ???
        """
        image = ConvertImageToBGR(image_arr)

        mask = np.any(image >= 50, axis=2)
        blacked = np.any(image < 50, axis=2)

        image[mask] = color
        image[blacked] = [0, 0, 0]

        return image

    def DeleteOtherMasks(self):
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

    def DeleteNoize(self):
        for image in self.MASKS:
            img = self.MASKS[image]
            img = ConvertImageToGRAY(img)

            _, binary_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

            min_size = 500
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    cleaned_mask[labels == i] = 0

            self.MASKS[image] = cleaned_mask


class RemoveIntersections:
    def __init__(self, CLASSES: list, MASKS: dict):
        """
        :param CLASSES: All classes (face parts from FaceSeg self.CLASSES)
        :type CLASSES: list
        :param MASKS: Dictionary with all masks class-wise
        :type MASKS: dict
        """
        self.CLASSES = CLASSES
        self.MASKS = MASKS

    @property
    def MainRemoveIntersections(self) -> dict:
        """
        :Description:
        Property {RemoveIntersections} removes all intersections of all masks and
        converts new masks to cv2.GRAY saving them to self.MASKS

        :rtype: dict
        :return: Dictionary with all cleaned masks class-wise
        """
        intersecting_classes: dict[str, list[str]] = \
            {
                "eyes": ["eyebrows", "nose"],
                "face": ["ears", "eyebrows", "eyes", "glasses", "mouth", "neck", "nose"],
                "glasses": ["eyebrows", "eyes", "nose"],
                "hair": ["ears", "face", "neck"],
                "nose": ["eyebrows", "mouth"]
            }

        for curr_class in self.CLASSES:
            if curr_class in intersecting_classes:
                self.MASKS[curr_class] = self.PerClassMain(curr_class, intersecting_classes[curr_class])
            else:
                self.MASKS[curr_class] = self.AllRest(curr_class)

        return self.MASKS

    def PerClassBase(self, main_part: np.ndarray, sub_part: str) -> np.ndarray:
        """
        :Description:
        Method {PerClassBase} removes intersection of main_part with <sub_part> from <main_part>

        :Inheritance: Method {PerClassBase} is sub-method of {PerClassMain}
        :param main_part: Mask that is being modified
        :type main_part: np.ndarray
        :param sub_part: Name of class that is intersecting with <main_part>
        :type sub_part: np.ndarray
        :rtype: np.ndarray
        :return: main_part class without intersection with sub_part
        """
        instance = self.MASKS[sub_part]
        instance = ConvertImageToGRAY(instance)

        intersection = cv2.bitwise_and(main_part, instance)
        main_part_wi = cv2.bitwise_and(main_part, cv2.bitwise_not(intersection))
        return main_part_wi

    def PerClassMain(self, current_class: str, other_classes: list) -> np.ndarray:
        """
        :Description:
        Method {PerClassMain} removes all intersections of all masks from
        <other_classes> with <current_class>, converts new masks to grayscale and saves
        as SPLIT_MASK_DIR/current_class.jpg

        :keyword class: Face part (from FaceSeg self.CLASSES)

        :param current_class: Name of class that is being modified
        :type current_class: str
        :param other_classes: Names of all classes that are intersecting with <current_class>
        :type other_classes: list
        :rtype: ???
        :return: np.ndarray
        """
        curr_mask = self.MASKS[current_class]
        curr_mask = ConvertImageToGRAY(curr_mask)

        for class_name in other_classes:
            curr_mask = self.PerClassBase(curr_mask, class_name)

        return curr_mask

    def AllRest(self, current_class: str) -> np.ndarray:
        """
        :Description:
        Method {AllRest} converts <current_class> to cv2.GRAY format

        :param current_class: Name of class that is converted to grayscale
        :type current_class: str
        :rtype: np.ndarray
        :return: current_class in cv2.GRAY format
        """
        curr_mask = self.MASKS[current_class]
        curr_mask = ConvertImageToGRAY(curr_mask)

        return curr_mask
