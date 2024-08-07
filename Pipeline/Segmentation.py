import numpy as np
from FaceSegmentation.Pipeline.Config import *
from FaceSegmentation.Pipeline.Tools import *


class FaceSeg:
    def __init__(self, image_path: str):
        """
        TODO: Finish all dockstrings
        
        :param image_path: Path to image desired for segmentation
        :return: Combined mask in COMBINED_MASK_DIR, All masks for each class separately in SPLIT_MASK_DIR
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

    def Segment(self):
        """

        :rtype: ???
        """
        self.Prepare()
        self.SegmentImage()
        self.RemoveIntersections()
        self.DeleteNoize()
        self.CombinedMask()

    def Prepare(self):
        os.makedirs(self.SEG_DIR, exist_ok=True)
        os.makedirs(self.WORK_DIR, exist_ok=True)
        os.makedirs(self.SPLIT_MASK_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_MASK_DIR, exist_ok=True)
        img = cv2.imread(self.image_path)
        Image.fromarray(img).save(f"{self.WORK_DIR}/{self.image_name}.jpg")

    def SegmentImage(self):
        """
        :rtype: ???
        """

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

            self.MASKS[class_name] = mask

    def RemoveIntersections(self) -> None:
        """
        :Description:
        Method {RemoveIntersections} removes all intersections of all masks,
        converts new masks to grayscale and saves them in SPLIT_MASK_DIR

        :rtype: None
        """
        RI = RemoveIntersections(self.SPLIT_MASK_DIR, self.CLASSES, self.MASKS)
        self.MASKS = RI.RemoveIntersections

    def SaveMasks(self):
        for i in self.MASKS:
            Image.fromarray(self.MASKS[i]).save(f"{self.SPLIT_MASK_DIR}/{i}_new.jpg")

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
        if len(image_arr.shape) == 2:
            image = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2BGR)
        else:
            image = image_arr

        mask = np.any(image >= 50, axis=2)
        blacked = np.any(image < 50, axis=2)

        image[mask] = color
        image[blacked] = [0, 0, 0]

        return image

    def CombinedMask(self):
        img = self.MASKS['face']
        print("shapes:\n", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print(img.shape)
        combined_mask = np.zeros_like(img)
        print(combined_mask.shape)

        for i in self.MASKS:
            mask = self.MASKS[i]
            print("shape of mask before coloring", mask.shape)
            mask = self.Paint(mask, self.COLORS[i])
            print("shape of mask after coloring", mask.shape)
            combined_mask += mask

        annotated_frame_pil = Image.fromarray(combined_mask)
        annotated_frame_pil.save(f"{self.COMBINED_MASK_DIR}/{self.image_name}.jpg")

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
        for image in os.listdir(self.SPLIT_MASK_DIR):
            if image.startswith('.'):
                print(f"Skipping hidden file or directory: {image}")
                continue

            image_path = f"{self.SPLIT_MASK_DIR}/{image}"
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            _, binary_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=3)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

            min_size = 500
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    cleaned_mask[labels == i] = 0

            Image.fromarray(cleaned_mask).save(image_path)


class RemoveIntersections:
    def __init__(self, CLASSES: list, MASKS: dict) -> None:
        """
        :Description:
        Class :RemoveIntersections: removes all intersections of all masks,
        converts new masks to grayscale and saves them in SPLIT_MASK_DIR

        :param CLASSES: All classes (face parts from FaceSeg self.CLASSES)
        :type CLASSES: list
        :param MASKS: Dictionary with all masks class-wise
        :type MASKS: dict
        :rtype: None
        """
        self.CLASSES = CLASSES
        self.MASKS = MASKS

    @property
    def RemoveIntersections(self) -> dict:
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
        instance = self.ConvertImageToGRAY(instance)

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
        curr_mask = self.ConvertImageToGRAY(curr_mask)

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
        curr_mask = self.ConvertImageToGRAY(curr_mask)

        return curr_mask

    @staticmethod
    def ConvertImageToGRAY(image: np.ndarray) -> np.ndarray:
        """
        :Description:
        StaticMethod {ConvertImageToGRAY} converts the image in cv2.GRAY format.

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

    @staticmethod
    def ConvertImageToBGR(image: np.ndarray) -> np.ndarray:
        """
        :Description:
        StaticMethod {ConvertImageToBGR} converts the image in cv2.BGR format.

        :param image: Original image
        :type image: np.ndarray
        :rtype: np.ndarray
        :return: Image in cv2.BGR
        """
        if image is None:
            raise ValueError("The image is None.")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2:
            return image
        else:
            raise ValueError("Invalid image format.")
