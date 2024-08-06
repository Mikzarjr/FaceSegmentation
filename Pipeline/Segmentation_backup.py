# from FaceSegmentation.Pipeline.Config import *
# from FaceSegmentation.Pipeline.Tools import *
#
#
# class FaceSeg:
#     def __init__(self, image_path: str):
#         """
#         :param image_path: Path to image desired for segmentation
#         :return: Combined mask in COMBINED_MASK_DIR, All masks for each class separately in SPLIT_MASK_DIR
#         """
#         self.image_path = image_path
#         self.SEG_DIR = f'{MAIN_DIR}/segmentation'
#         self.image_name = GetImageName(image_path)
#         self.WORK_DIR = f'{self.SEG_DIR}/{self.image_name}'
#         self.SPLIT_MASK_DIR = f'{self.WORK_DIR}/split_masks'
#         self.COMBINED_MASK_DIR = f'{self.WORK_DIR}/combined_masks'
#         self.CLASSES = ['face', 'eyebrows', 'eyes', 'hair', 'mouth', 'neck', 'ears', 'nose', 'glasses']
#         self.COLORS = [[255, 0, 0],
#                        [0, 255, 0],
#                        [0, 0, 255],
#                        [255, 255, 0],
#                        [0, 255, 255],
#                        [255, 0, 255],
#                        [255, 165, 0],
#                        [128, 0, 128],
#                        [165, 42, 42],
#                        [191, 255, 0]]
#
#     def Prepare(self):
#         os.makedirs(self.SEG_DIR, exist_ok=True)
#         os.makedirs(self.WORK_DIR, exist_ok=True)
#         os.makedirs(self.SPLIT_MASK_DIR, exist_ok=True)
#         os.makedirs(self.COMBINED_MASK_DIR, exist_ok=True)
#         img = cv2.imread(self.image_path)
#         Image.fromarray(img).save(f"{self.WORK_DIR}/{self.image_name}.jpg")
#
#     def Segment(self) -> object:
#         """
#
#         :rtype: object
#         """
#         self.Prepare()
#         self.SegmentImage()
#         self.RemoveIntersections()
#         self.SaveMask()
#         self.DeleteOtherMasks()
#         self.DeleteNoize()
#
#     def SegmentImage(self):
#         all_results = []
#         image = cv2.imread(self.image_path)
#
#         for i, class_name in enumerate(self.CLASSES):
#             classes_dict = {class_name: class_name}
#             MODEL = ComposedDetectionModel(
#                 detection_model=GroundedSAM(
#                     CaptionOntology(classes_dict)
#                 ),
#                 classification_model=CLIP(
#                     CaptionOntology({class_name: class_name})
#                 )
#             )
#
#             results = MODEL.predict(self.image_path)
#             all_results.append(results)
#
#             annotator = sv.MaskAnnotator()
#             mask = annotator.annotate(scene=np.zeros_like(image), detections=results)
#             Image.fromarray(mask).save(f"{self.SPLIT_MASK_DIR}/{class_name}.jpg")
#
#     def RemoveIntersections(self):
#         for i in self.CLASSES:
#             if i == 'face':
#                 FACE = self.FaceWithoutIntersections()
#                 Image.fromarray(FACE).save(f"{self.SPLIT_MASK_DIR}/face_WI.jpg")
#             elif i == 'eyes':
#                 EYES = self.EyesWithoutIntersections()
#                 Image.fromarray(EYES).save(f"{self.SPLIT_MASK_DIR}/eyes_WI.jpg")
#             elif i == 'nose':
#                 NOSE = self.NoseWithoutIntersections()
#                 Image.fromarray(NOSE).save(f"{self.SPLIT_MASK_DIR}/nose_WI.jpg")
#             elif i == 'glasses':
#                 GLASSES = self.GlassesWithoutIntersections()
#                 Image.fromarray(GLASSES).save(f"{self.SPLIT_MASK_DIR}/glasses_WI.jpg")
#             elif i == 'hair':
#                 HAIR = self.HairWithoutIntersections()
#                 Image.fromarray(HAIR).save(f"{self.SPLIT_MASK_DIR}/hair_WI.jpg")
#             else:
#                 CLASS = self.REST_WithoutIntersection(i)
#                 Image.fromarray(CLASS).save(f"{self.SPLIT_MASK_DIR}/{i}_WI.jpg")
#
#     def DeleteIntersection(self, main_part, sub_part):
#         instance = cv2.imread(f"{self.SPLIT_MASK_DIR}/{sub_part}.jpg", cv2.IMREAD_GRAYSCALE)
#         intersection = cv2.bitwise_and(main_part, instance)
#         main_part_wi = cv2.bitwise_and(main_part, cv2.bitwise_not(intersection))
#         return main_part_wi
#
#     def FaceWithoutIntersections(self):
#         face = cv2.imread(f"{self.SPLIT_MASK_DIR}/face.jpg", cv2.IMREAD_GRAYSCALE)
#
#         classes = ["ears", "neck", "nose", "glasses", "eyes", "eyebrows", "mouth"]
#         for instance in classes:
#             face = self.DeleteIntersection(face, instance)
#
#         return face
#
#     def HairWithoutIntersections(self):
#         hair = cv2.imread(f"{self.SPLIT_MASK_DIR}/hair.jpg", cv2.IMREAD_GRAYSCALE)
#
#         classes = ["face", "neck", "ears"]
#         for instance in classes:
#             hair = self.DeleteIntersection(hair, instance)
#
#         return hair
#
#     def EyesWithoutIntersections(self):
#         eyes = cv2.imread(f"{self.SPLIT_MASK_DIR}/eyes.jpg", cv2.IMREAD_GRAYSCALE)
#
#         classes = ["nose", "eyebrows"]
#         for instance in classes:
#             eyes = self.DeleteIntersection(eyes, instance)
#
#         return eyes
#
#     def GlassesWithoutIntersections(self):
#         glasses = cv2.imread(f"{self.SPLIT_MASK_DIR}/glasses.jpg", cv2.IMREAD_GRAYSCALE)
#
#         classes = ["eyes", "eyebrows", "nose"]
#         for instance in classes:
#             glasses = self.DeleteIntersection(glasses, instance)
#
#         return glasses
#
#     def NoseWithoutIntersections(self):
#         nose = cv2.imread(f"{self.SPLIT_MASK_DIR}/nose.jpg", cv2.IMREAD_GRAYSCALE)
#
#         classes = ["mouth", "eyebrows"]
#         for instance in classes:
#             nose = self.DeleteIntersection(nose, instance)
#
#         return nose
#
#     def REST_WithoutIntersection(self, CLASS):
#         CLASS = cv2.imread(f"{self.SPLIT_MASK_DIR}/{CLASS}.jpg", cv2.IMREAD_GRAYSCALE)
#         return CLASS
#
#     @staticmethod
#     def Paint(image_path, color):
#         image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
#         mask = np.any(image >= 50, axis=2)
#         blacked = np.any(image < 50, axis=2)
#
#         image[mask] = color
#         image[blacked] = [0, 0, 0]
#
#         return image
#
#     def SaveMask(self):
#         img = cv2.imread(f'{self.SPLIT_MASK_DIR}/face_WI.jpg')
#         combined_mask = np.zeros_like(img)
#
#         for i in range(len(self.CLASSES)):
#             path = f"{self.SPLIT_MASK_DIR}/{self.CLASSES[i]}_WI.jpg"
#             mask = self.Paint(path, self.COLORS[i])
#             combined_mask += mask
#
#         annotated_frame_pil = Image.fromarray(combined_mask)
#         annotated_frame_pil.save(f"{self.COMBINED_MASK_DIR}/{self.image_name}.jpg")
#
#     def DeleteOtherMasks(self):
#         for file in os.listdir(self.SPLIT_MASK_DIR):
#             if "WI" not in file:
#                 file_path = os.path.join(self.SPLIT_MASK_DIR, file)
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#
#         for file in os.listdir(self.SPLIT_MASK_DIR):
#             if "WI" in file:
#                 new_file_name = file.replace("_WI", "")
#                 old_file_path = os.path.join(self.SPLIT_MASK_DIR, file)
#                 new_file_path = os.path.join(self.SPLIT_MASK_DIR, new_file_name)
#                 if os.path.isfile(old_file_path):
#                     os.rename(old_file_path, new_file_path)
#
#     def DeleteNoize(self):
#         for image in os.listdir(self.SPLIT_MASK_DIR):
#             image_path = f"{self.SPLIT_MASK_DIR}/{image}"
#             img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#             _, binary_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
#
#             kernel = np.ones((3, 3), np.uint8)
#             cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
#             num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
#
#             min_size = 500
#             for i in range(1, num_labels):
#                 if stats[i, cv2.CC_STAT_AREA] < min_size:
#                     cleaned_mask[labels == i] = 0
#
#             Image.fromarray(cleaned_mask).save(image_path)
