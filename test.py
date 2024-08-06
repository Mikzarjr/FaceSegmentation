import cv2
instance1 = cv2.imread(f"FaceSegmentation/docks/TestImages/img1.jpeg", cv2.IMREAD_GRAYSCALE)
instance2 = cv2.imread(f"FaceSegmentation/docks/TestImages/img2.jpeg", cv2.IMREAD_GRAYSCALE)
print(type(instance1))
intersection = cv2.bitwise_and(instance1, instance2)
print(type(instance2))
