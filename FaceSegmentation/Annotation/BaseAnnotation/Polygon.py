import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure


class polygons:
    def __init__(self, binary_mask_path, original_image_path):
        self.binary_mask_path = binary_mask_path
        self.original_image_path = original_image_path
        self.binary_mask = cv2.imread(binary_mask_path, 0)
        self.original_image = cv2.imread(original_image_path)

        self.polygons = []

    @staticmethod
    def close_contour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    def binary_mask_to_polygon(self, tolerance=2):
        plgns = []
        padded_binary_mask = np.pad(self.binary_mask, pad_width=1, mode='constant', constant_values=0)
        blurred_binary_mask = cv2.GaussianBlur(padded_binary_mask, (5, 5), 0)
        contours = measure.find_contours(blurred_binary_mask, 10)

        for contour in contours:
            contour = contour - 1
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 10:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            segmentation = [0 if i < 0 else i for i in segmentation]
            plgns.append(segmentation)

        self.polygons = plgns
        return plgns

    def visualize_polygon_on_image(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        for polygon in self.polygons:
            polygon = np.array(polygon).reshape((-1, 2))
            plt.plot(polygon[:, 0], polygon[:, 1], linewidth=2, color='red')
        plt.show()
