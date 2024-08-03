from FaceSegmentation.Pipeline.Config import *


class bboxes():
    def __init__(self, image_path):
        self.image_path = image_path
        self.BBoxes = []
        self.Areas = []

    def GetBboxCoords(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            if w * h > 400:
                self.BBoxes.append([x, y, w, h])

        return self.BBoxes

    def Visualize(self):
        if not self.BBoxes:
            print("No bounding boxes detected \nRun GetBboxCoords")
        else:
            for bbox in self.BBoxes:
                img = cv2.imread(self.image_path)
                result = img.copy()
                x, y, w, h = bbox
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 250), 2)
                sv.plot_image(result, size=(8, 8))
                print("x,y,w,h:", x, y, w, h)

    def Area(self):
        if not self.BBoxes:
            print("No bounding boxes detected \nRun GetBboxCoords")
        else:
            self.Areas = []
            for bbox in self.BBoxes:
                area = bbox[2] * bbox[3]
                self.Areas.append(area)
        return self.Areas
