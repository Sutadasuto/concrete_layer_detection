import cv2
import numpy as np


class Bilateral_Filtered_Canny(object):

    def __init__(self, filter_parameters, canny_parameters):
        self.filter_parameters = filter_parameters
        self.canny_parameters = canny_parameters

    def predict(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
        # gaussian_blur = cv2.GaussianBlur(gray, (31, 31), sigmaX=15)
        # gray = gray - gaussian_blur
        # gray = (gray - gray.mean() + 127.0)
        # gray = (255*(gray - gray.min())/(gray.max() - gray.min())).astype(np.uint8)
        filtered_image = cv2.bilateralFilter(gray.astype(np.uint8), **self.filter_parameters)
        edges = cv2.Canny(filtered_image, **self.canny_parameters)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        lines = cv2.ximgproc.thinning(opened)

        pred = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
        pred = pred.astype(np.float)/255.0
        return pred[..., None]
