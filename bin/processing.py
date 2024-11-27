import cv2
import numpy as np
from PIL import Image

class ReceiptProcessor:
    def __init__(self):
        self.debug_mode = False

    def detect_receipt(self, image_np: np.array) -> np.array:
        # identify the receipt in the image and crop it
        # image_np = np.array(image)  # image is already grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        edged = cv2.Canny(thresh, 50, 200, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image")

        # Filter contours by area and aspect ratio
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Filter out small contours
                continue

            # Check aspect ratio
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:  # Filter out extremely elongated contours
                continue

            valid_contours.append(cnt)

        if not valid_contours:
            raise ValueError("No valid receipt contours found")

        receipt_contour = max(valid_contours, key=cv2.contourArea)  # largest valid

        peri = cv2.arcLength(receipt_contour, True)
        approx = cv2.approxPolyDP(receipt_contour, 0.02 * peri, True)

        if len(approx) != 4:  # no 4 corners
            rect = cv2.minAreaRect(receipt_contour)
            approx = cv2.boxPoints(rect)

        approx = self._sort_corners(approx)

        src_pts = approx.astype("float32")
        width = int(max(
            np.linalg.norm(src_pts[0] - src_pts[1]),
            np.linalg.norm(src_pts[2] - src_pts[3])
        ))
        height = int(max(
            np.linalg.norm(src_pts[0] - src_pts[3]),
            np.linalg.norm(src_pts[1] - src_pts[2])
        ))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
            ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image_np, M, (width, height))

        return warped
    
    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def detect_brightness_contrast(self, image: np.array):
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate brightness as the mean pixel value
        brightness = np.mean(image)
        # Calculate contrast as the standard deviation of pixel values
        contrast = np.std(image)
        return brightness, contrast

    def preprocess_receipt(self, image: Image) -> np.array:
        image_np = np.array(image)
        
        brightness, contrast = self.detect_brightness_contrast(image_np)
        #print(f"Brightness: {brightness}, Contrast: {contrast}")
        brightness = -(brightness-100)
        #ic(brightness)
        contrast = 1.5
        adjusted_image = cv2.addWeighted(image_np, contrast, np.zeros(image_np.shape, image_np.dtype), 0, brightness)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adjusted_image = cv2.filter2D(adjusted_image, -1, kernel)

        adjusted_image = self.detect_receipt(adjusted_image)
        cv2.imwrite("../data/adjusted_image.png", adjusted_image)
        return adjusted_image