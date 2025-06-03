import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ReceiptProcessor:
    """
    Processes receipt images: detection, cropping, preprocessing, and analysis.
    Parameters can be configured at initialization.
    """
    def __init__(
        self,
        debug_mode: bool = False,
        blur_kernel: Tuple[int, int] = (5, 5),
        adaptive_thresh_blocksize: int = 11,
        adaptive_thresh_C: int = 2,
        canny_thresh1: int = 50,
        canny_thresh2: int = 200,
        morph_kernel: Tuple[int, int] = (3, 3),
        min_contour_area: int = 1000,
        max_aspect_ratio: float = 5.0,
        contrast: float = 1.5,
        brightness_offset: float = 100.0
    ):
        """
        Initialize the processor with configurable parameters.
        """
        self.debug_mode = debug_mode
        self.blur_kernel = blur_kernel
        self.adaptive_thresh_blocksize = adaptive_thresh_blocksize
        self.adaptive_thresh_C = adaptive_thresh_C
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.morph_kernel = morph_kernel
        self.min_contour_area = min_contour_area
        self.max_aspect_ratio = max_aspect_ratio
        self.contrast = contrast
        self.brightness_offset = brightness_offset

    def detect_receipt(self, image_np: np.ndarray) -> np.ndarray:
        """
        Detect and crop the receipt from the input image.
        Args:
            image_np: Input image as a numpy array (RGB or grayscale).
        Returns:
            Cropped and perspective-corrected receipt image as numpy array.
        Raises:
            ValueError: If no valid receipt contour is found.
        """
        # Convert to grayscale if needed (assume input is RGB or grayscale)
        if len(image_np.shape) == 3:
            # Explicitly convert RGB to GRAY
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        if self.debug_mode:
            cv2.imwrite("../data/debug_gray.png", gray)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        if self.debug_mode:
            cv2.imwrite("../data/debug_blurred.png", blurred)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.adaptive_thresh_blocksize, self.adaptive_thresh_C)
        if self.debug_mode:
            cv2.imwrite("../data/debug_thresh.png", thresh)
        edged = cv2.Canny(thresh, self.canny_thresh1, self.canny_thresh2, apertureSize=3)
        if self.debug_mode:
            cv2.imwrite("../data/debug_edged.png", edged)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        edged = cv2.dilate(edged, kernel, iterations=1)
        if self.debug_mode:
            cv2.imwrite("../data/debug_dilated.png", edged)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image")
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.max_aspect_ratio:
                continue
            valid_contours.append(cnt)
        if not valid_contours:
            raise ValueError("No valid receipt contours found")
        receipt_contour = max(valid_contours, key=cv2.contourArea)
        peri = cv2.arcLength(receipt_contour, True)
        approx = cv2.approxPolyDP(receipt_contour, 0.02 * peri, True)
        if len(approx) != 4:
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
        if self.debug_mode:
            cv2.imwrite("../data/debug_warped.png", warped)
        return warped

    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Sort the four corners of a quadrilateral in order: top-left, top-right, bottom-right, bottom-left.
        Args:
            pts: Array of four points.
        Returns:
            Sorted array of four points.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_brightness_contrast(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Calculate brightness and contrast of an image.
        Args:
            image: Input image as numpy array (RGB or grayscale).
        Returns:
            Tuple of (brightness, contrast).
        """
        if len(image.shape) == 3:
            # Explicitly convert BGR to GRAY (OpenCV default is BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(image))
        contrast = float(np.std(image))
        return brightness, contrast

    def preprocess_receipt(self, image: Image) -> np.ndarray:
        """
        Preprocess the receipt image: adjust brightness/contrast, sharpen, and crop.
        Args:
            image: PIL Image (assumed RGB).
        Returns:
            Preprocessed receipt image as numpy array.
        Raises:
            ValueError: If receipt cannot be detected.
        """
        image_np = np.array(image)
        # Ensure image is in RGB for consistent processing
        if image_np.shape[-1] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[-1] == 3:
            pass  # Already RGB
        else:
            raise ValueError("Input image must have 3 (RGB) or 4 (RGBA) channels")
        brightness, _ = self.detect_brightness_contrast(image_np)
        brightness_adj = -(brightness - self.brightness_offset)
        adjusted_image = cv2.addWeighted(
            image_np, self.contrast, np.zeros(image_np.shape, image_np.dtype), 0, brightness_adj)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adjusted_image = cv2.filter2D(adjusted_image, -1, kernel)
        if self.debug_mode:
            cv2.imwrite("../data/adjusted_image.png", adjusted_image)
        try:
            adjusted_image = self.detect_receipt(adjusted_image)
        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] Error in detect_receipt: {e}")
            raise
        return adjusted_image
