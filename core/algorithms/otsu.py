import cv2

from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Optional


@dataclass
class OtsuOptions:
    """Options for Otsu's automatic thresholding algorithm."""

    blur_size: int = 5  # Size of Gaussian blur kernel (odd number)
    clean_size: int = 3  # Size of morphology kernel for cleaning
    min_area: int = 2  # Minimum contour area in pixels^2
    max_area: int = 86400  # Maximum contour area in pixels^2


def otsu_thresholding(frame: MatLike, blur_size=5, clean_size=3) -> MatLike:
    """
    Apply Otsu's automatic thresholding to a single frame.

    Args:
        frame: Input BGR image
        blur_size: Size of Gaussian blur kernel (odd number)
        clean_size: Size of morphology kernel for cleaning

    Returns:
        thresh: Binary thresholded image
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return thresh


def process_frame(frame: MatLike, options: Optional[OtsuOptions] = None):
    """
    Process a frame using Otsu's automatic thresholding.

    Args:
        frame: Input BGR image
        options: OtsuOptions instance with algorithm parameters (uses defaults if None)

    Returns:
        binary_mask: Binary thresholded image
        contours: List of detected contours
        centers: List of (cx, cy) center coordinates
        metadata: Dictionary with additional info (areas, bounding boxes, etc.)
    """

    if options is None:
        options = OtsuOptions()

    thresh = otsu_thresholding(frame, options.blur_size, options.clean_size)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    boxes = []
    areas = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < options.min_area or area > options.max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2

        centers.append((cx, cy))
        boxes.append((x, y, w, h))
        areas.append(area)

    metadata = {
        "centers": centers,
        "boxes": boxes,
        "areas": areas,
        "labels": [f"{i} {a:.2f}" for i, a in enumerate(areas)],
    }

    return thresh, contours, centers, metadata
