import cv2

from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Optional


@dataclass
class FixedOptions:
    """Options for fixed threshold algorithm."""

    margin: float = 0.7 # Intensity threshold as fraction of max [0-1] (higher = fewer detections) # fmt:skip
    blur_size: int = 7  # Size of Gaussian blur kernel (odd number)
    clean_size: int = 3  # Size of morphology kernel for cleaning
    min_area: int = 5  # Minimum contour area in pixels^2
    max_area: int = 86400  # Maximum contour area in pixels^2


def fixed_thresholding(frame: MatLike, margin=0.7, blur_size=7, clean_size=3) -> MatLike: # fmt: skip
    """
    Apply fixed threshold with margin to a single frame.

    Args:
        frame: Input BGR image
        margin: Intensity threshold as fraction of max [0-1]
        blur_size: Size of Gaussian blur kernel (odd number)
        clean_size: Size of morphology kernel for cleaning

    Returns:
        thresh: Binary thresholded image
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    _, max_val, _, _ = cv2.minMaxLoc(blur)
    thresh_val = int(max_val * margin)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return thresh


def process_frame(frame: MatLike, options: Optional[FixedOptions] = None):
    """
    Process a frame using fixed threshold with margin.

    Args:
        frame: Input BGR image
        options: FixedOptions instance with algorithm parameters (uses defaults if None)

    Returns:
        binary_mask: Binary thresholded image
        contours: List of detected contours
        centers: List of (cx, cy) center coordinates
        metadata: Dictionary with additional info (areas, bounding boxes, etc.)
    """
    if options is None:
        options = FixedOptions()

    thresh = fixed_thresholding(frame, options.margin, options.blur_size, options.clean_size) # fmt: skip
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
        "labels": [f"{i} {a:.1f}" for i, a in enumerate(areas)],
    }

    return thresh, contours, centers, metadata
