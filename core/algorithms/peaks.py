import cv2
import numpy as np

from cv2.typing import MatLike
from dataclasses import dataclass
from typing import Optional


@dataclass
class PeaksOptions:
    """Options for local peak detection algorithm."""

    margin: float = 0.7  # Intensity threshold as fraction of max [0-1] (higher = fewer detections) # fmt:skip
    blur_size: int = 5  # Size of Gaussian blur kernel (odd number)
    clean_size: int = 3  # Size of morphology kernel for cleaning
    peak_size: int = 15  # Size of kernel for local peak detection


def peaks_thresholding(frame: MatLike, margin=0.7, blur_size=5, clean_size=3, peak_size=15) -> tuple[MatLike, list[tuple[int, int]]]: # fmt: skip
    """
    Apply local peak detection thresholding to a single frame.

    Args:
        frame: Input BGR image
        margin: Intensity threshold as fraction of max [0-1]
        blur_size: Size of Gaussian blur kernel (odd number)
        clean_size: Size of morphology kernel for cleaning
        peak_size: Size of kernel for local peak detection

    Returns:
        thresh: Binary thresholded image
        centers: List of (cx, cy) center coordinates of detected peaks
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    _, max_val, _, _ = cv2.minMaxLoc(blur)
    thresh_val = int(max_val * margin)
    _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (peak_size, peak_size))
    dist_dil = cv2.dilate(dist, peak_kernel)
    local_max = ((dist == dist_dil) & (dist > 0)).astype(np.uint8) * 255

    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max, 8)

    centers = []
    for i in range(1, n_labels):
        cx, cy = centroids[i]
        centers.append((int(cx), int(cy)))

    return thresh, centers


def process_frame(frame: MatLike, options: Optional[PeaksOptions] = None):
    """
    Process a frame using local peak detection.

    Args:
        frame: Input BGR image
        options: PeaksOptions instance with algorithm parameters (uses defaults if None)

    Returns:
        binary_mask: Binary thresholded image
        contours: List of detected contours (empty for this method)
        centers: List of (cx, cy) center coordinates
        metadata: Dictionary with additional info
    """

    if options is None:
        options = PeaksOptions()

    _, centers = peaks_thresholding(
        frame,
        options.margin,
        options.blur_size,
        options.clean_size,
        options.peak_size,
    )

    metadata = {
        "centers": centers,
        "boxes": [],
        "areas": [],
        "labels": [],
    }

    return metadata
