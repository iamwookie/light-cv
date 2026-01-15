import cv2
import numpy as np

from utils import draw_stars, draw_lines

INPUT_PATH = "vid.mp4"
OUTPUT_PATH = "output.avi"

# controls the sensitivity of detection
MARGIN = 0.7

# controls the size of the blur, to reduce noise
BLUR_SIZE = (5, 5)

# controls the size of the cleaning morphology
CLEAN_SIZE = (3, 3)

# controls the size of the local peak detection
PEAK_SIZE = (15, 15)

# controls the area within which to merge points
MERGE_AREA = 25


def main():
    vid = cv2.VideoCapture(INPUT_PATH)
    if not vid.isOpened():
        raise RuntimeError("Failed to read input video")

    fps = vid.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # default fps

    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Failed to create VideoWriter")

    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(n_frames):
        ok, frame = vid.read()
        if not ok:
            raise RuntimeError("Failed to read frame")

        # -- processing --

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

        _, max_val, _, _ = cv2.minMaxLoc(blur)
        thresh = int(max_val * MARGIN)

        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLEAN_SIZE)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PEAK_SIZE) # fmt: skip
        dist_dil = cv2.dilate(dist, peak_kernel)
        local_max = ((dist == dist_dil) & (dist > 0)).astype(np.uint8) * 255

        n_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max, 8)

        print("detected peaks:", n_labels - 1)

        centers = []
        for i in range(1, n_labels):
            cx, cy = centroids[i]
            centers.append((int(cx), int(cy)))

        # -- drawing --

        overlay = frame.copy()

        draw_stars(overlay, centers, colour=(0, 255, 255))
        draw_lines(overlay, centers, colour=(248, 243, 221), degree=2)

        out.write(overlay)

    vid.release()
    out.release()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
