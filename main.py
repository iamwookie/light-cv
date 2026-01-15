import cv2
import numpy as np
from utils import draw_boxes

INPUT_PATH = "vid.mp4"
OUTPUT_PATH = "output.avi"

# controls the sensitivity of detection [0-100] (higher = fewer detections)
PERCENTILE = 95

# controls the minimum area of detected components [pixels^2]
MIN_AREA = 5


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
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh_val = np.percentile(blur, PERCENTILE)
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # fmt: skip

        print("detected contours:", len(contours))

        # -- drawing --

        overlay = frame.copy()

        centers = []
        boxes = []
        labels = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            x, y, w, h = cv2.boundingRect(c)

            cx = x + w // 2
            cy = y + h // 2

            centers.append((cx, cy))
            boxes.append((x, y, w, h))
            labels.append(f"{i} {area:.1f}")

        draw_boxes(
            overlay,
            boxes,
            colour=(0, 0, 255),
            thickness=1,
            labels=labels,
            font_scale=0.2,
        )

        out.write(overlay)

    vid.release()
    out.release()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
