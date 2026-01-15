import cv2
from utils import draw_stars, draw_lines

INPUT_PATH = "vid.mp4"
OUTPUT_PATH = "output.avi"

# controls the sensitivity of detection
MARGIN = 0.7

# controls the minimum area of detected components
MIN_AREA = 50

# controls the aspect ratio limits
MIN_ASPECT = 0.6
MAX_ASPECT = 1.4


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

        _, max_val, _, _ = cv2.minMaxLoc(blur)
        thresh = int(max_val * MARGIN)
        print(f"max_val={max_val:.0f}  thresh={thresh}")

        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # -- drawing --

        overlay = frame.copy()

        centers = []
        for i in range(1, n_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w == 0 or h == 0:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue

            cx, cy = x + w // 2, y + h // 2

            centers.append((cx, cy))

        draw_stars(overlay, centers, colour=(0, 255, 255))
        draw_lines(overlay, centers, colour=(248, 243, 221), degree=2)

        out.write(overlay)

    vid.release()
    out.release()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
