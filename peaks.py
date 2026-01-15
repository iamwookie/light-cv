import cv2
import numpy as np
import math

INPUT_PATH = "vid.mp4"
OUTPUT_PATH = "output.avi"

# controls the sensitivity of detection
MARGIN = 0.7

# controls the size of the blur, to reduce noise
BLUR_SIZE = 5

# controls the size of the cleaning morphology
CLEAN_SIZE = 3

# controls the size of the local peak detection
PEAK_SIZE = 15

# controls the area within which to merge points
MERGE_AREA = 25

# maximum degree of each peak
LINE_DEGREE = 2


# utility only
def show_image(image, title="Preview"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_star(img, center, size=6, colour=(0, 255, 255), thickness=1):
    cx, cy = center

    # star inner/outer ratio (~0.382)
    inner = size * 0.382

    points = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2  # point up
        r = size if i % 2 == 0 else inner
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        points.append((x, y))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, colour, thickness)


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
        blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        _, max_val, _, _ = cv2.minMaxLoc(blur)
        thresh = int(max_val * MARGIN)

        _, mask = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLEAN_SIZE, CLEAN_SIZE))
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
        peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PEAK_SIZE, PEAK_SIZE)) # fmt: skip
        dist_dil = cv2.dilate(dist, peak_kernel)
        local_max = ((dist == dist_dil) & (dist > 0)).astype(np.uint8) * 255

        n_labels, _, _, centroids = cv2.connectedComponentsWithStats(local_max, 8)

        print("Detected peaks:", n_labels - 1)

        centers = []
        for i in range(1, n_labels):
            cx, cy = centroids[i]
            centers.append((int(cx), int(cy)))

        # -- drawing peaks --

        overlay = frame.copy()

        for cx, cy in centers:
            draw_star(overlay, center=(cx, cy), color=(0, 255, 255))

        # -- drawing connections --

        deg = [0] * len(centers)

        pairs = []
        for i, (xi, yi) in enumerate(centers):
            for j in range(i + 1, len(centers)):
                xj, yj = centers[j]
                dx = xi - xj
                dy = yi - yj
                d2 = dx * dx + dy * dy
                pairs.append((d2, i, j))

        pairs.sort(key=lambda t: t[0])  # closest first

        for d2, i, j in pairs:
            if deg[i] >= LINE_DEGREE or deg[j] >= LINE_DEGREE:
                continue

            cv2.line(overlay, centers[i], centers[j], (248, 243, 221), 1)

            deg[i] += 1
            deg[j] += 1

        out.write(overlay)

    vid.release()
    out.release()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
