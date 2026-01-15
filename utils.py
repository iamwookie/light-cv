import cv2
import numpy as np
import math


def show_image(image, title="Preview"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_stars(img, centers, size=6, colour=(0, 255, 255), thickness=1):
    for cx, cy in centers:
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


def draw_lines(img, centers, colour=(0, 0, 255), degree=2):
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
        if deg[i] >= degree or deg[j] >= degree:
            continue

        cv2.line(img, centers[i], centers[j], colour, 1)

        deg[i] += 1
        deg[j] += 1


def draw_boxes(
    img,
    boxes,
    *,
    colour=(255, 255, 255),
    thickness=1,
    labels=None,
    label_colour=(255, 255, 255),
    font_scale=0.4,
):
    for i, (x, y, w, h) in enumerate(boxes):
        # draw rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            colour,
            thickness,
            cv2.LINE_AA,
        )

        if labels is None:
            continue

        label = labels[i]

        # place label above box if possible, else below
        tx = x + 2
        ty = y - 4 if y > 12 else y + h + 12

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_colour,
            thickness,
            cv2.LINE_AA,
        )
