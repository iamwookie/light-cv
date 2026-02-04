import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from cv2.typing import MatLike


@dataclass
class StarsOptions:
    """Options for drawing star markers."""

    size: int = 6
    colour: tuple[int, int, int] = (0, 255, 255)
    thickness: int = 1
    label_colour: tuple[int, int, int] = (255, 255, 255)
    label_thickness: int = 1
    font_scale: float = 0.4


def draw_frame(
    img: MatLike,
    centers: list[tuple[int, int]],
    boxes: list[tuple[int, int, int, int]],
    *,
    labels: Optional[list[str]] = None,
    options: Optional[StarsOptions] = None,
) -> None:
    """Draw star markers at the provided centers."""

    if options is None:
        options = StarsOptions()

    assert len(centers) == len(boxes), "centers and boxes must match"

    for i, ((cx, cy), (x, y, w, h)) in enumerate(zip(centers, boxes)):
        inner = options.size * 0.382
        points = []

        for k in range(10):
            angle = k * math.pi / 5 - math.pi / 2
            r = options.size if k % 2 == 0 else inner
            px = int(cx + r * math.cos(angle))
            py = int(cy + r * math.sin(angle))
            points.append((px, py))

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            img,
            [pts],
            True,
            options.colour,
            options.thickness,
            cv2.LINE_AA,
        )

        if labels is None:
            continue

        label = labels[i]

        tx = x + 2
        ty = y - 4 if y > 12 else y + h + 12

        cv2.putText(
            img,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            options.font_scale,
            options.label_colour,
            options.label_thickness,
            cv2.LINE_AA,
        )
