from dataclasses import dataclass
from typing import Optional

import cv2
from cv2.typing import MatLike


@dataclass
class LinesOptions:
    """Options for drawing connecting lines."""

    colour: tuple[int, int, int] = (0, 0, 255)
    degree: int = 2
    thickness: int = 1


def draw_frame(
    img: MatLike,
    centers: list[tuple[int, int]],
    *,
    options: Optional[LinesOptions] = None,
) -> None:
    """Draw lines connecting nearest centers."""

    if options is None:
        options = LinesOptions()

    deg = [0] * len(centers)

    pairs = []
    for i, (xi, yi) in enumerate(centers):
        for j in range(i + 1, len(centers)):
            xj, yj = centers[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + dy * dy
            pairs.append((d2, i, j))

    pairs.sort(key=lambda t: t[0])

    for _, i, j in pairs:
        if deg[i] >= options.degree or deg[j] >= options.degree:
            continue

        cv2.line(img, centers[i], centers[j], options.colour, options.thickness)

        deg[i] += 1
        deg[j] += 1
