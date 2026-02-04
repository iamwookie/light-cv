from dataclasses import dataclass
from typing import Optional

import cv2
from cv2.typing import MatLike


@dataclass
class BoxesOptions:
    """Options for drawing bounding boxes."""

    colour: tuple[int, int, int] = (255, 255, 255)
    thickness: int = 1
    label_colour: tuple[int, int, int] = (255, 255, 255)
    label_thickness: int = 1
    font_scale: float = 0.4


def draw_frame(
    img: MatLike,
    boxes: list[tuple[int, int, int, int]],
    *,
    labels: Optional[list[str]] = None,
    options: Optional[BoxesOptions] = None,
) -> None:
    """Draw bounding boxes with optional labels."""

    if options is None:
        options = BoxesOptions()

    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
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
