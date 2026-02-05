from dataclasses import dataclass
from typing import Any, Callable, List, Type, Optional

from cv2.typing import MatLike
from ..algorithms import AlgorithmMetadata

from .boxes import BoxesOptions, draw_frame as draw_boxes
from .lines import LinesOptions, draw_frame as draw_lines
from .stars import StarsOptions, draw_frame as draw_stars


@dataclass
class DrawConfig:
    """Configuration for a drawing helper."""

    name: str
    options_class: Type[Any]
    draw: Callable[[MatLike, AlgorithmMetadata, Optional[Any]], None]
    params: List[str]

    def options(self, **kwargs):
        return self.options_class(**kwargs)


DRAWING = {
    "stars": DrawConfig(
        name="Stars",
        options_class=StarsOptions,
        draw=draw_stars,
        params=[
            "size",
            "colour",
            "thickness",
            "label_colour",
            "label_thickness",
            "font_scale",
        ],
    ),
    "boxes": DrawConfig(
        name="Boxes",
        options_class=BoxesOptions,
        draw=draw_boxes,
        params=[
            "colour",
            "thickness",
            "label_colour",
            "label_thickness",
            "font_scale",
        ],
    ),
    "lines": DrawConfig(
        name="Lines",
        options_class=LinesOptions,
        draw=draw_lines,
        params=[
            "colour",
            "degree",
            "thickness",
        ],
    ),
}

__all__ = ["DrawConfig", "DRAWING"]
