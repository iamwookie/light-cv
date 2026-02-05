from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type, TypedDict

from cv2.typing import MatLike

from .otsu import OtsuOptions, process_frame as otsu_process_frame
from .percentile import PercentileOptions, process_frame as percentile_process_frame
from .peaks import PeaksOptions, process_frame as peaks_process_frame
from .fixed import FixedOptions, process_frame as fixed_process_frame


class AlgorithmMetadata(TypedDict):
    centers: list[tuple[int, int]]
    boxes: list[tuple[int, int, int, int]]
    areas: list[float]
    labels: list[str]


@dataclass
class AlgorithmConfig:
    """Configuration for a blob detection algorithm."""

    name: str
    options_class: Type[Any]
    process_frame: Callable[[MatLike, Optional[Any]], AlgorithmMetadata]
    params: List[str]

    def options(self, **kwargs):
        return self.options_class(**kwargs)


ALGORITHMS = {
    "otsu": AlgorithmConfig(
        name="Otsu",
        process_frame=otsu_process_frame,
        options_class=OtsuOptions,
        params=[
            "blur_size",
            "clean_size",
            "min_area",
            "max_area",
        ],
    ),
    "percentile": AlgorithmConfig(
        name="Percentile",
        process_frame=percentile_process_frame,
        options_class=PercentileOptions,
        params=[
            "percentile",
            "blur_size",
            "clean_size",
            "min_area",
            "max_area",
        ],
    ),
    "peaks": AlgorithmConfig(
        name="Peaks",
        process_frame=peaks_process_frame,
        options_class=PeaksOptions,
        params=[
            "margin",
            "blur_size",
            "clean_size",
        ],
    ),
    "fixed": AlgorithmConfig(
        name="Fixed",
        process_frame=fixed_process_frame,
        options_class=FixedOptions,
        params=[
            "margin",
            "blur_size",
            "clean_size",
            "min_area",
            "max_area",
        ],
    ),
}

__all__ = ["AlgorithmMetadata", "AlgorithmConfig", "ALGORITHMS"]
