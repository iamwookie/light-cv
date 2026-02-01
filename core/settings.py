from dataclasses import dataclass
from typing import Callable, Optional, Type, Any, List
from cv2.typing import MatLike
from .algorithms import percentile, otsu, peaks, fixed


@dataclass
class AlgorithmConfig:
    """Configuration for a blob detection algorithm."""

    name: str
    options_class: Type[Any]
    process_frame: Callable[[MatLike, Optional[Any]], Any]
    params: List[str]

    def options(self, **kwargs):
        return self.options_class(**kwargs)


ALGORITHMS = {
    "percentile": AlgorithmConfig(
        name="Percentile",
        process_frame=percentile.process_frame,
        options_class=percentile.PercentileOptions,
        params=["percentile", "blur_size", "clean_size", "min_area", "max_area"],
    ),
    "otsu": AlgorithmConfig(
        name="Otsu",
        process_frame=otsu.process_frame,
        options_class=otsu.OtsuOptions,
        params=["blur_size", "clean_size", "min_area", "max_area"],
    ),
    "peaks": AlgorithmConfig(
        name="Peaks",
        process_frame=peaks.process_frame,
        options_class=peaks.PeaksOptions,
        params=["margin", "blur_size", "clean_size"],
    ),
    "fixed": AlgorithmConfig(
        name="Fixed",
        process_frame=fixed.process_frame,
        options_class=fixed.FixedOptions,
        params=["margin", "blur_size", "clean_size", "min_area", "max_area"],
    ),
}
