"""Python Typings
Contains the shared python typings for this project
"""

from typing import Any, Generator, List, Tuple, Union

import numpy
from nptyping import NDArray
from PIL.Image import Image as PillowImage

# Type aliases
Box = Union[NDArray[(4,), numpy.float32]]  # type: ignore
Boxes = Union[NDArray[(Any, Any, 4), numpy.float32]]  # type: ignore
Classes = Union[NDArray[(Any, Any), numpy.float32]]  # type: ignore
Scores = Union[NDArray[(Any, Any), numpy.float32]]  # type: ignore

PredictionResult = Tuple[
    NDArray[(Any, 4), numpy.float32],  # type: ignore
    NDArray[(Any,), numpy.int32],  # type: ignore
    NDArray[(Any,), numpy.float32]  # type: ignore
]

Annotation = Union[NDArray[(10,), object]]  # type: ignore
Annotations = Union[NDArray[(Any, 10), object]]  # type: ignore
SplitData = Tuple[List[str], List[str], List[Annotations], List[Annotations]]

ImageData = Union[NDArray[(Any, Any, 3), numpy.float32]]  # type: ignore
Images = Union[NDArray[(Any, Any, Any, 3), numpy.float32]]  # type: ignore

BatchAnnotations = Union[NDArray[(Any, Any, 10), object]]  # type: ignore
Batch = Tuple[List[PillowImage], BatchAnnotations]
ProcessedBatch = Tuple[List[ImageData], List[BatchAnnotations]]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]
StatisticsEntry = Tuple[float, float, float, float]
