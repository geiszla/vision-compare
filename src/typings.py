"""Python Typings
Contains the shared python typings for this project
"""

from typing import Generator, List, Tuple, Union

import numpy
from nptyping import Array
from PIL.Image import Image as PillowImage

# Type aliases
Box = Union[Array[numpy.float32, 4]]  # type: ignore
Boxes = Union[Array[numpy.float32, None, None, 4]]  # type: ignore
Classes = Union[Array[numpy.float32, None, None]]  # type: ignore
Scores = Union[Array[numpy.float32, None, None]]  # type: ignore

PredictionResult = Tuple[
    Array[numpy.float32, None, 4],  # type: ignore
    Array[numpy.int32, None],  # type: ignore
    Array[numpy.float32, None]  # type: ignore
]

Annotation = Union[Array[object, 10]]  # type: ignore
Annotations = Union[Array[object, None, 10]]  # type: ignore
SplitData = Tuple[List[str], List[str], List[Annotations], List[Annotations]]

ImageData = Union[Array[numpy.float32, None, None, 3]]  # type: ignore
Images = Union[Array[numpy.float32, None, None, None, 3]]  # type: ignore

BatchAnnotations = Union[Array[object, None, None, 10]]  # type: ignore
Batch = Tuple[List[PillowImage], BatchAnnotations]
ProcessedBatch = Tuple[List[ImageData], List[BatchAnnotations]]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]
StatisticsEntry = Tuple[float, float, float, float]
