from typing import Generator, List, Tuple, Union

import numpy
from nptyping import Array
from PIL.Image import Image

# Type aliases
Box = Union[Array[numpy.float32, 4]]  # type: ignore
Boxes = Union[Array[numpy.float32, None, 4]]  # type: ignore
Classes = Union[Array[numpy.float32, None]]  # type: ignore
Scores = Union[Array[numpy.float32, None]]  # type: ignore

PredictionResult = Tuple[Array[numpy.float32, 4], numpy.int32, numpy.float32]  # type: ignore

Annotation = Union[Array[object, 10]]  # type: ignore
Annotations = Union[Array[object, None, 10]]  # type: ignore
SplitData = Tuple[List[str], List[str], List[Annotations], List[Annotations]]

ImageData = Union[Array[numpy.float32, None, None, 3]]  # type: ignore

BatchAnnotations = Union[Array[object, None, None, 10]]  # type: ignore
Batch = Tuple[List[Image], BatchAnnotations]
ProcessedBatch = Tuple[List[ImageData], List[BatchAnnotations]]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]
StatisticsEntry = Tuple[float, float, float, float]
