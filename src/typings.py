from typing import Generator, List, Tuple, Union

import numpy
from nptyping import Array
from PIL.Image import Image

# Type aliases
Box = Union[Array[numpy.float32, 4]]
Boxes = Union[Array[numpy.float32, ..., 4]]
Classes = Union[Array[numpy.float32, ...]]
Scores = Union[Array[numpy.float32, ...]]

PredictionResult = Tuple[Array[numpy.float32, 4], numpy.int32, numpy.float32]

Annotation = Union[Array[object, 10]]
Annotations = Union[Array[object, ..., 10]]
SplittedData = Tuple[List[str], List[str], List[Annotations], List[Annotations]]

ImageData = Union[Array[numpy.float32, ..., ..., 3]]

BatchAnnotations = Union[Array[object, ..., ..., 10]]
Batch = Tuple[List[Image], BatchAnnotations]
ProcessedBatch = Tuple[List[ImageData], List[BatchAnnotations]]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]
StatisticsEntry = Tuple[float, float, float, float]
