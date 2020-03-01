from typing import Any, Dict, Generator, List, Tuple, TypeVar, Union

import numpy
from nptyping import Array


# Type aliases
Box = Union[Array[numpy.float32, 4]]
Boxes = Union[Array[numpy.float32, ..., 4]]
Classes = Union[Array[numpy.float32, ...]]
Scores = Union[Array[numpy.float32, ...]]

PredictionResult = Tuple[Array[numpy.float32, ..., 4], Classes, Scores]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]

Annotation = Union[Array[object, 10]]
Annotations = Union[Array[object, ..., 10]]
SplittedData = Tuple[List[str], List[str], List[Annotations], List[Annotations]]

ImageType = TypeVar('ImageType')
ProcessedImageType = TypeVar('ProcessedImageType')

ImageData = Union[Array[numpy.float32, ..., ..., 3]]
ResizedImage = Tuple[ImageType, float]

BatchAnnotations = Union[Array[object, ..., ..., 10]]
Batch = Tuple[List[ImageType], BatchAnnotations]
ProcessedBatch = Tuple[List[ProcessedImageType], BatchAnnotations]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]

FilterResult = Tuple[
    List[List[float]],
    List[str],
    List[float],
    List[List[Annotations]],
    Dict[str, Any],
]
