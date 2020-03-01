from typing import Any, Dict, Generator, List, Tuple, TypeVar, Union

import numpy
from nptyping import Array


# Type aliases
PredictionBox = List[float]
PredictionResult = Tuple[
    Array[numpy.float32, None, 4],
    Array[numpy.int32, None],
    Array[numpy.float32, None],
]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]

# Annotation = Tuple[None, float, float, float, float, None, None, None, None, str]
Annotation = Union[Array[object, None, 10]]
SplittedData = Tuple[List[str], List[str], List[Annotation], List[Annotation]]

ImageType = TypeVar('ImageType')
ProcessedImageType = TypeVar('ProcessedImageType')

ImageData = Union[Array[numpy.float32, None, None, 3]]
ResizedImage = Tuple[ImageType, float]

Batch = Tuple[List[ImageType], Array[Annotation]]
ProcessedBatch = Tuple[List[ProcessedImageType], Array[Annotation]]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]

FilterResult = Tuple[
    List[PredictionBox],
    List[str],
    List[float],
    List[List[Annotation]],
    Dict[str, Any],
]
