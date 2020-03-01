from typing import Any, Dict, Generator, List, Tuple, TypeVar, Union

from nptyping import Array


# Type aliases
PredictionResult = Tuple[Array, Array, Array]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]

# Annotation = Tuple[None, float, float, float, float, None, None, None, None, str]
Annotation = Union[Array]
SplittedData = Tuple[List[str], List[str], List[Annotation], List[Annotation]]

ImageType = TypeVar('ImageType')
ProcessedImageType = TypeVar('ProcessedImageType')

ImageData = Union[Array]
ResizedImage = Tuple[ImageType, float]

Batch = Tuple[List[ImageType], Array]
ProcessedBatch = Tuple[List[ProcessedImageType], Array]

DataGenerator = Generator[Batch, None, None]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]

FilterResult = Tuple[
    List[List[float]],
    List[str],
    List[float],
    List[List[Annotation]],
    Dict[str, Any],
]
