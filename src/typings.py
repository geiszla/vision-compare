from typing import Any, Dict, Generator, List, Tuple, Union

import numpy
from nptyping import Array


# Type aliases
PredictionBox = List[float]
PredictionResult = Tuple[List[PredictionBox], List[str], List[float]]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]

# Annotation = Tuple[None, float, float, float, float, None, None, None, None, str]
Annotation = Union[Array[object, None, 10]]
SplittedData = Tuple[List[str], List[str], List[Annotation], List[Annotation]]

Image = Union[Array[numpy.float32, None, None, 3]]
ProcessedImage = Tuple[Image, float]
DataGenerator = Generator[
    Tuple[List[ProcessedImage], Array[Annotation]],
    None,
    None,
]

Statistics = Tuple[List[float], List[float], List[float], List[List[float]]]

FilterResult = Tuple[
    List[PredictionBox],
    List[str],
    List[float],
    List[List[Annotation]],
    Dict[str, Any],
]
