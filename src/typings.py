from typing import Generator, List, Tuple

# Type aliases
PredictionBox = Tuple[int, int, int, int]
PredictionResult = Tuple[List[PredictionBox], List[float], List[int]]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]

Annotation = Tuple[str, int, int, int, int, str]
SplittedData = Tuple[List[str], List[str], List[Annotation], List[Annotation]]
DataGenerator = Generator[Tuple[List[str], List[Annotation]], None, None]

EvaluationResult = Tuple[List[float], List[float], List[float], List[float]]
