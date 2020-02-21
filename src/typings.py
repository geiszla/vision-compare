from typing import List, Tuple

# Type aliases
PredictionBox = Tuple[int, int, int, int]
PredictionResult = Tuple[List[PredictionBox], List[float], List[int]]

ProcessedBox = Tuple[Tuple[int, int], Tuple[int, int]]
ProcessedResult = Tuple[List[ProcessedBox], List[float], List[str]]

RunnerResult = Tuple[object, ProcessedResult]
