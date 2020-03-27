"""Evaluation benchmark script
This script evaluates the computer vision models defined in src/models_

To be run from the project root (i.e. `python src/benchmark.py`)
"""

import os
from typing import Tuple

from tabulate import tabulate

from models_ import (  # pylint: disable=unused-import # noqa: F401
    Detector,
    RetinaNet,
    SSDTFLite,
    YOLOv3,
    SqueezeDet,
)
from typings import StatisticsEntry
from utilities import print_debug, initialize_environment

# Image and video data paths for evaluation
IMAGES_PATH = os.path.abspath('data/COCO/images')
ANNOTATIONS_PATH = os.path.abspath('data/COCO/labels')
VIDEO_PATH = os.path.abspath('data/object_tracking.mp4')

# Number of samples to evaluate the models on
SAMPLE_COUNT = 100


def __evaluate_model(model: Detector) -> Tuple[StatisticsEntry, float]:
    [accuracy_statistics], fps = model.evaluate(IMAGES_PATH, VIDEO_PATH, ANNOTATIONS_PATH,
        SAMPLE_COUNT)

    return accuracy_statistics, fps


if __name__ == '__main__':
    initialize_environment()

    # Run detection on video displaying the bounding boxes on it
    SSDTFLite('v2').evaluate_performance(0, is_display=True)

    MODELS = [
        'RetinaNet',
        'SSD TFLite v2',
        'SSD TFLite v1',
        'YOLOv3',
        'SqueezeDet',
    ]

    # Evaluate the models one-by-one
    STATISTICS = [
        __evaluate_model(RetinaNet()),
        __evaluate_model(SSDTFLite('v2')),
        __evaluate_model(SSDTFLite('v1')),
        __evaluate_model(YOLOv3()),
        __evaluate_model(SqueezeDet()),
    ]

    # Convert the statistics to the correct format to be passed to tabulate
    AGGREGATED_STATISTICS = [[MODELS[index], *statistics, fps]
        for index, (statistics, fps) in enumerate(STATISTICS)]

    # Show evaluation results in a table in command line
    print_debug('')
    print_debug(tabulate(
        AGGREGATED_STATISTICS,
        headers=['Model', 'Precision', 'Recall', 'F1 Score', 'mAP', 'FPS']
    ))

    print_debug('\nExiting...')
