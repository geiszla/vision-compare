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

# Number of samples to evaluate the models on (doesn't affect performance benchmark)
SAMPLE_COUNT = 100


def __evaluate_model(model: Detector) -> Tuple[StatisticsEntry, float]:
    # Evaluate the given model and return statistics in proper format
    [accuracy_statistics], fps = model.evaluate(IMAGES_PATH, ANNOTATIONS_PATH, SAMPLE_COUNT,
        VIDEO_PATH)

    return accuracy_statistics, fps


def __evaluate_models():
    # Uncomment the line below to disable GPU in tensorflow, so that models can be benchmarked on
    # CPU only
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    initialize_environment()

    # Uncomment line below to run detection on video displaying the bounding boxes on it
    # SSDTFLite('v2').evaluate_performance(0, is_display=True)

    models = [
        'RetinaNet',
        'SSD TFLite v2',
        'SSD TFLite v1',
        'YOLOv3',
        'SqueezeDet',
    ]

    # Evaluate accuracy of models one-by-one
    statistics = [
        __evaluate_model(RetinaNet()),
        __evaluate_model(SSDTFLite('v2')),
        __evaluate_model(SSDTFLite('v1')),
        __evaluate_model(YOLOv3()),
        __evaluate_model(SqueezeDet()),
    ]

    # Convert the statistics to the correct format to be passed to the tabulate library
    aggregated_statistics = [[models[index], *current_statistics, fps]
        for index, (current_statistics, fps) in enumerate(statistics)]

    # Show evaluation results in a table in command line
    print_debug('')
    print_debug(tabulate(
        aggregated_statistics,
        headers=['Model', 'Precision', 'Recall', 'F1 Score', 'mAP', 'FPS']
    ))

    print_debug('\nExiting...')


if __name__ == '__main__':
    __evaluate_models()
