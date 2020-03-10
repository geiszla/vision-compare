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


IMAGES_PATH = os.path.abspath('data/COCO/images')
ANNOTATIONS_PATH = os.path.abspath('data/COCO/labels')
VIDEO_PATH = os.path.abspath('data/object_tracking.mp4')


SAMPLE_COUNT = 10


def evaluate_model(model: Detector) -> Tuple[StatisticsEntry, float]:
    [accuracy_statistics], fps = model.evaluate(IMAGES_PATH, VIDEO_PATH, ANNOTATIONS_PATH,
        SAMPLE_COUNT)

    return accuracy_statistics, fps


if __name__ == '__main__':
    initialize_environment()

    # SSD().evaluate_performance(0, is_display=True)

    MODELS = [
        'RetinaNet',
        'SSD TFLite v2',
        'SSD TFLite v1',
        'YOLOv3',
        'SqueezeDet',
    ]

    STATISTICS = [
        # evaluate_model(RetinaNet()),
        evaluate_model(SSDTFLite('v2')),
        evaluate_model(SSDTFLite('v1')),
        evaluate_model(YOLOv3()),
        evaluate_model(SqueezeDet()),
    ]

    AGGREGATED_STATISTICS = [[MODELS[index], *statistics, fps]
        for index, (statistics, fps) in enumerate(STATISTICS)]

    print_debug('')
    print_debug(tabulate(
        AGGREGATED_STATISTICS,
        headers=['Model', 'Precision', 'Recall', 'F1 Score', 'mAP', 'FPS']
    ))

    print_debug('\nExiting...')
