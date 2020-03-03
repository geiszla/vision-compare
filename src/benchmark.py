import os

from utilities import print_debug, initialize_environment
from models_ import (  # pylint: disable=unused-import # noqa: F401
    RetinaNet,
    SqueezeDet,
    SSD,
    YOLOv3,
)


IMAGES_PATH = os.path.abspath('data/COCO/images')
ANNOTATIONS_PATH = os.path.abspath('data/COCO/labels')
VIDEO_PATH = os.path.abspath('data/object_tracking.mp4')


if __name__ == '__main__':
    initialize_environment()

    for Model in [SqueezeDet]:
        Model().evaluate(IMAGES_PATH, VIDEO_PATH, ANNOTATIONS_PATH, 10)

    print_debug('\nExiting...')
