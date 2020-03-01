import os

from utilities import print_debug, initialize_environment
from models_ import (  # pylint: disable=unused-import # noqa: F401
    YOLOv3,
    SqueezeDet,
    SSD,
    SSDv1,
    RetinaNet,
)


CLASS_NAMES = ['person']

IMAGES_PATH = os.path.abspath('data/COCO/images')
ANNOTATIONS_PATH = os.path.abspath('data/COCO/labels')
VIDEO_PATH = os.path.abspath('data/object_tracking.mp4')


if __name__ == '__main__':
    initialize_environment()

    for Model in [RetinaNet]:
        Model().evaluate(IMAGES_PATH, VIDEO_PATH, ANNOTATIONS_PATH, 10)

    print_debug('\nExiting...')
