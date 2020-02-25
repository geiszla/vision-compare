import os

from utilities import print_debug, initialize_environment
from models import YOLOv3, SqueezeDet, SSD, RetinaNet


CLASS_NAMES = ['person']

IMAGES_PATH = os.path.abspath('data/COCO/images')
ANNOTATION_PATH = os.path.abspath('data/COCO/labels')
VIDEO_PATH = os.path.abspath('data/object_tracking.mp4')


if __name__ == '__main__':
    initialize_environment()

    for Model in [SqueezeDet, RetinaNet, YOLOv3, SSD]:
        Model(['', *CLASS_NAMES]).evaluate(  # type: ignore
            IMAGES_PATH,
            VIDEO_PATH,
            ANNOTATION_PATH,
            10
        )

    print_debug('\nExiting...')
