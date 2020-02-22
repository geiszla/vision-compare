import os

from utilities import print_debug, initialize_environment
from models import YOLOv3, SqueezeDet, SSD, RetinaNet


IMAGES_PATH = os.path.abspath('data/COCO/images')
VIDEO_PATH = os.path.abspath('data/video.mp4')


initialize_environment()


if __name__ == "__main__":
    # Evaluate Yolov3
    YOLOv3().evaluate(IMAGES_PATH, VIDEO_PATH)

    # Evaluate SqueezeDet
    SqueezeDet().evaluate(IMAGES_PATH, VIDEO_PATH)

    # Evaluate SSD
    SSD().evaluate(IMAGES_PATH, VIDEO_PATH)

    # Evaluate RetinaNet
    RetinaNet().evaluate(IMAGES_PATH, VIDEO_PATH)

    print_debug('\nExiting...')
