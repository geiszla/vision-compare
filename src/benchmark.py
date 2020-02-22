import os

from utilities import print_debug, initialize_environment
from models import Detector, YOLOv3, SqueezeDet, SSD, RetinaNet


IMAGES_PATH = os.path.abspath('data/COCO/images')
VIDEO_PATH = os.path.abspath('data/video.mp4')


def evaluate_detector(detector: Detector):
    detector.evaluate(IMAGES_PATH, VIDEO_PATH)


if __name__ == "__main__":
    initialize_environment()

    # Evaluate Yolov3
    evaluate_detector(YOLOv3())

    # Evaluate SqueezeDet
    evaluate_detector(SqueezeDet())

    # Evaluate SSD
    evaluate_detector(SSD())

    # Evaluate RetinaNet
    evaluate_detector(RetinaNet())

    print_debug('\nExiting...')
