import os
from typing import List

import numpy
from keras import backend
from PIL.Image import Image

from definitions import PROJECT_PATH
from typings import DataGenerator, PredictionResult
from utilities import print_debug, data_generator
from .detector import Detector


class YOLOv3(Detector):
    def __init__(self):
        from lib.keras_yolo3.yolo import YOLO

        super().__init__('YOLOv3')

        self.config.BATCH_SIZE = 1

        yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
        os.chdir(yolo_directory)
        print_debug(f'Changed to YOLOv3 directory: {yolo_directory}\n')

        self.model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

        os.chdir(PROJECT_PATH)
        print_debug(f'\nChanged back to project directory: {PROJECT_PATH}\n')

        self.keras_model = self.model.yolo_model

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return data_generator(image_files, annotation_files, self.config, False)

    def detect_images(self, processed_images: List[Image]) -> PredictionResult:
        from lib.keras_yolo3.yolo3.utils import letterbox_image

        first_image = processed_images[0]

        image_size = (
            first_image.width - (first_image.width % 32),
            first_image.height - (first_image.height % 32)
        )
        boxed_image = letterbox_image(first_image, image_size)

        image_data = numpy.array(boxed_image, numpy.float32) / 255
        image_data = numpy.expand_dims(image_data, 0)

        boxes, scores, classes = self.model.sess.run(
            [self.model.boxes, self.model.scores, self.model.classes],
            feed_dict={
                self.model.yolo_model.input: processed_images,
                self.model.input_image_shape: [first_image.size[1], first_image.size[0]],
                backend.learning_phase(): 0
            }
        )

        return boxes, classes, scores
