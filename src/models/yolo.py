import os
from typing import List

import numpy
from keras import backend, Model

from definitions import PROJECT_PATH
from typings import DataGenerator, Image, PredictionResult
from utilities import print_debug, data_generator
from .detector import Detector


class YOLOv3(Detector):
    def __init__(self, class_names: List[str]):
        from lib.keras_yolo3.yolo import YOLO

        self.model: YOLO = None
        super().__init__('YOLOv3', class_names)

    def load_model(self) -> Model:
        from lib.keras_yolo3.yolo import YOLO

        super().load_model()

        yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
        os.chdir(yolo_directory)
        print_debug(f'Changed to YOLOv3 directory: {yolo_directory}\n')

        model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

        os.chdir(PROJECT_PATH)
        print_debug(f'\nChanged back to project directory: {PROJECT_PATH}\n')

        self.model = model

        return model.yolo_model

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return data_generator(image_files, annotation_files, self.config.BATCH_SIZE)

    def detect_images(self, images: List[Image]) -> PredictionResult:
        from lib.keras_yolo3.yolo3.utils import letterbox_image

        first_image = images[0]

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
                self.model.yolo_model.input: images,
                self.model.input_image_shape: [first_image.size[1], first_image.size[0]],
                backend.learning_phase(): 0
            }
        )

        return boxes, classes, scores
