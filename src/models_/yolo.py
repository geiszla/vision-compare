import os
from typing import List

import numpy
from keras import backend
from PIL import Image as PillowImage
from PIL.Image import Image

from definitions import PROJECT_PATH
from typings import Batch, DataGenerator, PredictionResult, ProcessedBatch
from utilities import print_debug
from .detector import Detector


class YOLOv3(Detector[Image]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.keras_yolo3.yolo import YOLO

        super().__init__('YOLOv3')

        self.config.BATCH_SIZE = 1

        yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
        os.chdir(yolo_directory)
        print_debug(f'\nChanged to YOLOv3 directory: {yolo_directory}')
        print_debug('Loading model...')

        self.model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

        os.chdir(PROJECT_PATH)
        print_debug(f'Changed back to project directory: {PROJECT_PATH}')

        self.keras_model = self.model.yolo_model

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        from lib.keras_yolo3.yolo3.utils import letterbox_image

        (images, scaling_factors), annotations = super().preprocess_data(data_batch)

        processed_images: List[Image] = []
        for processed_image in images:
            image = PillowImage.fromarray(processed_image)

            image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, image_size)

            processed_images.append(boxed_image)

        return (processed_images, scaling_factors), annotations

    def detect_image(self, image: Image) -> PredictionResult:
        image_data = numpy.array(image, numpy.float32) / 255
        image_data = numpy.expand_dims(image_data, 0)

        boxes, scores, classes = self.model.sess.run(
            [self.model.boxes, self.model.scores, self.model.classes],
            feed_dict={
                self.model.yolo_model.input: image_data,
                self.model.input_image_shape: [image.size[1], image.size[0]],
                backend.learning_phase(): 0,
            }
        )

        return boxes, classes, scores
