import os
from typing import List

import numpy
from keras import Model

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class SqueezeDet(Detector):
    def __init__(self):
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel

        self.model: SqueezeDetModel = None
        self.keras_model: Model = None

        super().__init__('SqueezeDet')

        self.config.SCORE_THRESHOLD = 0.3

    def load_model(self) -> str:
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel

        self.config.IMAGE_WIDTH = 1248
        self.config.IMAGE_HEIGHT = 384

        self.model = SqueezeDetModel(self.config)
        self.keras_model = self.model.model

        return 'model_data/squeezedet.h5'

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = super().preprocess_data(data_batch)
        processed_images = [(image - numpy.mean(image)) / numpy.std(image) for image in images]

        return processed_images, annotations

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        from lib.squeezedet_keras.main.model.evaluation import filter_batch
        from lib.squeezedet_keras.scripts.eval import eval

        os.chdir('lib/squeezedet_keras/')
        # eval()
        os.chdir('../../')


        [boxes], [classes], [scores] = filter_batch(
            self.keras_model.predict(numpy.expand_dims(processed_image, 0)), self.config
        )

        width = self.config.IMAGE_WIDTH
        height = self.config.IMAGE_HEIGHT

        for index, box in enumerate(boxes):
            boxes[index] = [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ]

        return numpy.array(boxes, numpy.float32), numpy.array(classes), numpy.array(scores)
