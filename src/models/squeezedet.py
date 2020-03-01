from typing import List

import numpy

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class SqueezeDet(Detector[ImageData, ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel

        super().__init__('SqueezeDet')

        self.config.BATCH_SIZE = 1
        self.model = SqueezeDetModel(self.config)

        self.keras_model = self.model.model
        self.keras_model.load_weights('model_data/squeezedet.h5')

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        from lib.squeezedet_keras.main.model.dataGenerator import generator_from_data_path

        return generator_from_data_path(image_files, annotation_files, self.config)

    @classmethod
    def preprocess_data(cls, data_batch: Batch) -> ProcessedBatch:
        return data_batch

    def detect_images(self, processed_images: List[ImageData]) -> PredictionResult:
        from lib.squeezedet_keras.main.model.evaluation import filter_batch

        predictions = numpy.expand_dims(processed_images[0], 0)
        return filter_batch(self.keras_model.predict(predictions), self.config)
