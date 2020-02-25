from typing import List

from typings import DataGenerator, Image, PredictionResult
from .detector import Detector


class SqueezeDet(Detector):
    def __init__(self, class_names: List[str]):
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel

        super().__init__('SqueezeDet', class_names)

        self.model = SqueezeDetModel(self.config)

        self.keras_model = self.model.model
        self.keras_model.load_weights('model_data/squeezedet.h5')

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        from lib.squeezedet_keras.main.model.dataGenerator import generator_from_data_path

        return generator_from_data_path(image_files, annotation_files, self.config)

    def detect_images(self, images: List[Image]) -> PredictionResult:
        from lib.squeezedet_keras.main.model.evaluation import filter_batch

        return filter_batch(self.keras_model.predict(images), self.config)
