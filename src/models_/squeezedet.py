from typing import List

import numpy

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class SqueezeDet(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel
        from lib.squeezedet_keras.main.model.modelLoading import load_only_possible_weights

        super().__init__('SqueezeDet')

        self.config.BATCH_SIZE = 1

        self.model = SqueezeDetModel(self.config)
        self.keras_model = self.model.model

        load_only_possible_weights(self.keras_model, 'model_data/squeezedet.h5')

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        processed_images: List[ImageData] = []
        for image in images:
            image = image.resize((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
            image = numpy.asarray(image.convert('RGB'))[:, :, ::-1]
            image = (image - numpy.mean(image)) / numpy.std(image)

            processed_images.append(image)

        return (processed_images, [1.0] * len(images)), annotations

    def detect_image(self, image: ImageData) -> PredictionResult:
        from lib.squeezedet_keras.main.model.evaluation import filter_batch

        [boxes], [classes], [scores] = filter_batch(
            self.keras_model.predict(numpy.expand_dims(image, 0)), self.config
        )

        return numpy.array(boxes), numpy.array(classes), numpy.array(scores)
