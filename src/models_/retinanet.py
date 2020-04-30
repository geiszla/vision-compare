"""Retinanet model
"""

from typing import List, cast

import numpy
from keras import Model
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import preprocess_image

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class RetinaNet(Detector):
    def __init__(self):
        self.keras_model: Model

        super().__init__('RetinaNet with ResNet50 backbone')

    def load_model(self) -> str:
        model_file = 'model_data/retinanet.h5'
        self.keras_model = load_model(model_file, backbone_name='resnet50')

        return model_file

    def data_generator(
        self, image_files: List[str], annotation_files: List[str], sample_count: int,
    ) -> DataGenerator:
        return super().data_generator(image_files, annotation_files, sample_count)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = super().preprocess_data(data_batch)

        # Preprocess images with method from keras_retinanet library
        processed_images = [cast(ImageData, preprocess_image(image)) for image in images]

        return processed_images, annotations

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        predictions: PredictionResult = cast(
            PredictionResult,
            self.keras_model.predict(numpy.expand_dims(processed_image, 0))
        )

        [predicted_boxes], [predicted_scores], [predicted_classes] = predictions

        width = self.config.IMAGE_WIDTH
        height = self.config.IMAGE_HEIGHT

        boxes: List[List[float]] = []  # type: ignore
        scores: List[numpy.float32] = []
        classes: List[numpy.int32] = []

        for box, score, class_id in zip(predicted_boxes, predicted_scores, predicted_classes):
            # Scale boxes with the original size of the image
            boxes.append([
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ])

            scores.append(score)
            classes.append(class_id)

        return cast(
            PredictionResult,
            (numpy.array(boxes), numpy.array(classes), numpy.array(scores))
        )
