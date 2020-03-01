from typing import List

import numpy
from keras_retinanet.utils.image import preprocess_image
from nptyping import Array

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class RetinaNet(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from keras_retinanet.models import load_model

        super().__init__('RetinaNet with ResNet50')

        self.keras_model = load_model('model_data/resnet50.h5', backbone_name='resnet50')
        self.config.BATCH_SIZE = 1

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        (images, scaling_factors), annotations = super().preprocess_data(data_batch)
        processed_images = [preprocess_image(image) for image in images]

        return (processed_images, scaling_factors), annotations

    def detect_image(self, image: ImageData) -> PredictionResult:
        [predicted_boxes], [predicted_scores], [predicted_classes] = self.keras_model.predict(
            numpy.expand_dims(image, 0)
        )

        boxes: List[Array[numpy.float32, 4]] = []
        scores: List[numpy.float32] = []
        classes: List[numpy.int32] = []

        for box, score, class_id in zip(predicted_boxes, predicted_scores, predicted_classes):
            if score < 0.5:
                break

            if class_id != 0:
                continue

            predicted_boxes.append(box)
            scores.append(score)
            classes.append(class_id)

        return numpy.array(boxes), numpy.array(classes), numpy.array(scores)
