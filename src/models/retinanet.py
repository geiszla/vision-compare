from typing import List

import numpy
from nptyping import Array
from keras_retinanet.utils.image import preprocess_image, resize_image

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch, ResizedImage
from utilities import data_generator
from .detector import Detector


class RetinaNet(Detector[ImageData, ResizedImage]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from keras_retinanet.models import load_model

        super().__init__('RetinaNet with ResNet50')

        self.keras_model = load_model('model_data/resnet50.h5', backbone_name='resnet50')
        self.config.BATCH_SIZE = 1

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return data_generator(image_files, annotation_files, self.config)

    @classmethod
    def preprocess_data(cls, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        return [resize_image(preprocess_image(image)) for image in images], annotations

    def detect_images(self, processed_images: List[ImageData]) -> PredictionResult:
        (image, scaling_factor) = processed_images[0]

        predicted_boxes, predicted_scores, predicted_classes = self.keras_model.predict(
            numpy.expand_dims(image, 0)
        )

        predicted_boxes /= scaling_factor

        boxes: List[Array[numpy.float32, 4]] = []
        scores: List[numpy.float32] = []
        classes: List[numpy.int32] = []

        predictions = zip(predicted_boxes[0], predicted_scores[0], predicted_classes[0])
        for box, score, class_id in predictions:
            if score < 0.5:
                break

            if class_id != 0:
                continue

            boxes.append(box)
            scores.append(score)
            classes.append(class_id)

        return (
            numpy.expand_dims(boxes, 0),
            numpy.expand_dims(classes, 0),
            numpy.expand_dims(scores, 0),
        )
