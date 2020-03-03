from typing import List

import numpy
from keras import Model
from keras_retinanet.utils.image import preprocess_image
from nptyping import Array

from typings import Batch, DataGenerator, ImageData, PredictionResult, ProcessedBatch
from .detector import Detector


class RetinaNet(Detector):
    def __init__(self):
        self.keras_model: Model = None

        super().__init__('RetinaNet with ResNet50 backbone')

        self.config.SCORE_THRESHOLD = 0.3

    def load_model(self) -> str:
        from keras_retinanet.models import load_model

        model_file = 'model_data/resnet50.h5'
        self.keras_model = load_model(model_file, backbone_name='resnet50')

        return model_file

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = super().preprocess_data(data_batch)
        processed_images = [preprocess_image(image) for image in images]

        return processed_images, annotations

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        [predicted_boxes], [predicted_scores], [predicted_classes] = self.keras_model.predict(
            numpy.expand_dims(processed_image, 0)
        )

        width = self.config.IMAGE_WIDTH
        height = self.config.IMAGE_HEIGHT

        boxes: List[Array[numpy.float32, 4]] = []
        scores: List[numpy.float32] = []
        classes: List[numpy.int32] = []

        for box, score, class_id in zip(predicted_boxes, predicted_scores, predicted_classes):
            boxes.append([
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height,
            ])

            scores.append(score)
            classes.append(class_id)

        return numpy.array(boxes), numpy.array(classes), numpy.array(scores)
