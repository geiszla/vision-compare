from typing import List

import numpy
from keras_retinanet.utils.image import preprocess_image, resize_image

from typings import DataGenerator, Image, PredictionResult, ProcessedImage
from utilities import data_generator
from .detector import Detector


# Model classes

class RetinaNet(Detector):
    def __init__(self):
        from keras_retinanet.models import load_model

        super().__init__('RetinaNet with ResNet50')

        self.keras_model = load_model('model_data/resnet50.h5', backbone_name='resnet50')
        self.config.BATCH_SIZE = 1

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        generator = data_generator(image_files, annotation_files, self.config)

        for image_batch, annotation_batch in generator:
            max_boxes = len(annotation_batch[0])

            processed_images: List[Image] = []
            annotations = numpy.zeros((len(image_batch), max_boxes, 10), object)

            for index, image in enumerate(image_batch):
                processed_image, scaling_factor = resize_image(preprocess_image(image))
                processed_images.append((processed_image, scaling_factor))

                annotations[index] = annotation_batch[index] \
                    + [[None] * 10] * (max_boxes - len(annotation_batch[index]))

            yield(processed_images, annotations)

    def detect_images(self, processed_images: List[ProcessedImage]) -> PredictionResult:
        (image, scaling_factor) = processed_images[0]

        predicted_boxes, predicted_scores, predicted_classes = self.keras_model.predict(
            numpy.expand_dims(image, 0)
        )

        predicted_boxes /= scaling_factor
        prediction_count = len(predicted_boxes[0])

        boxes = numpy.zeros((1, prediction_count, 4), float)
        scores = numpy.zeros((1, prediction_count, 1), float)
        classes = numpy.zeros((1, prediction_count, 1), str)

        predictions = zip(predicted_boxes[0], predicted_scores[0], predicted_classes[0])
        for index, (box, score, class_id) in enumerate(predictions):
            if score < 0.5:
                break

            if class_id != 0:
                continue

            boxes[0, index] = box
            scores[0, index] = score
            classes[0, index] = class_id

        return boxes, classes, scores
