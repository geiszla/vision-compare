from typing import List

import numpy
import tensorflow
from keras.backend.tensorflow_backend import set_session

from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSD(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import SSD as SSDModel
        from lib.squeezedet_keras.main.model.modelLoading import load_only_possible_weights

        super().__init__('SSD with MobileNetv2')

        config = tensorflow.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        set_session(tensorflow.Session(config=config))

        self.keras_model = SSDModel((300, 300, 3), 2)
        load_only_possible_weights(self.keras_model, 'model_data/ssd_mobilenetv2lite_p05-p84.h5')

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        processed_images: List[ImageData] = []
        for image in images:
            image = numpy.asarray(image.resize((300, 300)).convert('RGB'))[:, :, ::-1]
            image = (image - numpy.mean(image)) / numpy.std(image)

            processed_images.append(image)

        return (processed_images, [1.0] * len(images)), annotations

    def detect_image(self, image: ImageData) -> PredictionResult:
        from lib.ssd_kerasV2.ssd_utils import BBoxUtility

        predictions = self.keras_model.predict(numpy.expand_dims(image, 0))

        bbox_utility = BBoxUtility(2)
        [processed_predictions] = bbox_utility.detection_out(predictions)

        boxes: List[List[float]] = []
        classes: List[int] = []
        scores: List[float] = []

        for class_name, score, x_min, y_min, x_max, y_max in processed_predictions:
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(numpy.int32(class_name) - 1)
            scores.append(score)

        return (
            numpy.array(boxes, numpy.float32),
            numpy.array(classes),
            numpy.array(scores, numpy.float32)
        )
