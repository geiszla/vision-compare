from typing import List, Tuple

import tensorflow
from keras.backend.tensorflow_backend import set_session

from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSD(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import SSD as SSDModel

        super().__init__('SSD with MobileNetv2')

        config = tensorflow.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        set_session(tensorflow.Session(config=config))

        self.keras_model = SSDModel((300, 300, 3), 2)
        self.keras_model.load_weights('model_data/ssd_mobilenetv2lite_p05-p84.h5', by_name=True)

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        return super().preprocess_data(data_batch)

    def detect_images(self, processed_images: List[ImageData]) -> PredictionResult:
        from lib.ssd_kerasV2.ssd_utils import BBoxUtility

        predictions = self.keras_model.predict(processed_images)

        bbox_utility = BBoxUtility(2)
        processed_predictions: List[
            Tuple[str, float, int, int, int, int]
        ] = bbox_utility.detection_out(predictions)

        boxes: List[List[float]] = []
        classes: List[str] = []
        scores: List[float] = []

        for class_name, score, x_min, y_min, x_max, y_max in processed_predictions:
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(class_name)
            scores.append(score)

        return boxes, classes, scores
