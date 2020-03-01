from typing import List

import numpy

from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSDv1(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.mobilenet_ssd_keras.models.ssd_mobilenet import ssd_300
        from lib.squeezedet_keras.main.model.modelLoading import load_only_possible_weights

        super().__init__('SSD with MobileNetv1')

        aspect_ratios = [
            [1.001, 2.0, 0.5],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
            [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
        ]

        self.keras_model = ssd_300(
            'inference',
            (self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3),
            self.config.CLASSES,
            scales=[0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1],
            aspect_ratios_per_layer=aspect_ratios,
            steps=[16, 32, 64, 100, 150, 300],
            offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            normalize_coords=True,
            subtract_mean=[127.5, 127.5, 127.5],
            divide_by_stddev=127.5,
            swap_channels=False,
        )

        for layer in self.keras_model.layers:
            layer.name = f'{layer.name}_v1'

        load_only_possible_weights(self.keras_model, 'model_data/ssd_mobilenetv1_converted.h5')

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

    def detect_image(self, processed_images: ImageData) -> PredictionResult:
        from lib.mobilenet_ssd_keras.misc.ssd_box_encode_decode_utils import decode_y

        predictions = self.keras_model.predict(
            numpy.array(processed_images)
        )

        boxes = decode_y(
            (box_batch, class_batch, score_batch),
            img_height=self.config.IMAGE_HEIGHT,
            img_width=self.config.IMAGE_WIDTH,
        )

        return boxes, class_batch, score_batch
