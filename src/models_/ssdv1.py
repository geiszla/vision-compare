from typing import List

from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSDv1(Detector[ImageData]):  # pylint: disable=unsubscriptable-object
    def __init__(self):
        from lib.mobilenet_ssd_keras.models.ssd_mobilenet import ssd_300

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
            (300, 300, 3),
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

        self.keras_model.load_weights("model_data/ssd_mobilenetv1_converted.h5")

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    @classmethod
    def preprocess_data(cls, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        processed_images = [image.resize((300, 300)) for image in images]
        return (processed_images, [1.0] * len(images)), annotations

    def detect_images(self, processed_images: List[ImageData]) -> PredictionResult:
        from lib.mobilenet_ssd_keras.misc.ssd_box_encode_decode_utils import decode_y

        box_batch, class_batch, score_batch = self.keras_model.predict(processed_images)

        boxes = decode_y(
            (box_batch, class_batch, score_batch),
            confidence_thresh=0.25,
            top_k=100,
            img_height=300,
            img_width=300,
        )

        return boxes, class_batch, score_batch
