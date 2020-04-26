"""SSD MXNet Model
"""

from typing import Any, List, cast

# import mxnet
import numpy
# from gluoncv import model_zoo, data, utils

from typings import Annotations, Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from utilities import read_annotations
from .detector import Detector


class SSD(Detector):
    def __init__(self, variant: str = 'v2'):
        self.variant = variant
        self.model: Any = None

        super().__init__(f'SSD{self.variant} with MobileNet backbone')

        self.config.IMAGE_HEIGHT = 512
        self.config.IMAGE_WIDTH = 512

    def load_model(self) -> str:
        # self.model = model_zoo.get_model('ssd_512_mobilenet1.0_coco', pretrained=True)
        return ''

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        image_count = len(image_files)

        end_index = 0
        batch_number = 0

        while end_index < image_count:
            start_index = batch_number * self.config.BATCH_SIZE

            end_index = start_index + self.config.BATCH_SIZE
            end_index = end_index if end_index <= image_count else image_count

            # image_batch, _ = data.transforms.presets.ssd.load_test(
            #     image_files[start_index:end_index],
            #     short=512
            # )

            annotation_batch = [read_annotations(annotation_file, self.config) for annotation_file
                in annotation_files[start_index:end_index]]
            annotations = cast(Annotations, numpy.array(annotation_batch))

            yield ([], annotations)

            batch_number += 1

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        # return data_batch
        return super().preprocess_data(data_batch)

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        # classes, scores, boxes = self.model(mxnet.nd.array([processed_image]))

        return cast(
            PredictionResult,
            (numpy.zeros((0, 4)), numpy.zeros((0,)), numpy.zeros((0,)))
        )
