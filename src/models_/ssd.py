from typing import Any, cast, List

import numpy
import tensorflow
from tensorflow import saved_model

from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSD(Detector):
    def __init__(self, variant: str = 'v2'):
        self.variant = variant
        self.model: Any = None

        super().__init__(f'SSD{self.variant} with MobileNet backbone')

        self.config.IMAGE_HEIGHT = 300
        self.config.IMAGE_WIDTH = 300

    def load_model(self) -> str:
        with cast(Any, tensorflow.Session()) as session:
            self.model = tensorflow.saved_model.loader.load(
                session,
                [saved_model.tag_constants.SERVING],
                f'model_data/ssd{self.variant}/saved_model',
            )

        return ''

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        return super().preprocess_data(data_batch)

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        example = tensorflow.train.Example()
        example.features.feature["x"].float_list.value.extend([processed_image])

        tensors = [tensor for tensor in tensorflow.get_default_graph().get_operations()
            if tensor.type == 'Placeholder']

        with cast(Any, tensorflow.Session()) as session:
            session.run(tensorflow.global_variables_initializer())
            # coord = tensorflow.train.Coordinator()
            # threads = tensorflow.train.start_queue_runners()

            predictions = session.run(
                self.model,
                feed_dict={f'{tensors[0].name}:0': numpy.expand_dims(processed_image, 0)}
            )

        predictions = self.model.signatures["predict"](
            examples=tensorflow.constant([example.SerializeToString()])
        )

        return predictions
