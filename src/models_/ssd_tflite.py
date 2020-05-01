"""SSD TFLite model
"""

from typing import Any, Dict, List, cast

import numpy
import tflite_runtime.interpreter as tflite

from utilities import get_edgetpu_library_file
from typings import (Batch, Boxes, Classes, ImageData, Images, DataGenerator, PredictionResult,
    ProcessedBatch, Scores)
from .detector import Detector


class SSDTFLite(Detector):
    def __init__(self, variant: str = 'v1'):
        self.interpreter: tflite.Interpreter
        self.input_details: List[Dict[str, Any]] = []
        self.output_details: List[Dict[str, Any]] = []

        self.variant = variant

        super().__init__(f'SSD{self.variant} TFlite model with MobileNet backbone')

        # Model's input layer size is fixed (300, 300)
        self.config.IMAGE_HEIGHT = 300
        self.config.IMAGE_WIDTH = 300

    def load_model(self) -> str:
        model_file = f'model_data/ssd{self.variant}_edgetpu.tflite'

        try:
            # Load edge TPU model to TFLite interpreter
            self.interpreter = tflite.Interpreter(model_file,
                experimental_delegates=[
                    cast(Any, tflite.load_delegate(get_edgetpu_library_file()))
                ])
        except ValueError:
            # If no edge TPU device is connected, fall back to SSDv1 TFLite model
            model_file = 'model_data/ssdv1.tflite'
            self.interpreter = tflite.Interpreter(model_file)

        # Set up TFLite interpreter
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        return model_file

    def data_generator(
        self, image_files: List[str], annotation_files: List[str], sample_count: int,
    ) -> DataGenerator:
        return super().data_generator(image_files, annotation_files, sample_count)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        return super().preprocess_data(data_batch)

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        # Set interpreter input
        image = cast(Images, numpy.expand_dims(processed_image, 0))
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        # Perform prediction with interpreter
        self.interpreter.invoke()

        # Get predicted boxes, classes and scores from the output
        boxes: List[Boxes] = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes: List[Classes] = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores: List[Scores] = self.interpreter.get_tensor(self.output_details[2]['index'])

        image_boxes = cast(Boxes, numpy.array(
            [[box[1], box[0], box[3], box[2]] for box in boxes[0]],
            dtype=numpy.float32,
        ))

        return image_boxes, cast(Classes, numpy.int32(classes[0])), scores[0]
