from typing import Any, Dict, List

import numpy
import tflite_runtime.interpreter as tflite

from utilities import get_edgetpu_library_file
from typings import Batch, ImageData, DataGenerator, PredictionResult, ProcessedBatch
from .detector import Detector


class SSD(Detector):
    def __init__(self, variant: str = 'v1'):
        self.interpreter: tflite.Interpreter = None
        self.input_details: List[Dict[str, Any]] = []
        self.output_details: List[Dict[str, Any]] = []

        self.variant = variant

        super().__init__('SSD with MobileNet backbone')

        self.config.IMAGE_HEIGHT = 300
        self.config.IMAGE_WIDTH = 300

    def load_model(self) -> str:
        model_file = f'model_data/ssd{self.variant}_edgetpu.tflite'

        try:
            self.interpreter = tflite.Interpreter(model_file,
                experimental_delegates=[tflite.load_delegate(get_edgetpu_library_file())])
        except ValueError:
            model_file = f'model_data/ssd{self.variant}.tflite'
            self.interpreter = tflite.Interpreter(model_file)

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        return model_file

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        return super().preprocess_data(data_batch)

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        image = numpy.expand_dims(processed_image, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)

        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])

        boxes[0] = [[box[1], box[0], box[3], box[2]] for box in boxes[0]]

        return boxes[0], numpy.int32(classes[0]), scores[0]
