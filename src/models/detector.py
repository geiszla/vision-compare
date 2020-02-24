import os
import statistics
import time
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy
from easydict import EasyDict
from keras import Model
from PIL import Image
from PIL.Image import Image as ImageType

from ..typings import DataGenerator, EvaluationResult, PredictionResult
from ..utilities import print_debug, read_annotations


CLASS_NAMES = ['person']


class Detector(ABC):
    def __init__(self, description: str):
        self.keras_model = self.load_model()
        self.description = description

    @abstractmethod
    def load_model(self) -> Model:
        pass

    @abstractmethod
    def detect_image(self, image: ImageType) -> PredictionResult:
        pass

    def evaluate(self, images_path: str, video_path: str) -> None:
        from lib.squeezedet_keras.main.config.create_config import squeezeDet_config
        from lib.squeezedet_keras.main.model.dataGenerator import generator_from_data_path
        from lib.squeezedet_keras.main.model.evaluation import evaluate

        print_debug(f'Evaluating {self.description}...')

        # Load image names and annotations
        image_names: List[str] = [os.path.abspath(image) for image in os.listdir(images_path)]
        annotations = read_annotations('data/COCO/annotations.csv')

        # Create evaluation configuration
        config_overrides = {'CLASS_NAMES': CLASS_NAMES}
        config = EasyDict({**squeezeDet_config(''), **config_overrides})

        # Create generator
        generator: DataGenerator = generator_from_data_path(image_names, annotations, config)
        step_count = len(annotations) // config.BATCH_SIZE

        # Evaluate model
        evaluation_result: EvaluationResult = evaluate(self.keras_model, generator, step_count,
            config)

        # Print evaluation results (precision, recall, f1, AP)
        for index, (precision, recall, f1_score, average_precision) in enumerate(evaluation_result):
            print_debug(f'{CLASS_NAMES[index]}: precision - {precision} '
                f'recall - {recall}, f1 - {f1_score}, AP - {average_precision}')

        self.evaluate_performance(video_path)

    def evaluate_performance(self, video_path: str, display: bool = False) -> None:
        # Open video feed
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError("Couldn't open webcam or video")

        # Initialize timings and fps counter
        elapsed_time = 0.0
        frames_in_second = 0
        previous_time = time.time()

        fps_measurements = []

        while True:
            # Read a frame from video
            result, frame = video.read()
            if not result:
                break

            # Run object detection using the current model (self)
            image = self.detect_image(Image.fromarray(frame))

            # Calculate elapsed and detection time
            current_time = time.time()
            detection_time = current_time - previous_time
            previous_time = current_time

            elapsed_time += detection_time
            frames_in_second = frames_in_second + 1

            if elapsed_time > 1:
                # If one second elapsed, save current fps measurement and reset fps counter
                elapsed_time = elapsed_time - 1
                fps_measurements.append(frames_in_second)
                frames_in_second = 0

            if display:
                # If display is turned on, show the current detection result (bounding box on image)
                result = numpy.asarray(image)
                cv2.putText(result, text=f'FPS: {fps_measurements[-1]}', org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0),
                    thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)

            # Interrupt performance evaluation if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the mean of all the measured fps values
        print_debug(f'Mean FPS: {int(statistics.mean(fps_measurements))}')
