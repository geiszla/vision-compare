import os
import statistics
import time
from typing import Any, List

import cv2
from easydict import EasyDict
from keras import backend, Model
import numpy
from PIL import Image

from typings import DataGenerator, EvaluationResult, RunnerResult
from utilities import print_debug, read_annotations


CLASS_NAMES = ['person']


class Detector():
    def __init__(self, model: Model, model_description: str):
        self.model = model
        self.session = backend.get_session()
        self.model_description = model_description

    def close(self):
        self.session.close()

    def detect_image(self, image: Any) -> RunnerResult:
        from lib.keras_yolo3.yolo3.utils import letterbox_image

        image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, image_size)

        image_data = numpy.array(boxed_image, dtype='float32') / 255
        image_data = numpy.expand_dims(image_data, 0)

        return self.session.run(
            [self.model.boxes, self.model.scores, self.model.classes],
            feed_dict={
                self.model.input: image_data,
                self.model.input_image_shape: [image.size[1], image.size[0]],
                backend.learning_phase(): 0
            }
        )

    def evaluate(self, images_path: str):
        from lib.squeezedet_keras.main.config.create_config import squeezeDet_config
        from lib.squeezedet_keras.main.model.dataGenerator import generator_from_data_path
        from lib.squeezedet_keras.main.model.evaluation import evaluate

        print_debug(f'Evaluating {self.model_description}...')

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
        evaluation_result: EvaluationResult = evaluate(self.model, generator, step_count, config)

        # Print evaluation results (precision, recall, f1, AP)
        for index, (precision, recall, f1_score, average_precision) in enumerate(evaluation_result):
            print_debug(f'{CLASS_NAMES[index]}: precision - {precision} '
                f'recall - {recall}, f1 - {f1_score}, AP - {average_precision}')

    def evaluate_performance(self, video_path: str, display: bool = False) -> int:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError("Couldn't open webcam or video")

        elapsed_time = 0.0
        frames_in_second = 0
        previous_time = time.time()

        fps_measurements = []

        while True:
            result, frame = video.read()
            if not result:
                break

            image = self.detect_image(Image.fromarray(frame))

            current_time = time.time()
            detection_time = current_time - previous_time
            previous_time = current_time

            elapsed_time += detection_time
            frames_in_second = frames_in_second + 1

            if elapsed_time > 1:
                elapsed_time = elapsed_time - 1
                fps_measurements.append(frames_in_second)
                frames_in_second = 0

            if display:
                result = numpy.asarray(image)
                cv2.putText(result, text=f'FPS: {fps_measurements[-1]}', org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0),
                    thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return int(statistics.mean(fps_measurements))
