import os
import statistics
import time
from abc import ABC, abstractmethod
from typing import Generic, List

import cv2
import numpy
from nptyping import Array
from easydict import EasyDict
from keras import Model
from PIL import Image as PillowImage

from typings import Batch, DataGenerator, ImageType, PredictionResult, ProcessedBatch, \
    ProcessedImageType, Statistics
from utilities import print_debug


class Detector(ABC, Generic[ImageType, ProcessedImageType]):
    def __init__(self, description: str):
        from lib.squeezedet_keras.main.config.create_config import load_dict, squeezeDet_config

        self.keras_model: Model = None
        self.description = description

        config_overrides = load_dict(os.path.abspath('res/config.json'))
        self.config = EasyDict({**squeezeDet_config(''), **config_overrides})

    @abstractmethod
    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        pass

    @abstractmethod
    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        pass

    @abstractmethod
    def detect_images(self, processed_images: List[ProcessedImageType]) -> PredictionResult:
        pass

    def evaluate(
        self, images_path: str, video_path: str, annotations_path: str, total_samples: int,
    ) -> None:
        from lib.squeezedet_keras.main.model.evaluation import compute_statistics

        print_debug(f'\nEvaluating {self.description} on {total_samples} samples...')

        # Load image names and annotations
        image_files: List[str] = [os.path.abspath(os.path.join(images_path, image))
            for image in os.listdir(images_path)]
        annotation_files: List[str] = [os.path.abspath(os.path.join(annotations_path, annotations))
            for annotations in os.listdir(annotations_path)]

        # Create generator
        generator: DataGenerator = self.data_generator(
            image_files[:total_samples],
            annotation_files[:total_samples],
        )

        # Get predictions in batches
        boxes: List[List[Array]] = []
        classes: List[List[Array]] = []
        scores: List[List[Array]] = []
        annotations: List[List[Array]] = []
        sample_count = 0

        for data in generator:
            image_batch, annotation_batch = self.preprocess_data(data)
            box_batch, class_batch, score_batch = self.detect_images(image_batch)

            filtered_boxes: List[Array] = []
            filtered_classes: List[Array] = []
            filtered_scores: List[Array] = []

            for image_index, _ in enumerate(class_batch):
                filtered_indexes = [index for index, class_id in enumerate(class_batch[image_index])
                    if class_id in self.config.CLASS_TO_IDX.values()]

                if len(filtered_indexes) > 0:
                    filtered_boxes.append(box_batch[image_index][filtered_indexes])
                    filtered_classes.append(class_batch[image_index][filtered_indexes])
                    filtered_scores.append(score_batch[image_index][filtered_indexes])

            if len(filtered_classes) > 0:
                boxes.append(filtered_boxes)
                classes.append(filtered_classes)
                scores.append(filtered_scores)
                annotations.append(annotation_batch)

            sample_count += len(image_batch)
            print_debug(f'{sample_count}/{total_samples}')

        # Increment class ids, because evaluation script removes zero labels (person)
        incremented_class_ids = {'CLASS_TO_IDX': {name: id + 1 for name, id
            in self.config.CLASS_TO_IDX.items()}}

        # Get statistics on prediction results
        model_statistics: Statistics = compute_statistics(boxes, classes, scores, annotations,
            EasyDict({**self.config, **incremented_class_ids}))

        # Print evaluation results (precision, recall, f1, AP)
        statistics_zip = zip(*model_statistics)

        for index, (precision, recall, f1_score, average_precision) in enumerate(statistics_zip):
            print_debug(f'{self.config.CLASS_NAMES[index]} - precision: {precision}, '
                f'recall: {recall}, f1: {f1_score}, mAP: {statistics.mean(average_precision)}')

        self.evaluate_performance(video_path)

    def evaluate_performance(self, video_path: str, display: bool = False) -> None:
        print_debug('\nEvaluating performance...')

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
            processed_images, _ = self.preprocess_data(([PillowImage.fromarray(frame)], None))
            prediction = self.detect_images(processed_images)

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
                result = numpy.asarray(prediction)
                cv2.putText(result, text=f'FPS: {fps_measurements[-1]}', org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(255, 0, 0),
                    thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)
            else:
                print_debug(f'FPS: {fps_measurements[-1]}')

            # Interrupt performance evaluation if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the mean of all the measured fps values
        print_debug(f'Mean FPS: {int(statistics.mean(fps_measurements))}')
