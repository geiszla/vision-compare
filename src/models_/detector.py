import os
import statistics
import time
from abc import ABC, abstractmethod
from typing import Generic, List

import cv2
import numpy
from easydict import EasyDict
from keras import Model
from keras_retinanet.utils.image import resize_image
from PIL import Image as PillowImage

from typings import (Batch, BatchAnnotations, Boxes, Classes, DataGenerator, ImageData, ImageType,
    PredictionResult, ProcessedBatch, Scores, Statistics)
from utilities import print_debug, read_annotations


class Detector(ABC, Generic[ImageType]):
    def __init__(self, description: str):
        from lib.squeezedet_keras.main.config.create_config import load_dict, squeezeDet_config

        print_debug(f'\nPreparing {description}...')

        self.keras_model: Model = None
        self.description = description

        config_overrides = load_dict(os.path.abspath('res/config.json'))
        self.config = EasyDict({**squeezeDet_config(''), **config_overrides})

        self.config.BATCH_SIZE = 1

    @abstractmethod
    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        image_count = len(image_files)

        end_index = 0
        batch_number = 0

        while end_index < image_count:
            start_index = batch_number * self.config.BATCH_SIZE

            end_index = start_index + self.config.BATCH_SIZE
            end_index = end_index if end_index <= image_count else image_count

            image_batch = [PillowImage.open(image_file) for image_file
                in image_files[start_index:end_index]]
            annotation_batch = [read_annotations(annotation_file, self.config) for annotation_file
                in annotation_files[start_index:end_index]]

            yield image_batch, numpy.array(annotation_batch)

            batch_number += 1

    @abstractmethod
    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        processed_images: List[ImageData] = []
        scaling_factors: List[float] = []

        for image in images:
            image_data = numpy.asarray(image.convert('RGB'))[:, :, ::-1]

            processed_image, scaling_factor = resize_image(image_data, max_side=300)

            processed_images.append(processed_image)
            scaling_factors.append(scaling_factor)

        return (processed_images, scaling_factors), annotations

    @abstractmethod
    def detect_image(self, processed_images: ImageType) -> PredictionResult:
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
        boxes: List[List[Boxes]] = []
        classes: List[List[Classes]] = []
        scores: List[List[Scores]] = []
        annotations: List[BatchAnnotations] = []
        sample_count = 0

        for data in generator:
            ([processed_image], [scaling_factor]), annotation_batch = self.preprocess_data(data)
            image_boxes, image_classes, image_scores = self.detect_image(processed_image)

            filtered_indexes = [index for index, class_id in enumerate(image_classes)
                if class_id in self.config.CLASS_TO_IDX.values()]

            if len(filtered_indexes) > 0:
                boxes.append(numpy.expand_dims(image_boxes[filtered_indexes] / scaling_factor, 0))
                classes.append(numpy.expand_dims(image_classes[filtered_indexes], 0))
                scores.append(numpy.expand_dims(image_scores[filtered_indexes], 0))

                annotations.append(annotation_batch)

            sample_count += 1
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
            ([processed_image], _), _ = self.preprocess_data(([PillowImage.fromarray(frame)], None))
            prediction = self.detect_image(processed_image)

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

                print_debug(f'FPS: {fps_measurements[-1]}')

                if len(fps_measurements) > 9:
                    break

            if display:
                # If display is turned on, show the current detection result (bounding box on image)
                result = numpy.asarray(prediction)
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
