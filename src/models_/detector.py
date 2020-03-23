import os
import statistics
import time
from abc import ABC, abstractmethod
from typing import cast, Any, Dict, Iterator, List, Tuple, Union

import cv2
import numpy
from tensorflow import Tensor
from easydict import EasyDict
from keras import layers, Model
from PIL import Image
from PIL.Image import Image as PillowImage

from typings import (Batch, BatchAnnotations, Boxes, Classes, DataGenerator, ImageData,
    PredictionResult, ProcessedBatch, Scores, Statistics, StatisticsEntry)
from utilities import print_debug, read_annotations


class Detector(ABC):
    def __init__(self, description: str):
        from lib.squeezedet_keras.main.config.create_config import load_dict, squeezeDet_config
        from lib.squeezedet_keras.main.model.modelLoading import load_only_possible_weights

        print_debug(f'\nPreparing {description}...')

        self.keras_model: Model = None
        self.description = description

        config_overrides: EasyDict = load_dict(os.path.abspath('config.json'))
        self.config: Any = EasyDict(cast(Dict[str, Any], {
            **squeezeDet_config(''),
            **config_overrides,
            'SCORE_THRESHOLD': 0.5
        }))
        self.config.BATCH_SIZE = 1

        print_debug('Loading model...')
        model_file = self.load_model()

        if self.keras_model is not None and model_file != '':
            new_input: Tensor = layers.Input(
                batch_shape=(1, self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT, 3)
            )
            new_layers = self.keras_model(new_input)
            self.keras_model = Model(new_input, new_layers)

            load_only_possible_weights(self.keras_model, model_file)

    @abstractmethod
    def load_model(self) -> str:
        pass

    @abstractmethod
    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        image_count = len(image_files)

        end_index = 0
        batch_number = 0

        while end_index < image_count:
            start_index = batch_number * self.config.BATCH_SIZE

            end_index = start_index + self.config.BATCH_SIZE
            end_index = end_index if end_index <= image_count else image_count

            image_batch: List[PillowImage] = [Image.open(image_file) for image_file
                in image_files[start_index:end_index]]
            annotation_batch = [read_annotations(annotation_file, self.config) for annotation_file
                in annotation_files[start_index:end_index]]

            yield cast(Batch, (image_batch, numpy.array(annotation_batch)))

            batch_number += 1

    @abstractmethod
    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        images, annotations = data_batch

        processed_images: List[ImageData] = []

        for image in images:
            processed_image: PillowImage = image.resize(
                (self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH)
            ).convert('RGB')

            processed_images.append(numpy.array(processed_image)[:, :, ::-1])

        return processed_images, annotations

    @abstractmethod
    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        pass

    def evaluate(
        self, images_path: str, video_path: str, annotations_path: str, total_samples: int,
    ) -> Tuple[List[StatisticsEntry], float]:
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
        boxes: List[Boxes] = []
        classes: List[Classes] = []
        scores: List[Scores] = []
        annotations: List[BatchAnnotations] = []
        sample_count = 0

        for data in generator:
            _, batch_annotations = data
            (batch_boxes, batch_classes, batch_scores) = self.__detect_batch(data)

            boxes.append(batch_boxes[0])
            classes.append(batch_classes[0])
            scores.append(batch_scores[0])
            annotations.append(batch_annotations)

            sample_count += 1

            if sample_count % 10 == 0:
                print_debug(f'{sample_count}/{total_samples}')

        # Increment class ids, because evaluation script removes zero labels (person)
        incremented_class_ids = {'CLASS_TO_IDX': {name: id + 1 for name, id
            in self.config.CLASS_TO_IDX.items()}}

        # Get statistics on prediction results
        model_statistics: Statistics = compute_statistics(boxes, classes, scores, annotations,
            EasyDict({**self.config, **incremented_class_ids}))

        # Print evaluation results (precision, recall, f1, AP)
        statistics_zip = cast(
            Iterator[Tuple[float, float, float, List[float]]],
            zip(*model_statistics)
        )

        accuracy_statistics: List[StatisticsEntry] = []
        for index, (precision, recall, f1_score, average_precision) in enumerate(statistics_zip):
            print_debug(f'{self.config.CLASS_NAMES[index]} - precision: {precision}, '
                f'recall: {recall}, f1: {f1_score}, '
                f'mAP: {statistics.mean(average_precision)}')

            accuracy_statistics.append((
                precision, recall, f1_score, statistics.mean(average_precision)
            ))

        return accuracy_statistics, self.evaluate_performance(video_path)

    def evaluate_performance(self, video_path: Union[str, int], is_display: bool = False) -> float:
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
        fps_text = 'N/A'

        if is_display:
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', (1280, 1024))

        while True:
            # Read a frame from video
            result, frame = video.read()
            if not result:
                break

            # Run object detection using the current model (self)
            predictions = self.__detect_batch(cast(Batch, ([Image.fromarray(frame)], None)))
            batch_boxes = predictions[0] if len(predictions) > 0 else []
            boxes = batch_boxes[0][0] if len(batch_boxes) > 0 else []

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

                fps_text = str(fps_measurements[-1])
                print_debug(f'FPS: {fps_text}')

                if len(fps_measurements) > 9 and not is_display:
                    break

            if is_display:
                # If display is turned on, show the current detection result (bounding box on image)
                for [ymin, xmin, ymax, xmax] in boxes:
                    cv2.rectangle(frame, (ymin, xmin), (ymax, xmax), (0, 255, 0), 4)

                cv2.putText(frame, text=f'FPS: {fps_text}', org=(3, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.50, color=(0, 255, 0),
                    thickness=2)
                cv2.imshow('result', frame)

            # Interrupt performance evaluation if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the mean of all the measured fps values
        mean_fps = statistics.mean(fps_measurements)
        print_debug(f'Mean FPS: {mean_fps}')

        return mean_fps

    def __detect_batch(self, data_batch: Batch) -> Tuple[List[Boxes], List[Classes], List[Scores]]:
        boxes: List[Boxes] = []
        classes: List[Classes] = []
        scores: List[Scores] = []

        [original_image], _ = data_batch
        [processed_image], _ = self.preprocess_data(data_batch)
        image_boxes, image_classes, image_scores = self.detect_image(processed_image)

        filtered_indexes = [index for index, class_id in enumerate(image_classes)
            if class_id in self.config.CLASS_TO_IDX.values()
                and image_scores[index] > self.config.SCORE_THRESHOLD]  # noqa: W503

        (original_width, original_height) = cast(Tuple[int, int], original_image.size)

        filtered_boxes = image_boxes[filtered_indexes]
        if len(filtered_boxes) > 0:
            filtered_boxes[:] = numpy.transpose([
                filtered_boxes[:, 0] * original_width,
                filtered_boxes[:, 1] * original_height,
                filtered_boxes[:, 2] * original_width,
                filtered_boxes[:, 3] * original_height,
            ])

        boxes.append(cast(Boxes, numpy.expand_dims(filtered_boxes, 0)))

        classes.append(cast(Classes, numpy.expand_dims(image_classes[filtered_indexes], 0)))
        scores.append(cast(Scores, numpy.expand_dims(image_scores[filtered_indexes], 0)))

        return boxes, classes, scores
