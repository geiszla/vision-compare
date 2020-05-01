"""Detector
This module contains an abstract Detector class to be used as a base for creating specific detectors
"""

import os
import statistics
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Union, cast

import cv2
import numpy
from tensorflow import Tensor
from easydict import EasyDict
from keras import layers, Model
from PIL import Image
from PIL.Image import Image as PillowImage

from typings import (Annotation, Batch, BatchAnnotations, Boxes, Classes, DataGenerator, ImageData,
    PredictionResult, ProcessedBatch, Scores, Statistics, StatisticsEntry)
from utilities import print_debug, read_voc_annotations


class Detector(ABC):
    """Abstract Detector class
    Use this as a superclass to implement specific object detection models.
    """

    def __init__(self, description: str):
        """Create an instance of the detector class

        Parameters
        ----------
        description (str): A few-word description of the specific detector model
        """

        from lib.squeezedet_keras.main.config.create_config import load_dict, squeezeDet_config
        from lib.squeezedet_keras.main.model.modelLoading import load_only_possible_weights

        print_debug(f'\nPreparing {description}...')

        self.keras_model: Model
        self.description = description

        # Load configuration from config.json and set a few custom values
        config_dictionary: Dict[str, Any] = {
            **squeezeDet_config(''),
            **load_dict(os.path.abspath('config.json')),
            'SCORE_THRESHOLD': 0.5,
        }
        self.config: Any = EasyDict(config_dictionary)

        # Only supports one image in each batch yet
        self.config.BATCH_SIZE = 1

        # Load the model using the class' implemented "load_model" method
        print_debug('Loading model...')
        model_file = self.load_model()

        # If the model is loaded and a weights file is given, set the input layer's shape to be the
        # expected shape of the input image (from config.json)
        if hasattr(self, 'keras_model') and self.keras_model is not None and model_file != '':
            # Create new (fix-sized) input layer
            new_input: Tensor = layers.Input(
                batch_shape=(1, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3)
            )
            new_layers: Model = self.keras_model(new_input)
            self.keras_model = Model(new_input, new_layers)

            # Load weights for each layer if it exist
            load_only_possible_weights(self.keras_model, model_file)

    @abstractmethod
    def load_model(self) -> str:
        """Implement this abstract method to load your model and assign it to the this.keras_model
            variable.

        Returns
        -------
        str: The name of the file the model weights can be loaded from. This is used to reload the
            weights after modifications to the model.
        """

    @abstractmethod
    def data_generator(
        self, image_files: List[str], annotation_files: List[str], sample_count: int,
    ) -> DataGenerator:
        """Implement this abstract method to generate data from the model from the given image and
            annotation files

        You can call `super().data_generator()` to load images and convert them to a common format

        Parameters
        ----------
        image_files (List[str]): Filenames of the images to be used as input data for the model

        annotation_files (List[str]): Names of the annotation files to be used to get the correct
            annotations from

        Returns
        -------
        DataGenerator: Generator, which yields a batch of data on each iteration

        Yields
        -------
        Batch: A batch of images and corresponding annotations
        """

        if len(image_files) != len(annotation_files):
            print('Error: Number of image and annotation files given do not equal.'
                ' Each image must have a corresponding annotation file (in the same order).'
                ' Please check the arguments passed.')
            sys.exit(1)

        image_files = [image_files[index] for index in numpy.argsort(image_files)]
        annotation_files = [annotation_files[index] for index in numpy.argsort(annotation_files)]

        image_batch: List[PillowImage] = []
        annotation_batch: List[List[Annotation]] = []

        image_count = len(image_files)
        annotation_count = 0
        iteration_count = 0

        while iteration_count < image_count and annotation_count <= sample_count:
            # Read the current annotations from the annotation file
            annotations = read_voc_annotations(annotation_files[iteration_count], self.config)

            if len(annotations) > 0:
                # If it contains valid annotations for the target class(es),
                # add it to the current batch
                image_batch.append(Image.open(image_files[iteration_count]))
                annotation_batch.append(annotations)

                annotation_count += 1

            if len(image_batch) >= self.config.BATCH_SIZE:
                # If the number of images in the batch equal to the batch size,
                # yield loaded images and annotations
                yield image_batch, cast(BatchAnnotations, numpy.array(annotation_batch))

                image_batch = []
                annotation_batch = []

            iteration_count += 1

        if len(image_batch) > 0:
            # Yield remaining samples (if number of images are not divisible with the batch size)
            yield image_batch, cast(BatchAnnotations, numpy.array(annotation_batch))

    @abstractmethod
    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        """Implement this abstract method to process the data returned by `data_generator`

        You can call `super().preprocess_data()` to perform common preprocessing operations on the
        given batch

        Parameters
        ----------
        data_batch (Batch): Batch of images and annotations to be preprocessed

        Returns
        -------
        ProcessedBatch: The preprocessed batch of images and annotations
        """

        images, annotations = data_batch

        processed_images: List[ImageData] = []
        for image in images:
            # Resize image to the proper size (from config) and convert to RGB
            processed_image: PillowImage = image.resize(
                (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
            ).convert('RGB')

            # Convert image data format to BGR
            processed_images.append(numpy.array(processed_image)[:, :, ::-1])

        return processed_images, annotations

    @abstractmethod
    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        """Implement this abstract method to run object detection on the preprocessed image data

        Parameters
        ----------
        processed_image (ImageData): One preprocessed image

        Returns
        -------
        PredictionResult: Detections (boxes, classes, scores) made on the current image
        """

    def evaluate(
        self, images_path: str, annotations_path: str, total_samples: int, video_path: str,
    ) -> Tuple[List[StatisticsEntry], float]:
        """Evaluates the current model in terms of its image detection quality and performance

        Parameters
        ----------
        images_path (str): Path of the image directory, where the evaluation images can be loaded
            from

        annotations_path (str): Path of the annotations directory, where the correct annotations for
            the evaluation images can be loaded from

        total_samples (int): Number of samples to use to evaluate the model's detection quality

        video_path (str): Path of the video to use for performance evaluation

        Returns
        -------
        Tuple[List[StatisticsEntry], float]: Tuple of model evaluation statistics (precision,
            recall, F1 score, mAP) and the performance metric (FPS)

        Raises
        ------
        IOError: The error if the given video can't be opened
        """

        from lib.squeezedet_keras.main.model.evaluation import compute_statistics

        print_debug(f'\nEvaluating {self.description} on {total_samples} samples...')

        # Load image names and annotations
        image_files: List[str] = [os.path.abspath(os.path.join(images_path, image))
            for image in os.listdir(images_path)]
        annotation_files: List[str] = [os.path.abspath(os.path.join(annotations_path, annotations))
            for annotations in os.listdir(annotations_path)]

        # Get predictions in batches
        boxes: List[Boxes] = []
        classes: List[Classes] = []
        scores: List[Scores] = []
        annotations: List[BatchAnnotations] = []

        sample_count = 0
        for data in self.data_generator(image_files, annotation_files, total_samples):
            _, batch_annotations = data
            (batch_boxes, batch_classes, batch_scores) = self.__detect_batch(data)

            boxes.append(batch_boxes[0])
            classes.append(batch_classes[0])
            scores.append(batch_scores[0])
            annotations.append(batch_annotations)

            sample_count += 1

            if sample_count % 10 == 0:
                print_debug(f'{sample_count}/{total_samples}')

        # Increment class ids, because evaluation script removes zero-labels
        incremented_class_ids = {
            'CLASS_TO_IDX': {name: id + 1 for name, id in self.config.CLASS_TO_IDX.items()}
        }

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
        """Evaluates the performance of the current model in terms of frames per second

        Parameters
        ----------
        video_path (Union[str, int]): Path of the video used for evaluating model performance

        is_display (bool, optional (default: false)): Switch to enable or disable displaying the
            video and the detection boxes while doing the performance evaluation

        Returns
        -------
        float: Mean FPS count for the duration of the evaluation

        Raises
        ------
        IOError: The error if the given video can't be opened
        """

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
            # Initialize window for video display
            cv2.namedWindow('result', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('result', (1280, 1024))

        while True:
            # Read a frame from video
            result, frame = video.read()
            if not result:
                break

            # Run object detection using the current model (self)
            predictions = self.__detect_batch(
                ([Image.fromarray(frame)], cast(BatchAnnotations, numpy.zeros((0, 0, 10))))
            )

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
        # Only supports one image in a batch yet
        [original_image], _ = data_batch
        [processed_image], _ = self.preprocess_data(data_batch)
        image_boxes, image_classes, image_scores = self.detect_image(processed_image)

        # Filter detections, which have less confidence than the threshold
        filtered_indexes = [index for index, class_id in enumerate(image_classes)
            if class_id in self.config.CLASS_TO_IDX.values()
                and image_scores[index] > self.config.SCORE_THRESHOLD]  # noqa: W503

        # Resize back the images to their original size
        original_size: Tuple[int, int] = original_image.size
        (original_width, original_height) = original_size

        filtered_boxes = image_boxes[filtered_indexes]
        if len(filtered_boxes) > 0:
            filtered_boxes[:] = numpy.transpose([
                filtered_boxes[:, 0] * original_width,
                filtered_boxes[:, 1] * original_height,
                filtered_boxes[:, 2] * original_width,
                filtered_boxes[:, 3] * original_height,
            ])

        # Create a batch from the current detections
        boxes = [cast(Boxes, numpy.expand_dims(filtered_boxes, 0))]
        classes = [cast(Classes, numpy.expand_dims(image_classes[filtered_indexes], 0))]
        scores = [cast(Scores, numpy.expand_dims(image_scores[filtered_indexes], 0))]

        return boxes, classes, scores
