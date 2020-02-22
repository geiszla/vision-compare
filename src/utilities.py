import csv
import random
import sys
from os import path, listdir
from typing import cast, List

import numpy
from PIL.Image import Image

from typings import Annotation, PredictionResult, ProcessedBox, ProcessedResult, SplittedData


def read_annotations(annotations_csv_name: str) -> List[Annotation]:
    with open(annotations_csv_name, 'r') as annotation_file:
        annotation_reader = csv.reader(annotation_file, delimiter=',')

    return [cast(Annotation, tuple(row)) for row in annotation_reader]


def split_dataset(image_names: List[str], ground_truths: List[Annotation]) -> SplittedData:
    shuffled = list(zip(image_names, ground_truths))
    random.shuffle(shuffled)
    shuffled_image_names, shuffled_ground_truths = zip(*shuffled)

    image_count = len(shuffled_image_names)
    train_count = image_count * 70 // 100

    return (
        list(shuffled_image_names[0:train_count]),
        list(shuffled_image_names[train_count:image_count]),
        list(shuffled_ground_truths[0:train_count]),
        list(shuffled_ground_truths[train_count:image_count])
    )


def initialize_environment(project_path: str = '') -> None:
    project_path = project_path or path.abspath(path.join(path.dirname(__file__), ".."))
    sys.path.append(project_path)

    lib_path = path.join(project_path, "lib")
    for directory_name in listdir(lib_path):
        if directory_name != 'deep_sort_yolov3':
            sys.path.append(path.join(lib_path, directory_name))


def print_debug(message: str) -> None:
    print(f'\033[94m{message}\033[0m')


def get_image_data(image: Image, model_image_size: List[int]) -> numpy.ndarray:
    from lib.keras_yolo3.yolo3.utils import letterbox_image

    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)

    image_data = numpy.array(boxed_image, dtype='float32') / 255
    return numpy.expand_dims(image_data, 0)


def process_predictions(
    predictions: PredictionResult,
    class_names: List[str],
    image: Image
) -> ProcessedResult:
    (boxes, scores, classes) = predictions

    person_boxes: List[ProcessedBox] = []
    person_scores: List[float] = []
    person_classes: List[str] = []

    for index, box in enumerate(boxes):
        class_name = class_names[classes[index]]

        if class_name == 'person':
            top, left, bottom, right = box

            top = max(0, numpy.floor(top + 0.5).astype('int32'))
            left = max(0, numpy.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], numpy.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], numpy.floor(right + 0.5).astype('int32'))

            person_boxes.append(((left, top), (right, bottom)))
            person_scores.append(scores[index])
            person_classes.append(class_name)

    return (person_boxes, person_scores, person_classes)


def print_boxes(predictions: ProcessedResult) -> None:
    (boxes, scores, classes) = predictions

    for index, ((left, top), (right, bottom)) in enumerate(boxes):
        label = f'{classes[index]} {scores[index]:.2f}'
        print(label, (left, top), (right, bottom))
