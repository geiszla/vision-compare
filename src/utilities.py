import csv
import os
import random
import sys
import warnings
from typing import List

import numpy
from easydict import EasyDict
from PIL import Image

from typings import Annotation, DataGenerator, SplittedData


def initialize_environment(project_path: str = '') -> None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow  # pylint: disable=import-outside-toplevel

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    project_path = project_path or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_path)

    lib_path = os.path.join(project_path, "lib")
    for directory_name in os.listdir(lib_path):
        if directory_name != 'deep_sort_yolov3':
            sys.path.append(os.path.join(lib_path, directory_name))


def print_debug(message: str) -> None:
    print(f'\033[94m{message}\033[0m')


def read_annotations(file_name: str, config: EasyDict) -> List[Annotation]:
    with open(file_name, 'r') as annotation_file:
        annotation_lines: List[str] = annotation_file.readlines()

        annotations: List[Annotation] = []
        for line in annotation_lines:
            current_annotations = line.strip().split(' ')

            annotations.append([
                None,
                float(current_annotations[4]),
                float(current_annotations[5]),
                float(current_annotations[6]),
                float(current_annotations[7]),
                None, None, None, None,
                config.CLASS_TO_IDX[current_annotations[0]] + 1,
            ])

        return annotations

    return []


def read_annotations_csv(annotations_csv_name: str) -> List[Annotation]:
    with open(annotations_csv_name, 'r') as annotation_file:
        annotation_reader = csv.reader(annotation_file, delimiter=',')
        return [list(row) for row in annotation_reader]

    return []


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


def data_generator(
    image_files: List[str], annotation_files: List[str], config: EasyDict
) -> DataGenerator:
    image_count = len(image_files)

    end_index = 0
    batch_number = 0

    while end_index < image_count:
        start_index = batch_number * config.BATCH_SIZE

        end_index = start_index + config.BATCH_SIZE
        end_index = end_index if end_index <= image_count else image_count

        image_batch = [Image.open(image_file) for image_file in image_files[start_index:end_index]]
        annotation_batch = [read_annotations(annotation_file, config) for annotation_file
            in annotation_files[start_index:end_index]]

        yield image_batch, numpy.array(annotation_batch)

        batch_number += 1
