import csv
import os
import platform
import random
import sys
import warnings
from typing import Any, cast, List, Tuple

import numpy
from easydict import EasyDict
from PIL import Image, ImageDraw

from typings import Annotation, ImageData, SplitData


def initialize_environment() -> None:
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    import tensorflow  # pylint: disable=import-outside-toplevel
    tensorflow.get_logger().setLevel('ERROR')
    tensorflow.autograph.set_verbosity(0)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_path)

    lib_path = os.path.join(project_path, "lib")
    for directory_name in os.listdir(lib_path):
        if directory_name != 'deep_sort_yolov3':
            sys.path.insert(0, os.path.join(lib_path, directory_name))

    sys.path.insert(0, os.path.join(lib_path, 'mobilenet_ssd_keras/models'))


def print_debug(message: str) -> None:
    print(f'\033[94m{message}\033[0m')


def read_annotations(file_name: str, config: EasyDict) -> List[Annotation]:
    with open(file_name, 'r') as annotation_file:
        annotation_lines: List[str] = annotation_file.readlines()

        annotations: List[Annotation] = []
        for line in annotation_lines:
            current_annotations = line.strip().split(' ')

            annotations.append(cast(Annotation, numpy.array([
                None,
                float(current_annotations[4]),
                float(current_annotations[5]),
                float(current_annotations[6]),
                float(current_annotations[7]),
                None, None, None, None,
                cast(Any, config).CLASS_TO_IDX[current_annotations[0]] + 1,
            ])))

        return annotations


def read_annotations_csv(annotations_csv_name: str) -> List[Annotation]:
    with open(annotations_csv_name, 'r') as annotation_file:
        annotation_reader = csv.reader(annotation_file, delimiter=',')
        return [cast(Annotation, numpy.array(row)) for row in annotation_reader]


def split_dataset(image_names: List[str], ground_truths: List[Annotation]) -> SplitData:
    shuffled = list(zip(image_names, ground_truths))
    random.shuffle(shuffled)
    shuffled_image_names, shuffled_ground_truths = cast(
        Tuple[str, List[Annotation]],
        zip(*shuffled)
    )

    image_count = len(shuffled_image_names)
    train_count = image_count * 70 // 100

    return (
        list(shuffled_image_names[0:train_count]),
        list(shuffled_image_names[train_count:image_count]),
        list(shuffled_ground_truths[0:train_count]),
        list(shuffled_ground_truths[train_count:image_count])
    )


def show_image_with_box(image: ImageData, box: Tuple[float, float, float, float]) -> None:
    image = Image.fromarray(numpy.asarray(image, numpy.uint8))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), fill='black')

    image.show()


def get_edgetpu_library_file() -> str:
    file_names = {
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
        'Darwin': 'libedgetpu.1.dylib'
    }

    return file_names[platform.system()]
