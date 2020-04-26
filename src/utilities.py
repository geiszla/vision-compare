"""Utilities
This module contains utility functions to be used by other python scripts and classes.
"""

import csv
import os
import platform
import random
import sys
import warnings
from typing import Any, List, Tuple, cast

import numpy
from easydict import EasyDict
from PIL import Image, ImageDraw

from typings import Annotation, ImageData, SplitData


def initialize_environment() -> None:
    """Initializes enviroment for the scripts

    Initializes the environment to be able to run the scripts in /lib directory properly and to
    make sure all necessary environment variables are set and required paths are added to
    sys.path.

    This method needs to be run before any /lib imports (this also means that lib imports cannot
    be done in the global scope, only in the methods they are used in, so that initialization is
    done before they are imported).
    """

    # Ignore tnesorflow user/future warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Set tensorflow log level to only log errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Import tensorflow after all environment variables are set
    import tensorflow
    tensorflow = cast(Any, tensorflow)

    # Set other options to disable tensorflow logs
    tensorflow.get_logger().setLevel('ERROR')
    tensorflow.autograph.set_verbosity(0)
    tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

    # Add project root to path
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_path)

    # Add /lib directory and all directories inside to path
    lib_path = os.path.join(project_path, "lib")
    for directory_name in os.listdir(lib_path):
        sys.path.insert(0, os.path.join(lib_path, directory_name))


def print_debug(message: str) -> None:
    """Prints a colored message to the command line to be distinguishable from other logs

    Parameters
    ----------
    message (str): Message to be printed to the command line
    """

    # Print message in blue
    print(f'\033[94m{message}\033[0m')


def read_annotations(file_name: str, config: EasyDict) -> List[Annotation]:
    """Read in annotations from a text file

    Annotation format adheres to https://github.com/omni-us/squeezedet-keras to be able to
    use evaluation script from that project

    Parameters
    ----------
    file_name (str): Text file containing annotations of one image (each line representing a
        different annotated object)

    config (EasyDict): Model evaluation configuration (initiated from config.json in project root)

    Returns
    -------
    List[Annotation]: Annotations on the image corresponding to the annotation file given
    """

    with open(file_name, 'r') as annotation_file:
        annotation_lines: List[str] = annotation_file.readlines()

        # Add annotations to a list line-by-line
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
                # Evaluation script discards annotations with class 0, so we need to increment it
                cast(Any, config).CLASS_TO_IDX[current_annotations[0]] + 1,
            ])))

        return annotations


def read_annotations_csv(annotations_csv_name: str) -> List[Annotation]:
    """Read in annotations from a csv file

    Not currently used by these scripts

    Parameters
    ----------
    annotations_csv_name (str): File name of the annotations CSV file
        (see format in `src/download_coco.py:__create_csv`)

    Returns
    -------
    List[Annotation]: Annotations of the image corresponding to the given CSV file
    """

    with open(annotations_csv_name, 'r') as annotation_file:
        annotation_reader = csv.reader(annotation_file, delimiter=',')
        return [cast(Annotation, numpy.array(row)) for row in annotation_reader]


def split_dataset(image_names: List[str], annotations: List[Annotation]) -> SplitData:
    """Shuffle and split images and their annotations into training and validation datasets

    Parameters
    ----------
    image_names (List[str]): Names of the image files to use in generated datasets

    ground_truths (List[Annotation]): Annotations corresponding to the images passed in as the first
        parameter.

    Returns
    -------
    SplitData: Data shuffled and split into training and validation sets.
    """

    # Shuffle images with their annotations
    shuffled = list(zip(image_names, annotations))
    random.shuffle(shuffled)
    shuffled_image_names, shuffled_annotations = cast(
        Tuple[str, List[Annotation]],
        zip(*shuffled)
    )

    # Separate shuffled dataset to training and validation sets
    image_count = len(shuffled_image_names)
    train_count = image_count * 70 // 100

    return (
        list(shuffled_image_names[0:train_count]),
        list(shuffled_image_names[train_count:image_count]),
        list(shuffled_annotations[0:train_count]),
        list(shuffled_annotations[train_count:image_count])
    )


def show_image_with_box(image: ImageData, box: Tuple[float, float, float, float]) -> None:
    """Display the given image and a bounding box on it

    Parameters
    ----------
    image (ImageData): Numpy array of the image to be displayed

    box (Tuple[float, float, float, float]): Bounding box to be drawn on the image
    """

    # Draw image and box using Pillow
    pillow_image = Image.fromarray(numpy.asarray(image, numpy.uint8))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(pillow_image)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), fill='black')

    pillow_image.show()


def get_edgetpu_library_file() -> str:
    """Get the shared edge TPU library filename corresponding to the current operating system.

    Returns
    -------
    str: Filename of the edge TPU library on the current operating system
    """

    file_names = {
        'Linux': 'libedgetpu.so.1',
        'Windows': 'edgetpu.dll',
        'Darwin': 'libedgetpu.1.dylib'
    }

    return file_names[platform.system()]
