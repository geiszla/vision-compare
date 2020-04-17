"""Download from the COCO dataset
This script downloads images and their annotations from the specified categores of the COCO dataset

To be run from the project root (i.e. `python src/benchmark.py`)
"""

import csv
from pathlib import Path
from os import path
import sys
from typing import Any, Dict, List, Optional, Tuple, cast

from pycocotools.coco import COCO
import requests
from tqdm import tqdm


# COCO-specific types
Image = Dict[str, str]
Annotation = Dict[str, Tuple[float, float, float, float]]

# Set this to False to only download image annotations
IMAGES_PATH = 'data/COCO/images'
LABELS_PATH = 'data/COCO/labels'
IS_DOWNLOAD_IMAGES = True

# Dataset and the category ids and images loaded from it
DATASET = COCO('data/COCO/annotations/instances_val2017.json')
CATEGORY_IDS: List[int] = DATASET.getCatIds(catNms=['person'])
IMAGE_IDS: List[int] = DATASET.getImgIds(catIds=CATEGORY_IDS)
IMAGES: Optional[List[Image]] = DATASET.loadImgs(IMAGE_IDS)


def __get_annotations(image: Image) -> List[Annotation]:
    # Get annotations for an image
    annotation_ids: List[int] = DATASET.getAnnIds(
        imgIds=image['id'],
        catIds=CATEGORY_IDS,
        iscrowd=None
    )

    annotations: Optional[List[Annotation]] = DATASET.loadAnns(annotation_ids)

    if annotations is None:
        print('Something went wrong. No annotations were found.')
        sys.exit(1)

    return annotations


def __create_csv(images: List[Image]):  # type: ignore
    with open('data/COCO/annotation.csv', mode='w', newline='') as annotation_file:
        for image in images:
            annotations = __get_annotations(image)

            # Write all annotations for each image in the CSV file named after the image it
            # describes
            for i, _ in enumerate(annotations):
                annotation_writer = csv.writer(annotation_file)
                annotation_writer.writerow([
                    f'images/{image["file_name"]}',
                    int(round(annotations[i]['bbox'][0])),
                    int(round(annotations[i]['bbox'][1])),
                    int(round(annotations[i]['bbox'][0] + annotations[i]['bbox'][2])),
                    int(round(annotations[i]['bbox'][1] + annotations[i]['bbox'][3])),
                    'person',
                ])


def __create_annotation_files(images: List[Image]):
    Path(LABELS_PATH).mkdir(parents=True, exist_ok=True)

    for image in images:
        image_name = image['file_name'].split('.')[-2]

        with open(f'data/COCO/labels/{image_name}.txt', mode='w') as annotation_file:
            annotations = __get_annotations(image)

            for i, _ in enumerate(annotations):
                x_min = annotations[i]['bbox'][0]
                y_min = annotations[i]['bbox'][1]
                x_max = x_min + annotations[i]['bbox'][2]
                y_max = y_min + annotations[i]['bbox'][3]

                annotation_file.write(f'person 0 0 0'
                    f' {x_min} {y_min} {x_max} {y_max}\n')


def __download_data():
    if IMAGES is None:
        print('Something went wrong. No images were found.')
        sys.exit(1)

    if IS_DOWNLOAD_IMAGES:
        # Create images directory if it doesn't exist
        Path(IMAGES_PATH).mkdir(parents=True, exist_ok=True)

        print(f'Downloading COCO images to {path.abspath(IMAGES_PATH)}...')
        for image_properties in cast(List[Image], tqdm(IMAGES[:500])):
            # Get image data for the current image and write it to a file
            image_data: Any = requests.get(image_properties['coco_url']).content

            with open(f'data/COCO/images/{image_properties["file_name"]}', 'wb') as handler:
                handler.write(image_data)

    print(f'Downloading COCO annotations to {path.abspath(LABELS_PATH)}...')

    # Downlaod and write annotations to file (text or CSV)
    __create_annotation_files(IMAGES)
    # __create_csv(IMAGES)

    print('COCO data successfully downloaded. Exiting...')


if __name__ == '__main__':
    __download_data()
