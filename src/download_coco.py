"""Download from the COCO dataset
This script downloads images and their annotations from the specified categores of the COCO dataset

To be run from the project root (i.e. `python src/benchmark.py`)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from pycocotools.coco import COCO
import requests


# COCO-specific types
Image = Dict[str, str]
Annotation = Dict[str, Tuple[float, float, float, float]]

# Set this to False to only download image annotations
IS_DOWNLOAD_IMAGES = True

# Dataset and the category ids and images loaded from it
DATASET = COCO('data/COCO/annotations/instances_val2017.json')
CATEGORY_IDS: List[int] = DATASET.getCatIds(catNms=['person'])
IMAGE_IDS: List[int] = DATASET.getImgIds(catIds=CATEGORY_IDS)
IMAGES: List[Image] = DATASET.loadImgs(IMAGE_IDS)[:500]


def __get_annotations(image: Image) -> List[Annotation]:
    # Get annotations for an image
    annotation_ids: List[int] = DATASET.getAnnIds(
        imgIds=image['id'],
        catIds=CATEGORY_IDS,
        iscrowd=None
    )

    annotations: List[Annotation] = DATASET.loadAnns(annotation_ids)
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
    Path('data/COCO/labels').mkdir(parents=True, exist_ok=True)

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


if __name__ == '__main__':
    if IS_DOWNLOAD_IMAGES:
        # Create images directory if it doesn't exist
        Path('data/COCO/images').mkdir(parents=True, exist_ok=True)

        for index, image_data in enumerate(IMAGES):
            # Get image data for the current image and write it to a file
            imageData = requests.get(image_data['coco_url']).content

            with open(f'data/COCO/images/{image_data["file_name"]}', 'wb') as handler:
                handler.write(imageData)

                # After every 10th image downloaded, show progress
                if index % 10 == 0:
                    print(f'{index}/{len(IMAGES)}')

    # Downlaod and write annotations to file (text or CSV)
    __create_annotation_files(IMAGES)
    # __create_csv(IMAGES)
