import csv
from pathlib import Path

from pycocotools.coco import COCO
import requests


IS_DOWNLOAD_IMAGES = True


def get_annotations(image):
    annotation_ids = DATASET.getAnnIds(
        imgIds=image['id'],
        catIds=CATEGORY_IDS,
        iscrowd=None
    )

    return DATASET.loadAnns(annotation_ids)


def create_csv(images):
    with open('data/COCO/annotation.csv', mode='w', newline='') as annotation_file:
        for image in images:
            annotations = get_annotations(image)

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


def create_annotation_files(images):
    Path('data/COCO/labels').mkdir(parents=True, exist_ok=True)

    for image in images:
        image_name = image['file_name'].split('.')[-2]

        with open(f'data/COCO/labels/{image_name}.txt', mode='w') as annotation_file:
            annotations = get_annotations(image)

            for i, _ in enumerate(annotations):
                x_min = annotations[i]['bbox'][0]
                y_min = annotations[i]['bbox'][1]
                x_max = x_min + annotations[i]['bbox'][2]
                y_max = y_min + annotations[i]['bbox'][3]

                annotation_file.write(f'person 0 0 0'
                    f' {x_min} {y_min} {x_max} {y_max}\n')


if __name__ == '__main__':
    DATASET = COCO('data/COCO/annotations/instances_train2017.json')
    CATEGORY_IDS = DATASET.getCatIds(catNms=['person'])
    IMAGES = DATASET.loadImgs(DATASET.getImgIds(catIds=CATEGORY_IDS))[:1000]

    if not IS_DOWNLOAD_IMAGES:
        for index, image_data in enumerate(IMAGES):
            imageData = requests.get(image_data['coco_url']).content

            with open(f'data/COCO/images/{image_data["file_name"]}', 'wb') as handler:
                handler.write(imageData)

                if index % 10 == 0:
                    print(f'{index}/{len(IMAGES)}')

    create_annotation_files(IMAGES)
    # create_csv(IMAGES)
