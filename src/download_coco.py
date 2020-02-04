import csv

from pycocotools.coco import COCO
import requests

DATASET = COCO('data/COCO/annotations/instances_train2017.json')
CATEGORY_IDS = DATASET.getCatIds(catNms=['person'])
IMAGES = DATASET.loadImgs(DATASET.getImgIds(catIds=CATEGORY_IDS))[:1000]

for index, image in enumerate(IMAGES):
    imageData = requests.get(image['coco_url']).content

    with open(f'data/COCO/images/{image["file_name"]}', 'wb') as handler:
        handler.write(imageData)

        if index % 10 == 0:
            print(f'{index}/{len(IMAGES)}')

with open('data/COCO/annotation.csv', mode='w', newline='') as annotationFile:
    for image in IMAGES:
        annotationIds = DATASET.getAnnIds(imgIds=image['id'], catIds=CATEGORY_IDS, iscrowd=None)
        annotations = DATASET.loadAnns(annotationIds)

    for i in enumerate(annotations):
        annotationWriter = csv.writer(annotationFile)
        annotationWriter.writerow([
            f'images/{image["file_name"]}',
            int(round(annotations[i]['bbox'][0])),
            int(round(annotations[i]['bbox'][1])),
            int(round(annotations[i]['bbox'][0] + annotations[i]['bbox'][2])),
            int(round(annotations[i]['bbox'][1] + annotations[i]['bbox'][3])),
            'person'
        ])
