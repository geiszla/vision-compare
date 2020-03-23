import json
from typing import Any, cast, Dict, List, Optional

import numpy
from keras.models import load_model

from typings import Batch, Box, DataGenerator, ImageData, Images, PredictionResult, ProcessedBatch
from .detector import Detector


class YOLOv3(Detector):
    def __init__(self):
        from lib.keras_yolo3.generator import BatchGenerator

        self.yolo_generator: Optional[BatchGenerator] = None
        self.yolo_config: Dict[str, Any] = {}

        with open('lib/keras_yolo3/zoo/config_voc.json') as config_file:
            self.yolo_config = json.loads(config_file.read())

        super().__init__('YOLOv3')

    def load_model(self) -> str:
        self.config.IMAGE_WIDTH = self.config.IMAGE_WIDTH - self.config.IMAGE_WIDTH % 32
        self.config.IMAGE_HEIGHT = self.config.IMAGE_HEIGHT - self.config.IMAGE_HEIGHT % 32

        model_file = 'model_data/yolov3.h5'
        self.keras_model = load_model(model_file)

        return model_file

    def data_generator(self, image_files: List[str], annotation_files: List[str]) -> DataGenerator:
        return super().data_generator(image_files, annotation_files)

    def preprocess_data(self, data_batch: Batch) -> ProcessedBatch:
        from lib.keras_yolo3.generator import BatchGenerator
        from lib.keras_yolo3.utils.utils import normalize

        images, annotations = data_batch

        image_instances: List[Dict[str, Any]] = []
        if annotations is not None:
            image_instances = [{
                'filename': image.filename,
                'width': self.config.IMAGE_WIDTH,
                'height': self.config.IMAGE_HEIGHT,
                'object': [{
                    'name': self.config.CLASS_NAMES[file_annotations[9] - 1],
                    'xmin': file_annotations[1],
                    'ymin': file_annotations[2],
                    'xmax': file_annotations[3],
                    'ymax': file_annotations[4],
                } for file_annotations in annotations[index]]
            } for index, image in enumerate(images)]

        self.yolo_generator = BatchGenerator(
            image_instances,
            self.yolo_config['model']['anchors'],
            self.yolo_config['model']['labels'],
            max_box_per_image=0,
            batch_size=self.config.BATCH_SIZE,
            min_net_size=self.config.IMAGE_WIDTH,
            max_net_size=self.config.IMAGE_WIDTH,
            jitter=False,
            norm=cast(Any, normalize),
        )

        return super().preprocess_data(data_batch)

    def detect_image(self, processed_image: ImageData) -> PredictionResult:
        from lib.keras_yolo3.utils.utils import get_yolo_boxes
        from lib.keras_yolo3.utils.bbox import BoundBox

        assert self.yolo_generator is not None
        anchors: List[List[int]] = self.yolo_generator.get_anchors()

        predictions: List[BoundBox] = get_yolo_boxes(
            self.keras_model,
            cast(Images, numpy.expand_dims(numpy.uint8(processed_image), 0)),
            self.config.IMAGE_HEIGHT,
            self.config.IMAGE_WIDTH,
            anchors,
            self.config.IOU_THRESHOLD,
            self.config.NMS_THRESH,
        )[0]

        width = self.config.IMAGE_WIDTH
        height = self.config.IMAGE_HEIGHT

        predicted_boxes: List[List[Box]] = []
        predicted_classes: List[int] = []
        predicted_scores: List[float] = []

        for prediction in predictions:
            predicted_boxes.append([
                prediction.xmin / width,
                prediction.ymin / height,
                prediction.xmax / width,
                prediction.ymax / height,
            ])

            label: int = prediction.get_label()
            predicted_classes.append(label)

            score: float = prediction.get_score()
            predicted_scores.append(score)

        return cast(PredictionResult, (
            numpy.array(predicted_boxes, numpy.float32),
            numpy.array(predicted_classes, numpy.int32),
            numpy.array(predicted_scores, numpy.float32)
        ))
