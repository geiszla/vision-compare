import os

from keras import backend, Model
from PIL.Image import Image

import numpy

from ..definitions import PROJECT_PATH
from ..typings import PredictionResult
from ..utilities import print_debug
from .detector import Detector


class YOLOv3(Detector):
    def __init__(self):
        self.model = None
        super().__init__('YOLOv3')

    def load_model(self) -> Model:
        from lib.keras_yolo3.yolo import YOLO

        yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
        os.chdir(yolo_directory)
        print_debug(f'\nChanged to YOLOv3 directory: {yolo_directory}\n')

        model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

        os.chdir(PROJECT_PATH)
        print_debug(f'\nChanged back to project directory: {PROJECT_PATH}\n')

        self.model = model

        return model.yolo_model

    def detect_image(self, image: Image) -> PredictionResult:
        from lib.keras_yolo3.yolo3.utils import letterbox_image

        image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
        boxed_image = letterbox_image(image, image_size)

        image_data = numpy.array(boxed_image, dtype='float32') / 255
        image_data = numpy.expand_dims(image_data, 0)

        return self.model.sess.run(
            [self.model.boxes, self.model.scores, self.model.classes],
            feed_dict={
                self.keras_model.input: image_data,
                backend.placeholder(shape=(2, )): [image.size[1], image.size[0]],
                backend.learning_phase(): 0
            }
        )
