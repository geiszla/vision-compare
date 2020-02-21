import sys
from os import path, chdir
from typing import Tuple

from keras import backend
from keras_retinanet.models import load_model
from PIL import Image

from __init__ import initialize_environment, PROJECT_PATH
from utilities import get_image_data, process_predictions, print_boxes, print_debug, ProcessedResult

RunnerResult = Tuple[object, ProcessedResult]

initialize_environment()


# Model Runner Functions

def run_yolo(image: Image) -> RunnerResult:
    from lib.keras_yolo3.yolo import YOLO

    yolo_directory = path.join(PROJECT_PATH, 'lib/keras_yolo3')
    chdir(yolo_directory)
    print_debug('\nChanged to YOLOv3 directory: ' + yolo_directory)

    print_debug('Loading YOLOv3 model...\n')
    model = YOLO(**{'model_path': path.join(PROJECT_PATH, 'model_data/yolov3.h5')})
    image_data = get_image_data(image, model.model_image_size)  # pylint: disable=no-member

    print_debug('\nRunning predictions on "{}"\n'.format(image.filename))
    predictions = model.sess.run(
        [model.boxes, model.scores, model.classes],
        feed_dict={
            model.yolo_model.input: image_data,
            model.input_image_shape: [image.size[1], image.size[0]],
            backend.learning_phase(): 0
        }
    )

    return predictions


# def run_squeezedet(image: Image) -> RunnerResult:
#     from lib.squeezedet_keras.main.config.create_config import create_config_from_dict
#     from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet

#     model = SqueezeDet(create_config_from_dict())


# def run_ssd(image: Image) -> RunnerResult:
#     from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import Model

#     model = Model((300, 300, 3), 2)


# def run_retinanet(image: Image) -> RunnerResult:
#     model = load_model('model_data/retinanet.h5', backbone_name='resnet50')


# Benchmark

def benchmark():
    image = Image.open(r'D:\Users\Andris\Desktop\vision-compare\data\COCO\images\000000000110.jpg')

    predictions = run_yolo(image)
    # predictions = run_squeezedet(image)
    # predictions = run_ssd(image)
    # predictions = run_retinanet(image)

    chdir(PROJECT_PATH)
    print_debug('\nChanged back to project directory: {}\n'.format(PROJECT_PATH))

    print_debug('{} boxes found'.format(len(predictions[0])))
    print_debug('Loading class names...')
    with open('res/coco_classes.txt') as classes_file:
        class_names = classes_file.readlines()
    class_names = [class_name.strip() for class_name in class_names]

    processed_predictions = process_predictions(predictions, class_names, image)
    print_boxes(processed_predictions)

    print_debug('\nExiting...')
    sys.exit()


benchmark()
