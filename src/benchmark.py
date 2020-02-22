import os

from detector import Detector
from utilities import print_debug, initialize_environment


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_PATH = os.path.abspath('data/COCO/images')

initialize_environment()


# Model Loaders

def load_yolo() -> object:
    from lib.keras_yolo3.yolo import YOLO

    yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
    os.chdir(yolo_directory)
    print_debug('\nChanged to YOLOv3 directory: ' + yolo_directory)

    print_debug('Loading YOLOv3 model...\n')
    model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

    os.chdir(PROJECT_PATH)
    print_debug(f'\nChanged back to project directory: {PROJECT_PATH}\n')

    return model


def load_squeezedet() -> object:
    from lib.squeezedet_keras.main.config.create_config import squeezeDet_config
    from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet

    model = SqueezeDet(squeezeDet_config('vision_compare')).model
    model.load_weights('model_data/squeezedet.h5')

    return model


def load_ssd() -> object:
    from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import Model

    model = Model((300, 300, 3), 2)

    return model


def load_retinanet() -> object:
    from keras_retinanet.models import load_model

    model = load_model('model_data/retinanet.h5', backbone_name='resnet50')

    return model


# Benchmark Program

if __name__ == "__main__":
    # Evaluate models
    Detector(load_yolo(), 'YOLOv3').evaluate(IMAGES_PATH)
    Detector(load_squeezedet(), 'SqueezeDet').evaluate(IMAGES_PATH)
    Detector(load_ssd(), 'SSD with MobileNet v2').evaluate(IMAGES_PATH)
    Detector(load_retinanet(), 'RetinaNet with ResNet').evaluate(IMAGES_PATH)

    # image = Image.open(os.path.join(images_path, '000000000110.jpg'))

    # os.chdir(PROJECT_PATH)
    # print_debug(f'\nChanged back to project directory: {PROJECT_PATH}\n')

    # print_debug(f'{len(predictions[0])} boxes found')
    # print_debug('Loading class names...')
    # with open('res/coco_classes.txt') as classes_file:
    #     class_names = classes_file.readlines()
    # class_names = [class_name.strip() for class_name in class_names]

    # processed_predictions = process_predictions(predictions, class_names, image)
    # print_boxes(processed_predictions)

    print_debug('\nExiting...')
