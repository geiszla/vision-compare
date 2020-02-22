import os
from typing import List, Tuple

from keras import backend
from PIL import Image
from easydict import EasyDict

from typings import DataGenerator, RunnerResult, EvaluationResult
from utilities import get_image_data, process_predictions, print_boxes, print_debug, \
    initialize_environment, read_annotations


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_PATH = os.path.abspath('data/COCO/images')

initialize_environment()


# Model Loaders

def load_yolo() -> object:
    from lib.keras_yolo3.yolo import YOLO

    # yolo_directory = os.path.join(PROJECT_PATH, 'lib/keras_yolo3')
    # os.chdir(yolo_directory)
    # print_debug('\nChanged to YOLOv3 directory: ' + yolo_directory)

    print_debug('Loading YOLOv3 model...\n')
    model = YOLO(**{'model_path': os.path.join(PROJECT_PATH, 'model_data/yolov3.h5')})

    # image_data = get_image_data(image, model.model_image_size)  # pylint: disable=no-member

    # print_debug(f'\nRunning predictions on "{image.filename}"\n')
    # predictions = model.sess.run(
    #     [model.boxes, model.scores, model.classes],
    #     feed_dict={
    #         model.yolo_model.input: image_data,
    #         model.input_image_shape: [image.size[1], image.size[0]],
    #         backend.learning_phase(): 0
    #     }
    # )

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


# Benchmark Helpers

def evaluate(model: object, model_description: str) -> None:
    from lib.squeezedet_keras.main.model.evaluation import evaluate
    from lib.squeezedet_keras.main.config.create_config import squeezeDet_config

    print_debug(f'Evaluating {model_description}...')

    # Load image names and annotations
    image_names: List[str] = [os.path.abspath(image) for image in os.listdir(IMAGES_PATH)]
    annotations = read_annotations('data/COCO/annotations.csv')

    # Create evaluation configuration
    config_overrides = {'CLASS_NAMES': ['person']}
    config = EasyDict({**squeezeDet_config(''), **config_overrides})

    # Create generator
    generator: DataGenerator = generator_from_data_path(image_names, annotations, config)
    step_count = len(annotations) // config.BATCH_SIZE

    # Evaluate model
    evaluation_result: EvaluationResult = evaluate(model, generator, step_count, config)

    # Print evaluation results (precision, recall, f1, AP)
    for index, (precision, recall, f1, AP) in enumerate(evaluation_result):
        print_debug(f'{config.CLASS_NAMES[index]}: precision - {precision} ' +
            f'recall - {recall}, f1 - {f1}, AP - {AP}')


# Benchmark Program

if __name__ == "__main__":
    from lib.squeezedet_keras.main.model.dataGenerator import generator_from_data_path

    # Evaluate models
    evaluate(load_yolo(), 'YOLOv3')
    evaluate(load_squeezedet(), 'SqueezeDet')
    evaluate(load_ssd(), 'SSD with MobileNet v2')
    evaluate(load_retinanet(), 'RetinaNet with ResNet')

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
