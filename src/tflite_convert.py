import os
import inspect
from typing import List

import numpy
import tensorflow
from keras import backend

import models_
from models_ import Detector
from utilities import initialize_environment, print_debug


MODEL_DATA_PATH = 'model_data'

DATA_PATH = 'data/COCO'
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'labels')


def input_data_generator(model: Detector, image_files: List[str], annotation_files: List[str]):
    for data_batch in model.data_generator(image_files, annotation_files):
        ([processed_image], _), _ = model.preprocess_data(data_batch)
        yield [numpy.expand_dims(processed_image, 0)]


if __name__ == '__main__':
    initialize_environment()

    EXCLUDED_CLASS_NAMES = ['RetinaNet', 'SqueezeDet', 'YOLOv3']

    print_debug('The following models are excluded from convesion,'
        ' because they are not TFLite compatible: ')
    print_debug(", ".join(EXCLUDED_CLASS_NAMES))

    EXCLUDED_CLASS_NAMES.append('Detector')
    MODELS = {name: Model for name, Model in models_.__dict__.items()
        if inspect.isclass(Model) and issubclass(Model, Detector)
            and name not in EXCLUDED_CLASS_NAMES}  # noqa: W503

    for name, Model in MODELS.items():
        MODEL: Detector = Model()

        print_debug('Converting model, this will take some time...')

        IMAGE_FILES: List[str] = [os.path.abspath(os.path.join(IMAGES_PATH, image))
            for image in os.listdir(IMAGES_PATH)]
        ANNOTATION_FILES: List[str] = [os.path.abspath(os.path.join(ANNOTATIONS_PATH, annotations))
            for annotations in os.listdir(ANNOTATIONS_PATH)]

        try:
            CONVERTER = tensorflow.lite.TFLiteConverter.from_session(
                backend.get_session(),
                MODEL.keras_model.inputs,
                MODEL.keras_model.outputs,
            )

            CONVERTER.optimizations = [tensorflow.lite.Optimize.DEFAULT]
            CONVERTER.representative_dataset = lambda: input_data_generator(
                # pylint: disable=cell-var-from-loop
                MODEL,
                IMAGE_FILES,
                ANNOTATION_FILES
            )

            CONVERTER.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
            CONVERTER.inference_input_type = tensorflow.uint8
            CONVERTER.inference_output_type = tensorflow.uint8

            TFLITE_MODEL = CONVERTER.convert()
            TFLITE_FILE_NAME = f'model_data/{name}.tflite'

            with open(TFLITE_FILE_NAME, 'wb') as model_file:
                model_file.write(TFLITE_MODEL)
                print_debug(f'TensorFlow Lite model has been written to {TFLITE_FILE_NAME}')
        except Exception as exception:  # pylint: disable=broad-except
            print_debug(f'Error: Could not convert model to TFLite')
            print_debug(str(exception))

    print_debug('\nExiting...')
