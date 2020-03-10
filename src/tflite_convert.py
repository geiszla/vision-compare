import os
import inspect
from typing import Any, List

import numpy
import tensorflow
from tensorflow import saved_model
from keras import backend

import models_
from models_ import Detector, SSD
from utilities import initialize_environment, print_debug


MODEL_DATA_PATH = 'model_data'

DATA_PATH = 'data/COCO'
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'labels')

SAVED_MODELS = ['ssdlitev2', 'ssdv2']


def input_data_generator(model: Detector, image_files: List[str], annotation_files: List[str]):
    for data_batch in model.data_generator(image_files, annotation_files):
        [processed_image], _ = model.preprocess_data(data_batch)
        yield [numpy.expand_dims(numpy.int32(processed_image), 0)]


def quantize_model(converter: tensorflow.lite.TFLiteConverter, model: Detector) -> Any:
    print_debug('Converting model, this will take some time...')

    image_files: List[str] = [os.path.abspath(os.path.join(IMAGES_PATH, image))
        for image in os.listdir(IMAGES_PATH)]
    annotation_files: List[str] = [os.path.abspath(os.path.join(ANNOTATIONS_PATH, annotations))
        for annotations in os.listdir(ANNOTATIONS_PATH)]

    try:
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: input_data_generator(
            model,
            image_files,
            annotation_files
        )

        converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tensorflow.uint8
        converter.inference_output_type = tensorflow.uint8

        return converter.convert()
    except Exception as exception:  # pylint: disable=broad-except
        print_debug(f'Error: Could not convert model to TFLite')
        print_debug(str(exception))


def write_to_file(file_name: str, model: Any) -> None:
    with open(file_name, 'wb') as model_file:
        model_file.write(model)
        print_debug(f'TensorFlow Lite model has been written to {file_name}')


if __name__ == '__main__':
    initialize_environment()

    for MODEL_NAME in SAVED_MODELS:
        MODEL_PATH = os.path.join('model_data', MODEL_NAME, 'saved_model')

        with tensorflow.Session() as SESSION:
            print_debug('\nLoading SSDv2 model from checkpoint...')

            tensorflow.saved_model.loader.load(
                SESSION,
                [saved_model.tag_constants.SERVING],
                MODEL_PATH,
            )

            TENSORS = [tensor for tensor in tensorflow.get_default_graph().get_operations()\
                if tensor.type == 'Placeholder']

            SAVED_CONVERTER = tensorflow.lite.TFLiteConverter.from_saved_model(
                MODEL_PATH,
                input_shapes={
                    TENSORS[0].name: [1, 300, 300, 3]
                }
            )

            # TENSORS = tensorflow.get_default_graph().get_operations()
            # FIRST_TENSOR = tensorflow.shape(TENSORS[0].node_def.attr['value'].tensor)
            # LAST_TENSOR = tensorflow.shape(TENSORS[0].node_def.attr['value'].tensor)

            SAVED_TFLITE_MODEL = quantize_model(SAVED_CONVERTER, SSD())
            write_to_file(f'model_data/ssdv2.tflite', SAVED_TFLITE_MODEL)

    EXCLUDED_CLASS_NAMES = ['RetinaNet', 'SqueezeDet', 'YOLOv3']

    print_debug('The following models are excluded from convesion,'
        ' because they are not TFLite compatible: ')
    print_debug(", ".join(EXCLUDED_CLASS_NAMES))

    EXCLUDED_CLASS_NAMES.append('Detector')
    MODELS = {name: Model for name, Model in models_.__dict__.items()
        if inspect.isclass(Model) and issubclass(Model, Detector)
            and name not in EXCLUDED_CLASS_NAMES}  # noqa: W503

    for NAME, MODEL_CLASS in MODELS.items():
        MODEL: Detector = MODEL_CLASS()

        CONVERTER = tensorflow.lite.TFLiteConverter.from_session(
            backend.get_session(),
            MODEL.keras_model.inputs,
            MODEL.keras_model.outputs
        )

        TFLITE_MODEL = quantize_model(CONVERTER, MODEL)
        write_to_file(f'model_data/{NAME}.tflite', TFLITE_MODEL)

    print_debug('\nExiting...')
