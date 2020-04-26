"""TensorFlow Lite conversion script
This script converts models in src/models_ directory and all the saved keras models specified in
SAVED_MODELS to quantized TensorFlow Lite models, so that they can be used on edge TPUs.

To be run from the project root (i.e. `python src/benchmark.py`)
"""

import os
import inspect
from typing import Any, Generator, List

import numpy
from nptyping import NDArray
from keras import backend

from _environment import initialize_environment
import models_
from models_ import Detector, SSD
from utilities import print_debug


# Tensorflow imports
# pylint: disable=wrong-import-order
import tensorflow
from tensorflow import saved_model


# Model paths to be converted
SAVED_MODELS = ['model_data/ssdlitev2', 'model_data/ssdv2']

# Data paths to use sample data from when converting the model
DATA_PATH = 'data/COCO'
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'labels')


def __input_data_generator(
    model: Detector, image_files: List[str], annotation_files: List[str],
) -> Generator[List[NDArray[(Any, Any), numpy.float32]], None, None]:  # type: ignore
    for data_batch in model.data_generator(image_files, annotation_files):
        [processed_image], _ = model.preprocess_data(data_batch)
        yield [numpy.expand_dims(numpy.int32(processed_image), 0)]


def __quantize_model(converter: tensorflow.lite.TFLiteConverter, model: Detector) -> Any:
    print_debug('Converting model, this will take some time...')

    # Get sample data to be used during converstion
    image_files: List[str] = [os.path.abspath(os.path.join(IMAGES_PATH, image))
        for image in os.listdir(IMAGES_PATH)]
    annotation_files: List[str] = [os.path.abspath(os.path.join(ANNOTATIONS_PATH, annotations))
        for annotations in os.listdir(ANNOTATIONS_PATH)]

    try:
        # Set optimizations to perform on model to be compatible with edge TPUs
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: __input_data_generator(
            model,
            image_files,
            annotation_files
        )

        converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tensorflow.uint8
        converter.inference_output_type = tensorflow.uint8

        # Start conversion
        return converter.convert()
    except Exception as exception:  # pylint: disable=broad-except
        print_debug(f'Error: Could not convert model to TFLite')
        print_debug(str(exception))


def __write_to_file(file_name: str, model: Any) -> None:
    with open(file_name, 'wb') as model_file:
        model_file.write(model)
        print_debug(f'TensorFlow Lite model has been written to {file_name}')


def __convert_models():
    initialize_environment()

    # Convert all specified models
    for model_name in SAVED_MODELS:
        model_path = os.path.join(model_name, 'saved_model')

        with tensorflow.Session() as session:
            print_debug(f'\nLoading "{model_name}" from checkpoint...')

            # Load saved model to a new session
            tensorflow.saved_model.loader.load(
                session,
                [saved_model.tag_constants.SERVING],
                model_path,
            )

            # Get layers in model
            layers = [tensor for tensor in tensorflow.get_default_graph().get_operations()
                if tensor.type == 'Placeholder']

            # Create a model converter with the input shape given by the first layer
            saved_converter: Any = tensorflow.lite.TFLiteConverter.from_saved_model(
                model_path,
                input_shapes={
                    layers[0].name: [1, 300, 300, 3]
                }
            )

            # Quanize model using the converter and a generic detector (e.g. SSD) to generate sample
            # data
            saved_tflite_model = __quantize_model(saved_converter, SSD())
            __write_to_file(f'model_data/ssdv2.tflite', saved_tflite_model)

    # These models are not compatible with quantized TFLite models, so exclude them from the
    # conversion process
    excluded_class_names = ['RetinaNet', 'SqueezeDet', 'YOLOv3']

    print_debug('The following models are excluded from convesion,'
        ' because they are not TFLite compatible: ')
    print_debug(", ".join(excluded_class_names))

    # Get all the models from src/models_, which are not excluded (except the abstract Detector
    # class)
    excluded_class_names.append('Detector')
    models = {name: Model for name, Model in models_.__dict__.items()
        if inspect.isclass(Model) and issubclass(Model, Detector)
            and name not in excluded_class_names}  # noqa: W503

    # Convert all models found
    for model_name, model_class in models.items():
        # Initiate the model class
        model: Detector = model_class('TFLite Input')  # type: ignore

        # Get the current session created by the initiated class and create a converter from it
        converter = tensorflow.lite.TFLiteConverter.from_session(
            backend.get_session(),
            model.keras_model.inputs,
            model.keras_model.outputs
        )

        # Quantize model using the converter and the model to generate sample data
        tflite_model = __quantize_model(converter, model)
        __write_to_file(f'model_data/{model_name}.tflite', tflite_model)

    print_debug('\nExiting...')


if __name__ == '__main__':
    __convert_models()
