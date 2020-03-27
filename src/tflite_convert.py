"""TensorFlow Lite conversion script
This script converts models in src/models_ directory and all the saved keras models specified in
SAVED_MODELS to quantized TensorFlow Lite models, so that they can be used on edge TPUs.

To be run from the project root (i.e. `python src/benchmark.py`)
"""

import os
import inspect
from typing import Any, cast, Generator, List, Tuple, Type

import numpy
import tensorflow
from nptyping import Array
from tensorflow import saved_model
from keras import backend

import models_
from models_ import Detector, SSD
from utilities import initialize_environment, print_debug


# Model paths to be converted
SAVED_MODELS = ['model_data/ssdlitev2', 'model_data/ssdv2']

# Data paths to use sample data from when converting the model
DATA_PATH = 'data/COCO'
IMAGES_PATH = os.path.join(DATA_PATH, 'images')
ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'labels')


def __input_data_generator(
    model: Detector, image_files: List[str], annotation_files: List[str],
) -> Generator[List[Array[numpy.float32, None, None]], None, None]:  # type: ignore
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


if __name__ == '__main__':
    initialize_environment()

    # Convert all specified models
    for MODEL_NAME in SAVED_MODELS:
        MODEL_PATH = os.path.join(MODEL_NAME, 'saved_model')

        with cast(Any, tensorflow.Session()) as SESSION:
            print_debug(f'\nLoading "{MODEL_NAME}" from checkpoint...')

            # Load saved model to a new session
            tensorflow.saved_model.loader.load(
                SESSION,
                [saved_model.tag_constants.SERVING],
                MODEL_PATH,
            )

            # Get layers in model
            LAYERS = [tensor for tensor in tensorflow.get_default_graph().get_operations()
                if tensor.type == 'Placeholder']

            # Create a model converter with the input shape given by the first layer
            SAVED_CONVERTER: Any = tensorflow.lite.TFLiteConverter.from_saved_model(
                MODEL_PATH,
                input_shapes={
                    LAYERS[0].name: [1, 300, 300, 3]
                }
            )

            # Quanize model using the converter and a generic detector (e.g. SSD) to generate sample
            # data
            SAVED_TFLITE_MODEL = __quantize_model(SAVED_CONVERTER, SSD())
            __write_to_file(f'model_data/ssdv2.tflite', SAVED_TFLITE_MODEL)

    # These models are not compatible with quantized TFLite models, so exclude them from the
    # conversion process
    EXCLUDED_CLASS_NAMES = ['RetinaNet', 'SqueezeDet', 'YOLOv3']

    print_debug('The following models are excluded from convesion,'
        ' because they are not TFLite compatible: ')
    print_debug(", ".join(EXCLUDED_CLASS_NAMES))

    # Get all the models from src/models_, which are not excluded (except the abstract Detector
    # class)
    EXCLUDED_CLASS_NAMES.append('Detector')
    MODELS = {name: Model for name, Model
        in cast(List[Tuple[str, Type[object]]], models_.__dict__.items())  # type: ignore
        if inspect.isclass(Model) and issubclass(Model, Detector)
            and name not in EXCLUDED_CLASS_NAMES}  # noqa: W503

    # Convert all models found
    for NAME, MODEL_CLASS in MODELS.items():
        # Initiate the model class
        MODEL: Detector = MODEL_CLASS('TFLite Input')  # type: ignore

        # Get the current session created by the initiated class and create a converter from it
        CONVERTER = tensorflow.lite.TFLiteConverter.from_session(
            backend.get_session(),
            MODEL.keras_model.inputs,
            MODEL.keras_model.outputs
        )

        # Quantize model using the converter and the model to generate sample data
        TFLITE_MODEL = __quantize_model(CONVERTER, MODEL)
        __write_to_file(f'model_data/{NAME}.tflite', TFLITE_MODEL)

    print_debug('\nExiting...')
