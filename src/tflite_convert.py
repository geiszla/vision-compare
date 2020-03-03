import inspect

import tensorflow
from keras import backend

import models_
from models_ import Detector
from utilities import initialize_environment, print_debug


MODEL_DATA_PATH = 'model_data'


if __name__ == '__main__':
    initialize_environment()

    EXCLUDED_CLASS_NAMES = ['RetinaNet', 'SqueezeDet', 'SSDv1']

    print_debug('The following models are excluded from convesion,'
        ' because they are not TFLite compatible: ')
    print_debug(", ".join(EXCLUDED_CLASS_NAMES))

    EXCLUDED_CLASS_NAMES.append('Detector')
    MODELS = {name: Model for name, Model in models_.__dict__.items()
        if inspect.isclass(Model) and issubclass(Model, Detector)
            and name not in EXCLUDED_CLASS_NAMES}  # noqa: W503

    for name, Model in MODELS.items():
        model: Detector = Model()

        print_debug('Converting model...')

        try:
            converter = tensorflow.lite.TFLiteConverter.from_session(
                backend.get_session(),
                model.keras_model.inputs,
                model.keras_model.outputs,
            )
            tflite_model = converter.convert()
        except Exception as exception:  # pylint: disable=broad-except
            print_debug(f'Error: Could not convert model to TFLite')
            print_debug(str(exception))

        tflite_file_name = f'model_data/{name}.tflite'
        with open(tflite_file_name, 'wb') as model_file:
            model_file.write(tflite_model)
            print_debug(f'TensorFlow Lite model has been written to {tflite_file_name}')

    print_debug('\nExiting...')
