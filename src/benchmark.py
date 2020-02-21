import sys
from os import path, chdir

from keras import backend
from keras_retinanet.models import load_model
from PIL import Image

from __init__ import initialize_environment, PROJECT_PATH
initialize_environment()


def benchmark():
    from utilities import get_image_data, process_predictions, print_boxes, print_debug

    from lib.keras_yolo3.yolo import YOLO
    from lib.squeezedet_keras.main.config.create_config import create_config_from_dict
    from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet
    from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import Model

    image_path = r'D:\Users\Andris\Desktop\vision-compare\data\COCO\images\000000000110.jpg'
    image = Image.open(image_path)

    # YOLOv3
    yolo_directory = path.join(PROJECT_PATH, 'lib/keras_yolo3')
    chdir(yolo_directory)
    print_debug('Changed to YOLOv3 directory: ' + yolo_directory)

    print_debug('Loading YOLOv3 model...')
    model = YOLO(**{'model_path': path.join(PROJECT_PATH, 'model_data/yolov3.h5')})
    image_data = get_image_data(image, model.model_image_size)  # pylint: disable=no-member

    print_debug('Running predictions on "{}"'.format(image_path))
    predictions = model.sess.run(
        [model.boxes, model.scores, model.classes],
        feed_dict={
            model.yolo_model.input: image_data,
            model.input_image_shape: [image.size[1], image.size[0]],
            backend.learning_phase(): 0
        }
    )

    print_debug('{} boxes found'.format(len(predictions[0])))
    processed_predictions = process_predictions(predictions, model.class_names, image)
    print_boxes(processed_predictions)

    print_debug('Exiting...')
    sys.exit()

    # SqueezeDet
    model = SqueezeDet(create_config_from_dict())

    # SSD
    model = Model((300, 300, 3), 2)

    # RetinaNet
    model = load_model('model_data/retinanet.h5', backbone_name='resnet50')


benchmark()
