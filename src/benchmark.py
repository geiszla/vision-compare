from keras_retinanet.models import load_model

from lib.keras_yolo3.yolo import YOLO
from lib.squeezedet_keras.main.config.create_config import create_config_from_dict
from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet
from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import Model


YOLO._defaults['model_path'] = 'model_data/yolov3.h5' # pylint: disable=protected-access

# YOLOv3
MODEL = YOLO()

# SSD
MODEL = Model((300, 300, 3), 2)

# RetinaNet
MODEL = load_model('model_data/retinanet.h5', backbone_name='resnet50')

# SqueezeDet
MODEL = SqueezeDet(create_config_from_dict())
