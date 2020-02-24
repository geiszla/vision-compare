from keras import Model
from PIL.Image import Image

from ..typings import PredictionResult
from .detector import Detector


# Model classes

class RetinaNet(Detector):
    def __init__(self):
        self.model = None
        super().__init__('RetinaNet with ResNet')

    def load_model(self) -> Model:
        from keras_retinanet.models import load_model

        model = load_model('model_data/retinanet.h5', backbone_name='resnet50')
        return model

    def detect_image(self, image: Image) -> PredictionResult:
        return self.keras_model.predict(image)
