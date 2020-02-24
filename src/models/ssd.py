from keras import Model
from PIL.Image import Image

from ..typings import PredictionResult
from .detector import Detector


# Model classes

class SSD(Detector):
    def __init__(self):
        self.model = None
        super().__init__('SSD with MobileNetv2')

    def load_model(self) -> Model:
        from lib.ssd_kerasV2.model.ssd300MobileNetV2Lite import Model as SSDModel

        model = SSDModel((300, 300, 3), 2)
        # TODO: Load model weights
        return model

    def detect_image(self, image: Image) -> PredictionResult:
        return self.keras_model.predict(image)
