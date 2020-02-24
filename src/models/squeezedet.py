from keras import Model
from PIL.Image import Image

from ..typings import PredictionResult
from .detector import Detector


# Model classes

class SqueezeDet(Detector):
    def __init__(self):
        self.model = None
        super().__init__('SqueezeDet')

    def load_model(self) -> Model:
        from lib.squeezedet_keras.main.config.create_config import squeezeDet_config
        from lib.squeezedet_keras.main.model.squeezeDet import SqueezeDet as SqueezeDetModel

        self.model = SqueezeDetModel(squeezeDet_config('vision_compare'))

        keras_model = self.model.model
        keras_model.load_weights('model_data/squeezedet.h5')

        return keras_model

    def detect_image(self, image: Image) -> PredictionResult:
        return self.keras_model.predict(image)
