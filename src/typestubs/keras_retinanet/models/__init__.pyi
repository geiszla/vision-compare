"""
This type stub file was generated by pyright.
"""

import sys
from __future__ import print_function
from typing import Any, Optional

class Backbone(object):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone):
        self.custom_objects = ...
        self.backbone = ...
    
    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        ...
    
    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        ...
    
    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        ...
    
    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        ...
    


def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    ...

def load_model(filepath, backbone_name=...):
    """ Loads a retinanet model using the correct custom objects.

    Args
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    ...

def convert_model(model, nms: bool = ..., class_specific_filter: bool = ..., anchor_params: Optional[Any] = ...):
    """ Converts a training model to an inference model.

    Args
        model                 : A retinanet training model.
        nms                   : Boolean, whether to add NMS filtering to the converted model.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        anchor_params         : Anchor parameters object. If omitted, default values are used.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    ...

def assert_training_model(model):
    """ Assert that the model is a training model.
    """
    ...

def check_training_model(model):
    """ Check that model is a training model and exit otherwise.
    """
    ...
