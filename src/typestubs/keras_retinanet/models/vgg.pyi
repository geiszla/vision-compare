"""
This type stub file was generated by pyright.
"""

from . import Backbone
from typing import Any, Optional

"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """
    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        ...
    
    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        ...
    
    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        ...
    
    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        ...
    


def vgg_retinanet(num_classes, backbone=..., inputs: Optional[Any] = ..., modifier: Optional[Any] = ..., **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    ...
