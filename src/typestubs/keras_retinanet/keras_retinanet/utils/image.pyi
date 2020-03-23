"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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
def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    ...

def preprocess_image(x, mode=...):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    ...

def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    ...

class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(self, fill_mode=..., interpolation=..., cval=..., relative_translation: bool = ...):
        self.fill_mode = ...
        self.cval = ...
        self.interpolation = ...
        self.relative_translation = ...
    
    def cvBorderMode(self):
        ...
    
    def cvInterpolation(self):
        ...
    


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    ...

def compute_resize_scale(image_shape, min_side=..., max_side=...):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    ...

def resize_image(img, min_side=..., max_side=...):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    ...

def _uniform(val_range):
    """ Uniformly sample from the given range.

    Args
        val_range: A pair of lower and upper bound.
    """
    ...

def _check_range(val_range, min_val: Optional[Any] = ..., max_val: Optional[Any] = ...):
    """ Check whether the range is a valid range.

    Args
        val_range: A pair of lower and upper bound.
        min_val: Minimal value for the lower bound.
        max_val: Maximal value for the upper bound.
    """
    ...

def _clip(image):
    """
    Clip and convert an image to np.uint8.

    Args
        image: Image to clip.
    """
    ...

class VisualEffect:
    """ Struct holding parameters and applying image color transformation.

    Args
        contrast_factor:   A factor for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  Brightness offset between -1 and 1 added to the pixel values.
        hue_delta:         Hue offset between -1 and 1 added to the hue channel.
        saturation_factor: A factor multiplying the saturation values of each pixel.
    """
    def __init__(self, contrast_factor, brightness_delta, hue_delta, saturation_factor):
        self.contrast_factor = ...
        self.brightness_delta = ...
        self.hue_delta = ...
        self.saturation_factor = ...
    
    def __call__(self, image):
        """ Apply a visual effect on the image.

        Args
            image: Image to adjust
        """
        ...
    


def random_visual_effect_generator(contrast_range=..., brightness_range=..., hue_range=..., saturation_range=...):
    """ Generate visual effect parameters uniformly sampled from the given intervals.

    Args
        contrast_factor:   A factor interval for adjusting contrast. Should be between 0 and 3.
        brightness_delta:  An interval between -1 and 1 for the amount added to the pixels.
        hue_delta:         An interval between -1 and 1 for the amount added to the hue channel.
                           The values are rotated if they exceed 180.
        saturation_factor: An interval for the factor multiplying the saturation values of each
                           pixel.
    """
    ...

def adjust_contrast(image, factor):
    """ Adjust contrast of an image.

    Args
        image: Image to adjust.
        factor: A factor for adjusting contrast.
    """
    ...

def adjust_brightness(image, delta):
    """ Adjust brightness of an image

    Args
        image: Image to adjust.
        delta: Brightness offset between -1 and 1 added to the pixel values.
    """
    ...

def adjust_hue(image, delta):
    """ Adjust hue of an image.

    Args
        image: Image to adjust.
        delta: An interval between -1 and 1 for the amount added to the hue channel.
               The values are rotated if they exceed 180.
    """
    ...

def adjust_saturation(image, factor):
    """ Adjust saturation of an image.

    Args
        image: Image to adjust.
        factor: An interval for the factor multiplying the saturation values of each pixel.
    """
    ...
