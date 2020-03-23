"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

def constant(image, value):
    """Fill a channel with a given grey level.

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def duplicate(image):
    """Copy a channel. Alias for :py:meth:`PIL.Image.Image.copy`.

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def invert(image):
    """
    Invert an image (channel).

    .. code-block:: python

        out = MAX - image

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def lighter(image1, image2):
    """
    Compares the two images, pixel by pixel, and returns a new image containing
    the lighter values. At least one of the images must have mode "1".

    .. code-block:: python

        out = max(image1, image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def darker(image1, image2):
    """
    Compares the two images, pixel by pixel, and returns a new image containing
    the darker values. At least one of the images must have mode "1".

    .. code-block:: python

        out = min(image1, image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def difference(image1, image2):
    """
    Returns the absolute value of the pixel-by-pixel difference between the two
    images. At least one of the images must have mode "1".

    .. code-block:: python

        out = abs(image1 - image2)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def multiply(image1, image2):
    """
    Superimposes two images on top of each other.

    If you multiply an image with a solid black image, the result is black. If
    you multiply with a solid white image, the image is unaffected. At least
    one of the images must have mode "1".

    .. code-block:: python

        out = image1 * image2 / MAX

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def screen(image1, image2):
    """
    Superimposes two inverted images on top of each other. At least one of the
    images must have mode "1".

    .. code-block:: python

        out = MAX - ((MAX - image1) * (MAX - image2) / MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def add(image1, image2, scale=..., offset=...):
    """
    Adds two images, dividing the result by scale and adding the
    offset. If omitted, scale defaults to 1.0, and offset to 0.0.
    At least one of the images must have mode "1".

    .. code-block:: python

        out = ((image1 + image2) / scale + offset)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def subtract(image1, image2, scale=..., offset=...):
    """
    Subtracts two images, dividing the result by scale and adding the offset.
    If omitted, scale defaults to 1.0, and offset to 0.0. At least one of the
    images must have mode "1".

    .. code-block:: python

        out = ((image1 - image2) / scale + offset)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def add_modulo(image1, image2):
    """Add two images, without clipping the result. At least one of the images
    must have mode "1".

    .. code-block:: python

        out = ((image1 + image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def subtract_modulo(image1, image2):
    """Subtract two images, without clipping the result. At least one of the
    images must have mode "1".

    .. code-block:: python

        out = ((image1 - image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def logical_and(image1, image2):
    """Logical AND between two images. At least one of the images must have
    mode "1".

    .. code-block:: python

        out = ((image1 and image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def logical_or(image1, image2):
    """Logical OR between two images. At least one of the images must have
    mode "1".

    .. code-block:: python

        out = ((image1 or image2) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def logical_xor(image1, image2):
    """Logical XOR between two images. At least one of the images must have
    mode "1".

    .. code-block:: python

        out = ((bool(image1) != bool(image2)) % MAX)

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def blend(image1, image2, alpha):
    """Blend images using constant transparency weight. Alias for
    :py:meth:`PIL.Image.Image.blend`.

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def composite(image1, image2, mask):
    """Create composite using transparency mask. Alias for
    :py:meth:`PIL.Image.Image.composite`.

    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...

def offset(image, xoffset, yoffset: Optional[Any] = ...):
    """Returns a copy of the image where data has been offset by the given
    distances. Data wraps around the edges. If **yoffset** is omitted, it
    is assumed to be equal to **xoffset**.

    :param xoffset: The horizontal distance.
    :param yoffset: The vertical distance.  If omitted, both
        distances are set to the same value.
    :rtype: :py:class:`~PIL.Image.Image`
    """
    ...
