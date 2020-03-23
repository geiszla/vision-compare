"""
This type stub file was generated by pyright.
"""

import tkinter
from typing import Any, Optional

_pilbitmap_ok = None
def _pilbitmap_check():
    ...

def _get_image_from_kw(kw):
    ...

class PhotoImage:
    """
    A Tkinter-compatible photo image.  This can be used
    everywhere Tkinter expects an image object.  If the image is an RGBA
    image, pixels having alpha 0 are treated as transparent.

    The constructor takes either a PIL image, or a mode and a size.
    Alternatively, you can use the **file** or **data** options to initialize
    the photo image object.

    :param image: Either a PIL image, or a mode string.  If a mode string is
                  used, a size must also be given.
    :param size: If the first argument is a mode string, this defines the size
                 of the image.
    :keyword file: A filename to load the image from (using
                   ``Image.open(file)``).
    :keyword data: An 8-bit string containing image data (as loaded from an
                   image file).
    """
    def __init__(self, image: Optional[Any] = ..., size: Optional[Any] = ..., **kw):
        self.tk = ...
    
    def __del__(self):
        ...
    
    def __str__(self):
        """
        Get the Tkinter photo image identifier.  This method is automatically
        called by Tkinter whenever a PhotoImage object is passed to a Tkinter
        method.

        :return: A Tkinter photo image identifier (a string).
        """
        ...
    
    def width(self):
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        ...
    
    def height(self):
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        ...
    
    def paste(self, im, box: Optional[Any] = ...):
        """
        Paste a PIL image into the photo image.  Note that this can
        be very slow if the photo image is displayed.

        :param im: A PIL image. The size must match the target region.  If the
                   mode does not match, the image is converted to the mode of
                   the bitmap image.
        :param box: A 4-tuple defining the left, upper, right, and lower pixel
                    coordinate. See :ref:`coordinate-system`. If None is given
                    instead of a tuple, all of the image is assumed.
        """
        ...
    


class BitmapImage:
    """
    A Tkinter-compatible bitmap image.  This can be used everywhere Tkinter
    expects an image object.

    The given image must have mode "1".  Pixels having value 0 are treated as
    transparent.  Options, if any, are passed on to Tkinter.  The most commonly
    used option is **foreground**, which is used to specify the color for the
    non-transparent parts.  See the Tkinter documentation for information on
    how to specify colours.

    :param image: A PIL image.
    """
    def __init__(self, image: Optional[Any] = ..., **kw):
        ...
    
    def __del__(self):
        ...
    
    def width(self):
        """
        Get the width of the image.

        :return: The width, in pixels.
        """
        ...
    
    def height(self):
        """
        Get the height of the image.

        :return: The height, in pixels.
        """
        ...
    
    def __str__(self):
        """
        Get the Tkinter bitmap image identifier.  This method is automatically
        called by Tkinter whenever a BitmapImage object is passed to a Tkinter
        method.

        :return: A Tkinter bitmap image identifier (a string).
        """
        ...
    


def getimage(photo):
    """Copies the contents of a PhotoImage to a PIL image memory."""
    ...

def _show(image, title):
    """Helper for the Image.show method."""
    class UI(tkinter.Label):
        ...
    
    

