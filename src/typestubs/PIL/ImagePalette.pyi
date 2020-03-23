"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

class ImagePalette:
    """
    Color palette for palette mapped images

    :param mode: The mode to use for the Palette. See:
        :ref:`concept-modes`. Defaults to "RGB"
    :param palette: An optional palette. If given, it must be a bytearray,
        an array or a list of ints between 0-255 and of length ``size``
        times the number of colors in ``mode``. The list must be aligned
        by channel (All R values must be contiguous in the list before G
        and B values.) Defaults to 0 through 255 per channel.
    :param size: An optional palette size. If given, it cannot be equal to
        or greater than 256. Defaults to 0.
    """
    def __init__(self, mode=..., palette: Optional[Any] = ..., size=...):
        self.mode = ...
        self.rawmode = ...
        self.palette = ...
        self.colors = ...
        self.dirty = ...
    
    def copy(self):
        ...
    
    def getdata(self):
        """
        Get palette contents in format suitable for the low-level
        ``im.putpalette`` primitive.

        .. warning:: This method is experimental.
        """
        ...
    
    def tobytes(self):
        """Convert palette to bytes.

        .. warning:: This method is experimental.
        """
        ...
    
    tostring = ...
    def getcolor(self, color):
        """Given an rgb tuple, allocate palette entry.

        .. warning:: This method is experimental.
        """
        ...
    
    def save(self, fp):
        """Save palette to text file.

        .. warning:: This method is experimental.
        """
        ...
    


def raw(rawmode, data):
    ...

def make_linear_lut(black, white):
    ...

def make_gamma_lut(exp):
    ...

def negative(mode=...):
    ...

def random(mode=...):
    ...

def sepia(white=...):
    ...

def wedge(mode=...):
    ...

def load(filename):
    ...

