"""
This type stub file was generated by pyright.
"""

from . import ImageFile

b_whitespace = b"\x20\x09\x0a\x0b\x0c\x0d"
MODES = { b"P4": "1",b"P5": "L",b"P6": "RGB",b"P0CMYK": "CMYK",b"PyP": "P",b"PyRGBA": "RGBA",b"PyCMYK": "CMYK" }
def _accept(prefix):
    ...

class PpmImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    def _token(self, s=...):
        ...
    
    def _open(self):
        self.custom_mimetype = ...
        self.tile = ...
    


def _save(im, fp, filename):
    ...
