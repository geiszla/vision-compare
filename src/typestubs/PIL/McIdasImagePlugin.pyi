"""
This type stub file was generated by pyright.
"""

from . import ImageFile

def _accept(s):
    ...

class McIdasImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    def _open(self):
        self.area_descriptor_raw = ...
        self.area_descriptor = ...
        self.mode = ...
        self.tile = ...
    

