"""
This type stub file was generated by pyright.
"""

from . import ImageFile

class PcdImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    def _open(self):
        self.tile_post_rotate = ...
        self.mode = ...
        self.tile = ...
    
    def load_end(self):
        ...
    

