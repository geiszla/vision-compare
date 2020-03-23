"""
This type stub file was generated by pyright.
"""

from . import ImageFile

MODES = { (0, 1): ("1", 1),(0, 8): ("L", 1),(1, 8): ("L", 1),(2, 8): ("P", 1),(3, 8): ("RGB", 3),(4, 8): ("CMYK", 4),(7, 8): ("L", 1),(8, 8): ("L", 1),(9, 8): ("LAB", 3) }
def _accept(prefix):
    ...

class PsdImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    _close_exclusive_fp_after_loading = ...
    def _open(self):
        self.mode = ...
        self.resources = ...
        self.layers = ...
        self.tile = ...
        self.frame = ...
    
    @property
    def n_frames(self):
        ...
    
    @property
    def is_animated(self):
        ...
    
    def seek(self, layer):
        ...
    
    def tell(self):
        ...
    
    def load_prepare(self):
        ...
    
    def _close__fp(self):
        ...
    


def _layerinfo(file):
    ...

def _maketile(file, mode, bbox, channels):
    ...

