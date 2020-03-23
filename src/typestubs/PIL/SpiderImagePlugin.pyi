"""
This type stub file was generated by pyright.
"""

import sys
from PIL import Image, ImageFile
from typing import Any, Optional

def isInt(f):
    ...

iforms = [1, 3, - 11, - 12, - 21, - 22]
def isSpiderHeader(t):
    ...

def isSpiderImage(filename):
    ...

class SpiderImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    _close_exclusive_fp_after_loading = ...
    def _open(self):
        self.istack = ...
        self.imgnumber = ...
        self.mode = ...
        self.tile = ...
    
    @property
    def n_frames(self):
        ...
    
    @property
    def is_animated(self):
        ...
    
    def tell(self):
        ...
    
    def seek(self, frame):
        self.stkoffset = ...
        self.fp = ...
    
    def convert2byte(self, depth=...):
        ...
    
    def tkPhotoImage(self):
        ...
    
    def _close__fp(self):
        ...
    


def loadImageSeries(filelist: Optional[Any] = ...):
    """create a list of :py:class:`~PIL.Image.Image` objects for use in a montage"""
    ...

def makeSpiderHeader(im):
    ...

def _save(im, fp, filename):
    ...

def _save_spider(im, fp, filename):
    ...

if __name__ == "__main__":
    filename = sys.argv[1]
    im = Image.open(filename)
