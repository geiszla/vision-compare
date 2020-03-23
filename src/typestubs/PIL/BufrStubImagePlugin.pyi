"""
This type stub file was generated by pyright.
"""

from . import ImageFile

_handler = None
def register_handler(handler):
    """
    Install application-specific BUFR image handler.

    :param handler: Handler object.
    """
    ...

def _accept(prefix):
    ...

class BufrStubImageFile(ImageFile.StubImageFile):
    format = ...
    format_description = ...
    def _open(self):
        self.mode = ...
    
    def _load(self):
        ...
    


def _save(im, fp, filename):
    ...

