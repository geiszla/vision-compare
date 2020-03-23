"""
This type stub file was generated by pyright.
"""

from . import ContainerIO

class TarIO(ContainerIO.ContainerIO):
    def __init__(self, tarfile, file):
        """
        Create file object.

        :param tarfile: Name of TAR file.
        :param file: Name of member file.
        """
        self.fh = ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *args):
        ...
    
    def close(self):
        ...
    


