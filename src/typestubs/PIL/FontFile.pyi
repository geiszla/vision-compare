"""
This type stub file was generated by pyright.
"""

WIDTH = 800
def puti16(fp, values):
    ...

class FontFile:
    bitmap = ...
    def __init__(self):
        self.info = ...
        self.glyph = ...
    
    def __getitem__(self, ix):
        ...
    
    def compile(self):
        """Create metrics and bitmap"""
        self.ysize = ...
        self.bitmap = ...
        self.metrics = ...
    
    def save(self, filename):
        """Save font"""
        ...
    


