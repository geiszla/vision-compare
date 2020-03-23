"""
This type stub file was generated by pyright.
"""

from numpy.core.overrides import set_module
from typing import Any, Optional

"""
Various richly-typed exceptions, that also help us deal with string formatting
in python where it's easier.

By putting the formatting in `__str__`, we also avoid paying the cost for
users who silence the exceptions.
"""
def _unpack_tuple(tup):
    ...

def _display_as_base(cls):
    """
    A decorator that makes an exception class look like its base.

    We use this to hide subclasses that are implementation details - the user
    should catch the base type, which is what the traceback will show them.

    Classes decorated with this decorator are subject to removal without a
    deprecation warning.
    """
    ...

class UFuncTypeError(TypeError):
    """ Base class for all ufunc exceptions """
    def __init__(self, ufunc):
        self.ufunc = ...
    


@_display_as_base
class _UFuncBinaryResolutionError(UFuncTypeError):
    """ Thrown when a binary resolution fails """
    def __init__(self, ufunc, dtypes):
        self.dtypes = ...
    
    def __str__(self):
        ...
    


@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """
    def __init__(self, ufunc, dtypes):
        self.dtypes = ...
    
    def __str__(self):
        ...
    


@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    def __init__(self, ufunc, casting, from_, to):
        self.casting = ...
        self.from_ = ...
        self.to = ...
    


@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc input cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        self.in_i = ...
    
    def __str__(self):
        ...
    


@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        self.out_i = ...
    
    def __str__(self):
        ...
    


@set_module('numpy')
class TooHardError(RuntimeError):
    ...


@set_module('numpy')
class AxisError(ValueError, IndexError):
    """ Axis supplied was invalid. """
    def __init__(self, axis, ndim: Optional[Any] = ..., msg_prefix: Optional[Any] = ...):
        ...
    


@_display_as_base
class _ArrayMemoryError(MemoryError):
    """ Thrown when an array cannot be allocated"""
    def __init__(self, shape, dtype):
        self.shape = ...
        self.dtype = ...
    
    @property
    def _total_size(self):
        ...
    
    @staticmethod
    def _size_to_string(num_bytes):
        """ Convert a number of bytes into a binary size string """
        ...
    
    def __str__(self):
        ...
    

