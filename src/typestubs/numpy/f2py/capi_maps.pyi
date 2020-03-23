"""
This type stub file was generated by pyright.
"""

import copy
import sys
from . import __version__
from .auxfuncs import *
from typing import Any, Optional

"""

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 10:57:33 $
Pearu Peterson

"""
__version__ = "$Revision: 1.60 $"[10: - 1]
f2py_version = __version__.version
__all__ = ['getctype', 'getstrlength', 'getarrdims', 'getpydocsign', 'getarrdocsign', 'getinit', 'sign2map', 'routsign2map', 'modsign2map', 'cb_sign2map', 'cb_routsign2map', 'common_sign2map']
using_newcore = True
depargs = []
lcb_map = {  }
lcb2_map = {  }
c2py_map = { 'double': 'float','float': 'float','long_double': 'float','char': 'int','signed_char': 'int','unsigned_char': 'int','short': 'int','unsigned_short': 'int','int': 'int','long': 'int','long_long': 'long','unsigned': 'int','complex_float': 'complex','complex_double': 'complex','complex_long_double': 'complex','string': 'string' }
c2capi_map = { 'double': 'NPY_DOUBLE','float': 'NPY_FLOAT','long_double': 'NPY_DOUBLE','char': 'NPY_STRING','unsigned_char': 'NPY_UBYTE','signed_char': 'NPY_BYTE','short': 'NPY_SHORT','unsigned_short': 'NPY_USHORT','int': 'NPY_INT','unsigned': 'NPY_UINT','long': 'NPY_LONG','long_long': 'NPY_LONG','complex_float': 'NPY_CFLOAT','complex_double': 'NPY_CDOUBLE','complex_long_double': 'NPY_CDOUBLE','string': 'NPY_STRING' }
if using_newcore:
    c2capi_map = { 'double': 'NPY_DOUBLE','float': 'NPY_FLOAT','long_double': 'NPY_LONGDOUBLE','char': 'NPY_BYTE','unsigned_char': 'NPY_UBYTE','signed_char': 'NPY_BYTE','short': 'NPY_SHORT','unsigned_short': 'NPY_USHORT','int': 'NPY_INT','unsigned': 'NPY_UINT','long': 'NPY_LONG','unsigned_long': 'NPY_ULONG','long_long': 'NPY_LONGLONG','unsigned_long_long': 'NPY_ULONGLONG','complex_float': 'NPY_CFLOAT','complex_double': 'NPY_CDOUBLE','complex_long_double': 'NPY_CDOUBLE','string': 'NPY_STRING' }
c2pycode_map = { 'double': 'd','float': 'f','long_double': 'd','char': '1','signed_char': '1','unsigned_char': 'b','short': 's','unsigned_short': 'w','int': 'i','unsigned': 'u','long': 'l','long_long': 'L','complex_float': 'F','complex_double': 'D','complex_long_double': 'D','string': 'c' }
if using_newcore:
    c2pycode_map = { 'double': 'd','float': 'f','long_double': 'g','char': 'b','unsigned_char': 'B','signed_char': 'b','short': 'h','unsigned_short': 'H','int': 'i','unsigned': 'I','long': 'l','unsigned_long': 'L','long_long': 'q','unsigned_long_long': 'Q','complex_float': 'F','complex_double': 'D','complex_long_double': 'G','string': 'S' }
c2buildvalue_map = { 'double': 'd','float': 'f','char': 'b','signed_char': 'b','short': 'h','int': 'i','long': 'l','long_long': 'L','complex_float': 'N','complex_double': 'N','complex_long_double': 'N','string': 'z' }
if sys.version_info[0] >= 3:
    ...
if using_newcore:
    ...
f2cmap_all = { 'real': { '': 'float','4': 'float','8': 'double','12': 'long_double','16': 'long_double' },'integer': { '': 'int','1': 'signed_char','2': 'short','4': 'int','8': 'long_long','-1': 'unsigned_char','-2': 'unsigned_short','-4': 'unsigned','-8': 'unsigned_long_long' },'complex': { '': 'complex_float','8': 'complex_float','16': 'complex_double','24': 'complex_long_double','32': 'complex_long_double' },'complexkind': { '': 'complex_float','4': 'complex_float','8': 'complex_double','12': 'complex_long_double','16': 'complex_long_double' },'logical': { '': 'int','1': 'char','2': 'short','4': 'int','8': 'long_long' },'double complex': { '': 'complex_double' },'double precision': { '': 'double' },'byte': { '': 'char' },'character': { '': 'string' } }
f2cmap_default = copy.deepcopy(f2cmap_all)
def load_f2cmap_file(f2cmap_file):
    ...

cformat_map = { 'double': '%g','float': '%g','long_double': '%Lg','char': '%d','signed_char': '%d','unsigned_char': '%hhu','short': '%hd','unsigned_short': '%hu','int': '%d','unsigned': '%u','long': '%ld','unsigned_long': '%lu','long_long': '%ld','complex_float': '(%g,%g)','complex_double': '(%g,%g)','complex_long_double': '(%Lg,%Lg)','string': '%s' }
def getctype(var):
    """
    Determines C type
    """
    ...

def getstrlength(var):
    ...

def getarrdims(a, var, verbose=...):
    ...

def getpydocsign(a, var):
    ...

def getarrdocsign(a, var):
    ...

def getinit(a, var):
    ...

def sign2map(a, var):
    """
    varname,ctype,atype
    init,init.r,init.i,pytype
    vardebuginfo,vardebugshowvalue,varshowvalue
    varrfromat
    intent
    """
    ...

def routsign2map(rout):
    """
    name,NAME,begintitle,endtitle
    rname,ctype,rformat
    routdebugshowvalue
    """
    ...

def modsign2map(m):
    """
    modulename
    """
    ...

def cb_sign2map(a, var, index: Optional[Any] = ...):
    ...

def cb_routsign2map(rout, um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    ...

def common_sign2map(a, var):
    ...

