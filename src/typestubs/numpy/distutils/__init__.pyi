"""
This type stub file was generated by pyright.
"""

from __future__ import absolute_import, division, print_function
from . import ccompiler, unixccompiler
from .npy_pkg_config import *
from typing import Any, Optional

"""
An enhanced distutils, providing support for Fortran compilers, for BLAS,
LAPACK and other common libraries for numerical computing, and more.

Public submodules are::

    misc_util
    system_info
    cpu_info
    log
    exec_command

For details, please see the *Packaging* and *NumPy Distutils User Guide*
sections of the NumPy Reference Guide.

For configuring the preference for and location of libraries like BLAS and
LAPACK, and for setting include paths and similar build options, please see
``site.cfg.example`` in the root of the NumPy repository or sdist.

"""
def customized_fcompiler(plat: Optional[Any] = ..., compiler: Optional[Any] = ...):
    ...

def customized_ccompiler(plat: Optional[Any] = ..., compiler: Optional[Any] = ..., verbose=...):
    ...

