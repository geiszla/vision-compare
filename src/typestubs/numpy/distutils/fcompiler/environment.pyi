"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

__metaclass__ = type
class EnvironmentConfig(object):
    def __init__(self, distutils_section=..., **kw):
        ...
    
    def dump_variable(self, name):
        ...
    
    def dump_variables(self):
        ...
    
    def __getattr__(self, name):
        ...
    
    def get(self, name, default: Optional[Any] = ...):
        ...
    
    def _get_var(self, name, conf_desc):
        ...
    
    def clone(self, hook_handler):
        ...
    
    def use_distribution(self, dist):
        ...
    

