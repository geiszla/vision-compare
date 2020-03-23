"""
This type stub file was generated by pyright.
"""

import os
from .common import epsilon, floatx, image_data_format
from .cntk_backend import *
from .theano_backend import *

if 'KERAS_HOME' in os.environ:
    _keras_dir = os.environ.get('KERAS_HOME')
else:
    _keras_base_dir = os.path.expanduser('~')
    _keras_dir = os.path.join(_keras_base_dir, '.keras')
_BACKEND = 'tensorflow'
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
    _floatx = _config.get('floatx', floatx())
    _epsilon = _config.get('epsilon', epsilon())
    _backend = _config.get('backend', _BACKEND)
    _image_data_format = _config.get('image_data_format', image_data_format())
    _BACKEND = _backend
if not os.path.exists(_keras_dir):
    ...
if not os.path.exists(_config_path):
    _config = { 'floatx': floatx(),'epsilon': epsilon(),'backend': _BACKEND,'image_data_format': image_data_format() }
if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ['KERAS_BACKEND']
if _BACKEND == 'cntk':
    ...
else:
    ...
def backend():
    """Returns the name of the current backend (e.g. "tensorflow").

    # Returns
        String, the name of the backend Keras is currently using.

    # Example
    ```python
        >>> keras.backend.backend()
        'tensorflow'
    ```
    """
    ...
