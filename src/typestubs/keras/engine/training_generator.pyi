"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

"""Part of the training engine related to Python generators of array data.
"""
def fit_generator(model, generator, steps_per_epoch: Optional[Any] = ..., epochs=..., verbose=..., callbacks: Optional[Any] = ..., validation_data: Optional[Any] = ..., validation_steps: Optional[Any] = ..., validation_freq=..., class_weight: Optional[Any] = ..., max_queue_size=..., workers=..., use_multiprocessing: bool = ..., shuffle: bool = ..., initial_epoch=...):
    """See docstring for `Model.fit_generator`."""
    ...

def evaluate_generator(model, generator, steps: Optional[Any] = ..., callbacks: Optional[Any] = ..., max_queue_size=..., workers=..., use_multiprocessing: bool = ..., verbose=...):
    """See docstring for `Model.evaluate_generator`."""
    ...

def predict_generator(model, generator, steps: Optional[Any] = ..., callbacks: Optional[Any] = ..., max_queue_size=..., workers=..., use_multiprocessing: bool = ..., verbose=...):
    """See docstring for `Model.predict_generator`."""
    ...

