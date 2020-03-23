"""
This type stub file was generated by pyright.
"""

from .base_layer import Layer
from ..legacy import interfaces
from typing import Any, Optional

"""Input layer code (`Input` and `InputLayer`).
"""
class InputLayer(Layer):
    """Layer to be used as an entry point into a model.

    It can either wrap an existing tensor (pass an `input_tensor` argument)
    or create its a placeholder tensor (pass arguments `input_shape`
    or `batch_input_shape` as well as `dtype`).

    # Arguments
        input_shape: Shape tuple, not including the batch axis.
        batch_size: Optional input batch size (integer or None).
        batch_input_shape: Shape tuple, including the batch axis.
        dtype: Datatype of the input.
        input_tensor: Optional tensor to use as layer input
            instead of creating a placeholder.
        sparse: Boolean, whether the placeholder created
            is meant to be sparse.
        name: Name of the layer (string).
    """
    @interfaces.legacy_input_support
    def __init__(self, input_shape: Optional[Any] = ..., batch_size: Optional[Any] = ..., batch_input_shape: Optional[Any] = ..., dtype: Optional[Any] = ..., input_tensor: Optional[Any] = ..., sparse: bool = ..., name: Optional[Any] = ...):
        self.trainable = ...
        self.built = ...
        self.sparse = ...
        self.supports_masking = ...
        self.batch_input_shape = ...
        self.dtype = ...
    
    def get_config(self):
        ...
    


def Input(shape: Optional[Any] = ..., batch_shape: Optional[Any] = ..., name: Optional[Any] = ..., dtype: Optional[Any] = ..., sparse: bool = ..., tensor: Optional[Any] = ...):
    """`Input()` is used to instantiate a Keras tensor.

    A Keras tensor is a tensor object from the underlying backend
    (Theano, TensorFlow or CNTK), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.

    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`

    The added Keras attributes are:
        `_keras_shape`: Integer shape tuple propagated
            via Keras-side shape inference.
        `_keras_history`: Last layer applied to the tensor.
            the entire layer graph is retrievable from that layer,
            recursively.

    # Arguments
        shape: A shape tuple (integer), not including the batch size.
            For instance, `shape=(32,)` indicates that the expected input
            will be batches of 32-dimensional vectors.
        batch_shape: A shape tuple (integer), including the batch size.
            For instance, `batch_shape=(10, 32)` indicates that
            the expected input will be batches of 10 32-dimensional vectors.
            `batch_shape=(None, 32)` indicates batches of an arbitrary number
            of 32-dimensional vectors.
        name: An optional name string for the layer.
            Should be unique in a model (do not reuse the same name twice).
            It will be autogenerated if it isn't provided.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
        sparse: A boolean specifying whether the placeholder
            to be created is sparse.
        tensor: Optional existing tensor to wrap into the `Input` layer.
            If set, the layer will not create a placeholder tensor.

    # Returns
        A tensor.

    # Example

    ```python
    # this is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    """
    ...
