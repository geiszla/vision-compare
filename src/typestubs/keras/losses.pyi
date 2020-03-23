"""
This type stub file was generated by pyright.
"""

import abc
import six
from typing import Any, Optional

"""Built-in loss functions.
"""
@six.add_metaclass(abc.ABCMeta)
class Loss(object):
    """Loss base class.

    To be implemented by subclasses:
        * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

    Example subclass implementation:
    ```python
    class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    ```

    # Arguments
        reduction: (Optional) Type of loss Reduction to apply to loss.
          Default value is `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the object.
    """
    def __init__(self, reduction=..., name: Optional[Any] = ...):
        self.reduction = ...
        self.name = ...
    
    def __call__(self, y_true, y_pred, sample_weight: Optional[Any] = ...):
        """Invokes the `Loss` instance.

        # Arguments
            y_true: Ground truth values.
            y_pred: The predicted values.
            sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
            as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
            coefficient for the loss. If a scalar is provided, then the loss is
            simply scaled by the given value. If `sample_weight` is a tensor of size
            `[batch_size]`, then the total loss for each sample of the batch is
            rescaled by the corresponding element in the `sample_weight` vector. If
            the shape of `sample_weight` matches the shape of `y_pred`, then the
            loss of each measurable element of `y_pred` is scaled by the
            corresponding value of `sample_weight`.

        # Returns
            Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
                shape as `y_true`; otherwise, it is scalar.

        # Raises
            ValueError: If the shape of `sample_weight` is invalid.
        """
        ...
    
    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).

        # Arguments
            config: Output of `get_config()`.

        # Returns
            A `Loss` instance.
        """
        ...
    
    def get_config(self):
        ...
    
    @abc.abstractmethod
    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

        # Arguments
            y_true: Ground truth values, with the same shape as 'y_pred'.
            y_pred: The predicted values.
        """
        ...
    


class LossFunctionWrapper(Loss):
    """Wraps a loss function in the `Loss` class.

    # Arguments
        fn: The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
        **kwargs: The keyword arguments that are passed on to `fn`.
    """
    def __init__(self, fn, reduction=..., name: Optional[Any] = ..., **kwargs):
        self.fn = ...
    
    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.

        # Arguments
            y_true: Ground truth values.
            y_pred: The predicted values.

        # Returns
            Loss values per sample.
        """
        ...
    
    def get_config(self):
        ...
    


class MeanSquaredError(LossFunctionWrapper):
    """Computes the mean of squares of errors between labels and predictions.

    Standalone usage:

    ```python
    mse = keras.losses.MeanSquaredError()
    loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanSquaredError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class MeanAbsoluteError(LossFunctionWrapper):
    """Computes the mean of absolute difference between labels and predictions.

    Standalone usage:

    ```python
    mae = keras.losses.MeanAbsoluteError()
    loss = mae([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanAbsoluteError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class MeanAbsolutePercentageError(LossFunctionWrapper):
    """Computes the mean absolute percentage error between `y_true` and `y_pred`.

    Standalone usage:

    ```python
    mape = keras.losses.MeanAbsolutePercentageError()
    loss = mape([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanAbsolutePercentageError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class MeanSquaredLogarithmicError(LossFunctionWrapper):
    """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

    Standalone usage:

    ```python
    msle = keras.losses.MeanSquaredLogarithmicError()
    loss = msle([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.MeanSquaredLogarithmicError())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class BinaryCrossentropy(LossFunctionWrapper):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss when there are only two label classes (assumed to
    be 0 and 1). For each example, there should be a single floating-point value
    per prediction.

    In the snippet below, each of the four examples has only a single
    floating-pointing value, and both `y_pred` and `y_true` have the shape
    `[batch_size]`.

    Standalone usage:

    ```python
    bce = keras.losses.BinaryCrossentropy()
    loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.BinaryCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
            compute the loss between the predicted labels and a smoothed version of
            the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, from_logits: bool = ..., label_smoothing=..., reduction=..., name=...):
        self.from_logits = ...
    


class CategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided in a `one_hot` representation. If you want to
    provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature.

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Standalone usage:

    ```python
    cce = keras.losses.CategoricalCrossentropy()
    loss = cce(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.CategoricalCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
            compute the loss between the predicted labels and a smoothed version of
            the true labels, where the smoothing squeezes the labels towards 0.5.
            Larger values of `label_smoothing` correspond to heavier smoothing.
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, from_logits: bool = ..., label_smoothing=..., reduction=..., name=...):
        ...
    


class SparseCategoricalCrossentropy(LossFunctionWrapper):
    """Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided as integers. If you want to provide labels
    using `one-hot` representation, please use `CategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature for `y_pred`
    and a single floating point value per feature for `y_true`.

    In the snippet below, there is a single floating point value per example for
    `y_true` and `# classes` floating pointing values per example for `y_pred`.
    The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
    `[batch_size, num_classes]`.

    Standalone usage:

    ```python
    cce = keras.losses.SparseCategoricalCrossentropy()
    loss = cce(
        [0, 1, 2],
        [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
    ```

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.SparseCategoricalCrossentropy())
    ```

    # Arguments
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default,
            we assume that `y_pred` contains probabilities
            (i.e., values in [0, 1]).
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, from_logits: bool = ..., reduction=..., name=...):
        ...
    


class Hinge(LossFunctionWrapper):
    """Computes the hinge loss between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Hinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class SquaredHinge(LossFunctionWrapper):
    """Computes the squared hinge loss between `y_true` and `y_pred`.

    `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
    provided we will convert them to -1 or 1.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.SquaredHinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class CategoricalHinge(LossFunctionWrapper):
    """Computes the categorical hinge loss between `y_true` and `y_pred`.

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.CategoricalHinge())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class Poisson(LossFunctionWrapper):
    """Computes the Poisson loss between `y_true` and `y_pred`.

    `loss = y_pred - y_true * log(y_pred)`

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Poisson())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class LogCosh(LossFunctionWrapper):
    """Computes the logarithm of the hyperbolic cosine of the prediction error.

    `logcosh = log((exp(x) + exp(-x))/2)`,
    where x is the error (y_pred - y_true)

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.LogCosh())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class KLDivergence(LossFunctionWrapper):
    """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

    `loss = y_true * log(y_true / y_pred)`

    See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.KLDivergence())
    ```

    # Arguments
        reduction: (Optional) Type of loss reduction to apply to loss.
            Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) Name for the object.
    """
    def __init__(self, reduction=..., name=...):
        ...
    


class Huber(LossFunctionWrapper):
    """Computes the Huber loss between `y_true` and `y_pred`.

    Given `x = y_true - y_pred`:
    ```
    loss = 0.5 * x^2                  if |x| <= d
    loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
    ```
    where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

    Usage with the `compile` API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('sgd', loss=keras.losses.Huber())
    ```

    # Arguments
        delta: A float, the point where the Huber loss function changes from a
            quadratic to linear.
        reduction: (Optional) Type of reduction to apply to loss.
        name: Optional name for the object.
    """
    def __init__(self, delta=..., reduction=..., name=...):
        ...
    


def mean_squared_error(y_true, y_pred):
    ...

def mean_absolute_error(y_true, y_pred):
    ...

def mean_absolute_percentage_error(y_true, y_pred):
    ...

def mean_squared_logarithmic_error(y_true, y_pred):
    ...

def squared_hinge(y_true, y_pred):
    ...

def hinge(y_true, y_pred):
    ...

def categorical_hinge(y_true, y_pred):
    ...

def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.

    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    # Returns
        Tensor with one scalar loss entry per sample.
    """
    ...

def huber_loss(y_true, y_pred, delta=...):
    ...

def categorical_crossentropy(y_true, y_pred, from_logits: bool = ..., label_smoothing=...):
    ...

def sparse_categorical_crossentropy(y_true, y_pred, from_logits: bool = ..., axis=...):
    ...

def binary_crossentropy(y_true, y_pred, from_logits: bool = ..., label_smoothing=...):
    ...

def kullback_leibler_divergence(y_true, y_pred):
    ...

def poisson(y_true, y_pred):
    ...

def cosine_proximity(y_true, y_pred, axis=...):
    ...

def _maybe_convert_labels(y_true):
    """Converts binary labels into -1/1."""
    ...

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_similarity = cosine_proximity
def is_categorical_crossentropy(loss):
    ...

def serialize(loss):
    ...

def deserialize(name, custom_objects: Optional[Any] = ...):
    ...

def get(identifier):
    """Get the `identifier` loss function.

    # Arguments
        identifier: None or str, name of the function.

    # Returns
        The loss function or None if `identifier` is None.

    # Raises
        ValueError if unknown identifier.
    """
    ...

