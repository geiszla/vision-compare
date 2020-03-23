"""
This type stub file was generated by pyright.
"""

from . import backend as K
from typing import Any, Optional

"""Built-in weight initializers.
"""
class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    
    @classmethod
    def from_config(cls, config):
        ...
    


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    # Arguments
        value: float; the value of the generator tensors.
    """
    def __init__(self, value=...):
        self.value = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """
    def __init__(self, mean=..., stddev=..., seed: Optional[Any] = ...):
        self.mean = ...
        self.stddev = ...
        self.seed = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    # Arguments
        minval: A python scalar or a scalar tensor. Lower bound of the range
          of random values to generate.
        maxval: A python scalar or a scalar tensor. Upper bound of the range
          of random values to generate.  Defaults to 1 for float types.
        seed: A Python integer. Used to seed the random generator.
    """
    def __init__(self, minval=..., maxval=..., seed: Optional[Any] = ...):
        self.minval = ...
        self.maxval = ...
        self.seed = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and redrawn. This is the recommended initializer for
    neural network weights and filters.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """
    def __init__(self, mean=..., stddev=..., seed: Optional[Any] = ...):
        self.mean = ...
        self.stddev = ...
        self.seed = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class VarianceScaling(Initializer):
    """Initializer capable of adapting its scale to the shape of weights.

    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

    # Arguments
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "normal", "uniform".
        seed: A Python integer. Used to seed the random generator.

    # Raises
        ValueError: In case of an invalid value for the "scale", mode" or
          "distribution" arguments.
    """
    def __init__(self, scale=..., mode=..., distribution=..., seed: Optional[Any] = ...):
        self.scale = ...
        self.mode = ...
        self.distribution = ...
        self.seed = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class Orthogonal(Initializer):
    """Initializer that generates a random orthogonal matrix.

    # Arguments
        gain: Multiplicative factor to apply to the orthogonal matrix.
        seed: A Python integer. Used to seed the random generator.

    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
    """
    def __init__(self, gain=..., seed: Optional[Any] = ...):
        self.gain = ...
        self.seed = ...
    
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


class Identity(Initializer):
    """Initializer that generates the identity matrix.

    Only use for 2D matrices.
    If the desired matrix is not square, it gets padded
    with zeros for the additional rows/columns.

    # Arguments
        gain: Multiplicative factor to apply to the identity matrix.
    """
    def __init__(self, gain=...):
        self.gain = ...
    
    @K.eager
    def __call__(self, shape, dtype: Optional[Any] = ...):
        ...
    
    def get_config(self):
        ...
    


def lecun_uniform(seed: Optional[Any] = ...):
    """LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    ...

def glorot_normal(seed: Optional[Any] = ...):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    ...

def glorot_uniform(seed: Optional[Any] = ...):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    ...

def he_normal(seed: Optional[Any] = ...):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    ...

def lecun_normal(seed: Optional[Any] = ...):
    """LeCun normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    ...

def he_uniform(seed: Optional[Any] = ...):
    """He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    ...

zero = zeros = Zeros
one = ones = Ones
constant = Constant
uniform = random_uniform = RandomUniform
normal = random_normal = RandomNormal
truncated_normal = TruncatedNormal
identity = Identity
orthogonal = Orthogonal
def _compute_fans(shape, data_format=...):
    """Computes the number of input and output units for a weight shape.

    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.

    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    ...

def serialize(initializer):
    ...

def deserialize(config, custom_objects: Optional[Any] = ...):
    ...

def get(identifier):
    ...
