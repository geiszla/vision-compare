"""
This type stub file was generated by pyright.
"""

import sys
import functools
import contextlib
from .numerictypes import bool_, complex_, float_, int_, intc, longlong
from .overrides import array_function_dispatch, set_module
from typing import Any, Optional

"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $

"""
__all__ = ["array2string", "array_str", "array_repr", "set_string_function", "set_printoptions", "get_printoptions", "printoptions", "format_float_positional", "format_float_scientific"]
__docformat__ = 'restructuredtext'
if sys.version_info[0] >= 3:
    ...
else:
    ...
_format_options = { 'edgeitems': 3,'threshold': 1000,'floatmode': 'maxprec','precision': 8,'suppress': False,'linewidth': 75,'nanstr': 'nan','infstr': 'inf','sign': '-','formatter': None,'legacy': False }
def _make_options_dict(precision: Optional[Any] = ..., threshold: Optional[Any] = ..., edgeitems: Optional[Any] = ..., linewidth: Optional[Any] = ..., suppress: Optional[Any] = ..., nanstr: Optional[Any] = ..., infstr: Optional[Any] = ..., sign: Optional[Any] = ..., formatter: Optional[Any] = ..., floatmode: Optional[Any] = ..., legacy: Optional[Any] = ...):
    """ make a dictionary out of the non-None arguments, plus sanity checks """
    ...

@set_module('numpy')
def set_printoptions(precision: Optional[Any] = ..., threshold: Optional[Any] = ..., edgeitems: Optional[Any] = ..., linewidth: Optional[Any] = ..., suppress: Optional[Any] = ..., nanstr: Optional[Any] = ..., infstr: Optional[Any] = ..., formatter: Optional[Any] = ..., sign: Optional[Any] = ..., floatmode: Optional[Any] = ..., **kwarg):
    """
    Set printing options.

    These options determine the way floating point numbers, arrays and
    other NumPy objects are displayed.

    Parameters
    ----------
    precision : int or None, optional
        Number of digits of precision for floating point output (default 8).
        May be None if `floatmode` is not `fixed`, to print as many digits as
        necessary to uniquely specify the value.
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).
        To always use the full repr without summarization, pass `sys.maxsize`.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension (default 3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting
        line breaks (default 75).
    suppress : bool, optional
        If True, always print floating point numbers using fixed point
        notation, in which case numbers equal to zero in the current precision
        will print as zero.  If False, then scientific notation is used when
        absolute value of the smallest number is < 1e-4 or the ratio of the
        maximum absolute value to the minimum is > 1e3. The default is False.
    nanstr : str, optional
        String representation of floating point not-a-number (default nan).
    infstr : str, optional
        String representation of floating point infinity (default inf).
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values. (default '-')
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'numpystr' : types `numpy.string_` and `numpy.unicode_`
        - 'object' : `np.object_` arrays
        - 'str' : all other strings

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'str' and 'numpystr'
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types. Can take the following values
        (default maxprec_equal):

        * 'fixed': Always print exactly `precision` fractional digits,
                even if this would print more or fewer digits than
                necessary to specify the value uniquely.
        * 'unique': Print the minimum number of fractional digits necessary
                to represent each value uniquely. Different elements may
                have a different number of digits. The value of the
                `precision` option is ignored.
        * 'maxprec': Print at most `precision` fractional digits, but if
                an element can be uniquely represented with fewer digits
                only print it with that many.
        * 'maxprec_equal': Print at most `precision` fractional digits,
                but if every element in the array can be uniquely
                represented with an equal number of fewer digits, use that
                many digits for all elements.
    legacy : string or `False`, optional
        If set to the string `'1.13'` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. If set to
        `False`, disables legacy mode. Unrecognized strings will be ignored
        with a warning for forward compatibility.

        .. versionadded:: 1.14.0

    See Also
    --------
    get_printoptions, printoptions, set_string_function, array2string

    Notes
    -----
    `formatter` is always reset with a call to `set_printoptions`.

    Use `printoptions` as a context manager to set the values temporarily.

    Examples
    --------
    Floating point precision can be set:

    >>> np.set_printoptions(precision=4)
    >>> np.array([1.123456789])
    [1.1235]

    Long arrays can be summarised:

    >>> np.set_printoptions(threshold=5)
    >>> np.arange(10)
    array([0, 1, 2, ..., 7, 8, 9])

    Small results can be suppressed:

    >>> eps = np.finfo(float).eps
    >>> x = np.arange(4.)
    >>> x**2 - (x + eps)**2
    array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])
    >>> np.set_printoptions(suppress=True)
    >>> x**2 - (x + eps)**2
    array([-0., -0.,  0.,  0.])

    A custom formatter can be used to display array elements as desired:

    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
    >>> x = np.arange(3)
    >>> x
    array([int: 0, int: -1, int: -2])
    >>> np.set_printoptions()  # formatter gets reset
    >>> x
    array([0, 1, 2])

    To put back the default options, you can use:

    >>> np.set_printoptions(edgeitems=3, infstr='inf',
    ... linewidth=75, nanstr='nan', precision=8,
    ... suppress=False, threshold=1000, formatter=None)

    Also to temporarily override options, use `printoptions` as a context manager:

    >>> with np.printoptions(precision=2, suppress=True, threshold=5):
    ...     np.linspace(0, 10, 10)
    array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ])

    """
    ...

@set_module('numpy')
def get_printoptions():
    """
    Return the current print options.

    Returns
    -------
    print_opts : dict
        Dictionary of current print options with keys

          - precision : int
          - threshold : int
          - edgeitems : int
          - linewidth : int
          - suppress : bool
          - nanstr : str
          - infstr : str
          - formatter : dict of callables
          - sign : str

        For a full description of these options, see `set_printoptions`.

    See Also
    --------
    set_printoptions, printoptions, set_string_function

    """
    ...

@set_module('numpy')
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for setting print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `set_printoptions` for the full description of
    available options.

    Examples
    --------

    >>> from numpy.testing import assert_equal
    >>> with np.printoptions(precision=2):
    ...     np.array([2.0]) / 3
    array([0.67])

    The `as`-clause of the `with`-statement gives the current print options:

    >>> with np.printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    See Also
    --------
    set_printoptions, get_printoptions

    """
    ...

def _leading_trailing(a, edgeitems, index=...):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.

    Should be passed a base-class ndarray, since it makes no guarantees about
    preserving subclasses.
    """
    ...

def _object_format(o):
    """ Object arrays containing lists should be printed unambiguously """
    ...

def repr_format(x):
    ...

def str_format(x):
    ...

def _get_formatdict(data, **opt):
    ...

def _get_format_function(data, **options):
    """
    find the right formatting function for the dtype_
    """
    ...

def _recursive_guard(fillvalue=...):
    """
    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

    Decorates a function such that if it calls itself with the same first
    argument, it returns `fillvalue` instead of recursing.

    Largely copied from reprlib.recursive_repr
    """
    ...

@_recursive_guard()
def _array2string(a, options, separator=..., prefix=...):
    ...

def _array2string_dispatcher(a, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ..., separator: Optional[Any] = ..., prefix: Optional[Any] = ..., style: Optional[Any] = ..., formatter: Optional[Any] = ..., threshold: Optional[Any] = ..., edgeitems: Optional[Any] = ..., sign: Optional[Any] = ..., floatmode: Optional[Any] = ..., suffix: Optional[Any] = ..., **kwarg):
    ...

@array_function_dispatch(_array2string_dispatcher, module='numpy')
def array2string(a, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ..., separator=..., prefix=..., style=..., formatter: Optional[Any] = ..., threshold: Optional[Any] = ..., edgeitems: Optional[Any] = ..., sign: Optional[Any] = ..., floatmode: Optional[Any] = ..., suffix=..., **kwarg):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int or None, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.
    separator : str, optional
        Inserted between elements.
    prefix : str, optional
    suffix: str, optional
        The length of the prefix and suffix strings are used to respectively
        align and wrap the output. An array is typically printed as::

          prefix + array2string(a) + suffix

        The output is left-padded by the length of the prefix string, and
        wrapping is forced at the column ``max_line_width - len(suffix)``.
        It should be noted that the content of prefix and suffix strings are
        not included in the output.
    style : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.14.0
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'void' : type `numpy.void`
        - 'numpystr' : types `numpy.string_` and `numpy.unicode_`
        - 'str' : all other strings

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'str' and 'numpystr'
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr.
        Defaults to ``numpy.get_printoptions()['threshold']``.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension.
        Defaults to ``numpy.get_printoptions()['edgeitems']``.
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values.
        Defaults to ``numpy.get_printoptions()['sign']``.
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types.
        Defaults to ``numpy.get_printoptions()['floatmode']``.
        Can take the following values:

        - 'fixed': Always print exactly `precision` fractional digits,
          even if this would print more or fewer digits than
          necessary to specify the value uniquely.
        - 'unique': Print the minimum number of fractional digits necessary
          to represent each value uniquely. Different elements may
          have a different number of digits.  The value of the
          `precision` option is ignored.
        - 'maxprec': Print at most `precision` fractional digits, but if
          an element can be uniquely represented with fewer digits
          only print it with that many.
        - 'maxprec_equal': Print at most `precision` fractional digits,
          but if every element in the array can be uniquely
          represented with an equal number of fewer digits, use that
          many digits for all elements.
    legacy : string or `False`, optional
        If set to the string `'1.13'` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. If set to
        `False`, disables legacy mode. Unrecognized strings will be ignored
        with a warning for forward compatibility.

        .. versionadded:: 1.14.0

    Returns
    -------
    array_str : str
        String representation of the array.

    Raises
    ------
    TypeError
        if a callable in `formatter` does not return a string.

    See Also
    --------
    array_str, array_repr, set_printoptions, get_printoptions

    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    This is a very flexible function; `array_repr` and `array_str` are using
    `array2string` internally so keywords with the same name should work
    identically in all three functions.

    Examples
    --------
    >>> x = np.array([1e-16,1,2,3])
    >>> np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True)
    '[0.,1.,2.,3.]'

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0 0x1 0x2]'

    """
    ...

def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    ...

def _formatArray(a, format_function, line_width, next_line_prefix, separator, edge_items, summary_insert, legacy):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    ...

def _none_or_positive_arg(x, name):
    ...

class FloatingFormat(object):
    """ Formatter for subtypes of np.floating """
    def __init__(self, data, precision, floatmode, suppress_small, sign: bool = ..., **kwarg):
        self.floatmode = ...
        self.precision = ...
        self.suppress_small = ...
        self.sign = ...
        self.exp_format = ...
        self.large_exponent = ...
    
    def fillFormat(self, data):
        ...
    
    def __call__(self, x):
        ...
    


@set_module('numpy')
def format_float_scientific(x, precision: Optional[Any] = ..., unique: bool = ..., trim=..., sign: bool = ..., pad_left: Optional[Any] = ..., exp_digits: Optional[Any] = ...):
    """
    Format a floating-point scalar as a decimal string in scientific notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        was omitted, print all necessary digits, otherwise digit generation is
        cut off after `precision` digits and the remaining value is rounded.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value.
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    exp_digits : non-negative integer, optional
        Pad the exponent with zeros until it contains at least this many digits.
        If omitted, the exponent will be at least 2 digits.

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_positional

    Examples
    --------
    >>> np.format_float_scientific(np.float32(np.pi))
    '3.1415927e+00'
    >>> s = np.float32(1.23e24)
    >>> np.format_float_scientific(s, unique=False, precision=15)
    '1.230000071797338e+24'
    >>> np.format_float_scientific(s, exp_digits=4)
    '1.23e+0024'
    """
    ...

@set_module('numpy')
def format_float_positional(x, precision: Optional[Any] = ..., unique: bool = ..., fractional: bool = ..., trim=..., sign: bool = ..., pad_left: Optional[Any] = ..., pad_right: Optional[Any] = ...):
    """
    Format a floating-point scalar as a decimal string in positional notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        was omitted, print out all necessary digits, otherwise digit generation
        is cut off after `precision` digits and the remaining value is rounded.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value.
    fractional : boolean, optional
        If `True`, the cutoff of `precision` digits refers to the total number
        of digits after the decimal point, including leading zeros.
        If `False`, `precision` refers to the total number of significant
        digits, before or after the decimal point, ignoring leading zeros.
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    pad_right : non-negative integer, optional
        Pad the right side of the string with whitespace until at least that
        many characters are to the right of the decimal point.

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_scientific

    Examples
    --------
    >>> np.format_float_positional(np.float32(np.pi))
    '3.1415927'
    >>> np.format_float_positional(np.float16(np.pi))
    '3.14'
    >>> np.format_float_positional(np.float16(0.3))
    '0.3'
    >>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
    '0.3000488281'
    """
    ...

class IntegerFormat(object):
    def __init__(self, data):
        self.format = ...
    
    def __call__(self, x):
        ...
    


class BoolFormat(object):
    def __init__(self, data, **kwargs):
        self.truestr = ...
    
    def __call__(self, x):
        ...
    


class ComplexFloatingFormat(object):
    """ Formatter for subtypes of np.complexfloating """
    def __init__(self, x, precision, floatmode, suppress_small, sign: bool = ..., **kwarg):
        self.real_format = ...
        self.imag_format = ...
    
    def __call__(self, x):
        ...
    


class _TimelikeFormat(object):
    def __init__(self, data):
        ...
    
    def _format_non_nat(self, x):
        ...
    
    def __call__(self, x):
        ...
    


class DatetimeFormat(_TimelikeFormat):
    def __init__(self, x, unit: Optional[Any] = ..., timezone: Optional[Any] = ..., casting=..., legacy: bool = ...):
        self.timezone = ...
        self.unit = ...
        self.casting = ...
        self.legacy = ...
    
    def __call__(self, x):
        ...
    
    def _format_non_nat(self, x):
        ...
    


class TimedeltaFormat(_TimelikeFormat):
    def _format_non_nat(self, x):
        ...
    


class SubArrayFormat(object):
    def __init__(self, format_function):
        self.format_function = ...
    
    def __call__(self, arr):
        ...
    


class StructuredVoidFormat(object):
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like np.dtype(('i4', 'i2,i2')),
    as alias scalars lose their field information, and the implementation
    relies upon np.void.__getitem__.
    """
    def __init__(self, format_functions):
        self.format_functions = ...
    
    @classmethod
    def from_data(cls, data, **options):
        """
        This is a second way to initialize StructuredVoidFormat, using the raw data
        as input. Added to avoid changing the signature of __init__.
        """
        ...
    
    def __call__(self, x):
        ...
    


def _void_scalar_repr(x):
    """
    Implements the repr for structured-void scalars. It is called from the
    scalartypes.c.src code, and is placed here because it uses the elementwise
    formatters defined above.
    """
    ...

_typelessdata = [int_, float_, complex_, bool_]
if issubclass(intc, int):
    ...
if issubclass(longlong, int):
    ...
def dtype_is_implied(dtype):
    """
    Determine if the given dtype is implied by the representation of its values.

    Parameters
    ----------
    dtype : dtype
        Data type

    Returns
    -------
    implied : bool
        True if the dtype is implied by the representation of its values.

    Examples
    --------
    >>> np.core.arrayprint.dtype_is_implied(int)
    True
    >>> np.array([1, 2, 3], int)
    array([1, 2, 3])
    >>> np.core.arrayprint.dtype_is_implied(np.int8)
    False
    >>> np.array([1, 2, 3], np.int8)
    array([1, 2, 3], dtype=int8)
    """
    ...

def dtype_short_repr(dtype):
    """
    Convert a dtype to a short form which evaluates to the same dtype.

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> dt = np.int64([1, 2]).dtype
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    ...

def _array_repr_implementation(arr, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ..., array2string=...):
    """Internal version of array_repr() that allows overriding array2string."""
    ...

def _array_repr_dispatcher(arr, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ...):
    ...

@array_function_dispatch(_array_repr_dispatcher, module='numpy')
def array_repr(arr, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ...):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns
    -------
    string : str
      The string representation of an array.

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([0.000001,  0.      ,  2.      ,  3.      ])'

    """
    ...

@_recursive_guard()
def _guarded_repr_or_str(v):
    ...

def _array_str_implementation(a, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ..., array2string=...):
    """Internal version of array_str() that allows overriding array2string."""
    ...

def _array_str_dispatcher(a, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ...):
    ...

@array_function_dispatch(_array_str_dispatcher, module='numpy')
def array_str(a, max_line_width: Optional[Any] = ..., precision: Optional[Any] = ..., suppress_small: Optional[Any] = ...):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    See Also
    --------
    array2string, array_repr, set_printoptions

    Examples
    --------
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

    """
    ...

_array2string_impl = getattr(array2string, '__wrapped__', array2string)
_default_array_str = functools.partial(_array_str_implementation, array2string=_array2string_impl)
_default_array_repr = functools.partial(_array_repr_implementation, array2string=_array2string_impl)
def set_string_function(f, repr: bool = ...):
    """
    Set a Python function to be used when pretty printing arrays.

    Parameters
    ----------
    f : function or None
        Function to be used to pretty print arrays. The function should expect
        a single array argument and return a string of the representation of
        the array. If None, the function is reset to the default NumPy function
        to print arrays.
    repr : bool, optional
        If True (default), the function for pretty printing (``__repr__``)
        is set, if False the function that returns the default string
        representation (``__str__``) is set.

    See Also
    --------
    set_printoptions, get_printoptions

    Examples
    --------
    >>> def pprint(arr):
    ...     return 'HA! - What are you going to do now?'
    ...
    >>> np.set_string_function(pprint)
    >>> a = np.arange(10)
    >>> a
    HA! - What are you going to do now?
    >>> _ = a
    >>> # [0 1 2 3 4 5 6 7 8 9]

    We can reset the function to the default:

    >>> np.set_string_function(None)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    `repr` affects either pretty printing or normal string representation.
    Note that ``__repr__`` is still affected by setting ``__str__``
    because the width of each array element in the returned string becomes
    equal to the length of the result of ``__str__()``.

    >>> x = np.arange(4)
    >>> np.set_string_function(lambda x:'random', repr=False)
    >>> x.__str__()
    'random'
    >>> x.__repr__()
    'array([0, 1, 2, 3])'

    """
    ...

