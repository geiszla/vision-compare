"""
This type stub file was generated by pyright.
"""

from collections.abc import MutableMapping
from numbers import Rational
from . import ImageFile
from typing import Any, Optional

DEBUG = False
READ_LIBTIFF = False
WRITE_LIBTIFF = False
IFD_LEGACY_API = True
II = b"II"
MM = b"MM"
IMAGEWIDTH = 256
IMAGELENGTH = 257
BITSPERSAMPLE = 258
COMPRESSION = 259
PHOTOMETRIC_INTERPRETATION = 262
FILLORDER = 266
IMAGEDESCRIPTION = 270
STRIPOFFSETS = 273
SAMPLESPERPIXEL = 277
ROWSPERSTRIP = 278
STRIPBYTECOUNTS = 279
X_RESOLUTION = 282
Y_RESOLUTION = 283
PLANAR_CONFIGURATION = 284
RESOLUTION_UNIT = 296
TRANSFERFUNCTION = 301
SOFTWARE = 305
DATE_TIME = 306
ARTIST = 315
PREDICTOR = 317
COLORMAP = 320
TILEOFFSETS = 324
EXTRASAMPLES = 338
SAMPLEFORMAT = 339
JPEGTABLES = 347
REFERENCEBLACKWHITE = 532
COPYRIGHT = 33432
IPTC_NAA_CHUNK = 33723
PHOTOSHOP_CHUNK = 34377
ICCPROFILE = 34675
EXIFIFD = 34665
XMP = 700
JPEGQUALITY = 65537
IMAGEJ_META_DATA_BYTE_COUNTS = 50838
IMAGEJ_META_DATA = 50839
COMPRESSION_INFO = { 1: "raw",2: "tiff_ccitt",3: "group3",4: "group4",5: "tiff_lzw",6: "tiff_jpeg",7: "jpeg",8: "tiff_adobe_deflate",32771: "tiff_raw_16",32773: "packbits",32809: "tiff_thunderscan",32946: "tiff_deflate",34676: "tiff_sgilog",34677: "tiff_sgilog24",34925: "lzma",50000: "zstd",50001: "webp" }
COMPRESSION_INFO_REV = { v: k for (k, v) in COMPRESSION_INFO.items() }
OPEN_INFO = { (II, 0, (1, ), 1, (1, ), ()): ("1", "1;I"),(MM, 0, (1, ), 1, (1, ), ()): ("1", "1;I"),(II, 0, (1, ), 2, (1, ), ()): ("1", "1;IR"),(MM, 0, (1, ), 2, (1, ), ()): ("1", "1;IR"),(II, 1, (1, ), 1, (1, ), ()): ("1", "1"),(MM, 1, (1, ), 1, (1, ), ()): ("1", "1"),(II, 1, (1, ), 2, (1, ), ()): ("1", "1;R"),(MM, 1, (1, ), 2, (1, ), ()): ("1", "1;R"),(II, 0, (1, ), 1, (2, ), ()): ("L", "L;2I"),(MM, 0, (1, ), 1, (2, ), ()): ("L", "L;2I"),(II, 0, (1, ), 2, (2, ), ()): ("L", "L;2IR"),(MM, 0, (1, ), 2, (2, ), ()): ("L", "L;2IR"),(II, 1, (1, ), 1, (2, ), ()): ("L", "L;2"),(MM, 1, (1, ), 1, (2, ), ()): ("L", "L;2"),(II, 1, (1, ), 2, (2, ), ()): ("L", "L;2R"),(MM, 1, (1, ), 2, (2, ), ()): ("L", "L;2R"),(II, 0, (1, ), 1, (4, ), ()): ("L", "L;4I"),(MM, 0, (1, ), 1, (4, ), ()): ("L", "L;4I"),(II, 0, (1, ), 2, (4, ), ()): ("L", "L;4IR"),(MM, 0, (1, ), 2, (4, ), ()): ("L", "L;4IR"),(II, 1, (1, ), 1, (4, ), ()): ("L", "L;4"),(MM, 1, (1, ), 1, (4, ), ()): ("L", "L;4"),(II, 1, (1, ), 2, (4, ), ()): ("L", "L;4R"),(MM, 1, (1, ), 2, (4, ), ()): ("L", "L;4R"),(II, 0, (1, ), 1, (8, ), ()): ("L", "L;I"),(MM, 0, (1, ), 1, (8, ), ()): ("L", "L;I"),(II, 0, (1, ), 2, (8, ), ()): ("L", "L;IR"),(MM, 0, (1, ), 2, (8, ), ()): ("L", "L;IR"),(II, 1, (1, ), 1, (8, ), ()): ("L", "L"),(MM, 1, (1, ), 1, (8, ), ()): ("L", "L"),(II, 1, (1, ), 2, (8, ), ()): ("L", "L;R"),(MM, 1, (1, ), 2, (8, ), ()): ("L", "L;R"),(II, 1, (1, ), 1, (12, ), ()): ("I;16", "I;12"),(II, 1, (1, ), 1, (16, ), ()): ("I;16", "I;16"),(MM, 1, (1, ), 1, (16, ), ()): ("I;16B", "I;16B"),(II, 1, (2, ), 1, (16, ), ()): ("I", "I;16S"),(MM, 1, (2, ), 1, (16, ), ()): ("I", "I;16BS"),(II, 0, (3, ), 1, (32, ), ()): ("F", "F;32F"),(MM, 0, (3, ), 1, (32, ), ()): ("F", "F;32BF"),(II, 1, (1, ), 1, (32, ), ()): ("I", "I;32N"),(II, 1, (2, ), 1, (32, ), ()): ("I", "I;32S"),(MM, 1, (2, ), 1, (32, ), ()): ("I", "I;32BS"),(II, 1, (3, ), 1, (32, ), ()): ("F", "F;32F"),(MM, 1, (3, ), 1, (32, ), ()): ("F", "F;32BF"),(II, 1, (1, ), 1, (8, 8), (2, )): ("LA", "LA"),(MM, 1, (1, ), 1, (8, 8), (2, )): ("LA", "LA"),(II, 2, (1, ), 1, (8, 8, 8), ()): ("RGB", "RGB"),(MM, 2, (1, ), 1, (8, 8, 8), ()): ("RGB", "RGB"),(II, 2, (1, ), 2, (8, 8, 8), ()): ("RGB", "RGB;R"),(MM, 2, (1, ), 2, (8, 8, 8), ()): ("RGB", "RGB;R"),(II, 2, (1, ), 1, (8, 8, 8, 8), ()): ("RGBA", "RGBA"),(MM, 2, (1, ), 1, (8, 8, 8, 8), ()): ("RGBA", "RGBA"),(II, 2, (1, ), 1, (8, 8, 8, 8), (0, )): ("RGBX", "RGBX"),(MM, 2, (1, ), 1, (8, 8, 8, 8), (0, )): ("RGBX", "RGBX"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8), (0, 0)): ("RGBX", "RGBXX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8), (0, 0)): ("RGBX", "RGBXX"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ("RGBX", "RGBXXX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (0, 0, 0)): ("RGBX", "RGBXXX"),(II, 2, (1, ), 1, (8, 8, 8, 8), (1, )): ("RGBA", "RGBa"),(MM, 2, (1, ), 1, (8, 8, 8, 8), (1, )): ("RGBA", "RGBa"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8), (1, 0)): ("RGBA", "RGBaX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8), (1, 0)): ("RGBA", "RGBaX"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ("RGBA", "RGBaXX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (1, 0, 0)): ("RGBA", "RGBaXX"),(II, 2, (1, ), 1, (8, 8, 8, 8), (2, )): ("RGBA", "RGBA"),(MM, 2, (1, ), 1, (8, 8, 8, 8), (2, )): ("RGBA", "RGBA"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8), (2, 0)): ("RGBA", "RGBAX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8), (2, 0)): ("RGBA", "RGBAX"),(II, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ("RGBA", "RGBAXX"),(MM, 2, (1, ), 1, (8, 8, 8, 8, 8, 8), (2, 0, 0)): ("RGBA", "RGBAXX"),(II, 2, (1, ), 1, (8, 8, 8, 8), (999, )): ("RGBA", "RGBA"),(MM, 2, (1, ), 1, (8, 8, 8, 8), (999, )): ("RGBA", "RGBA"),(II, 2, (1, ), 1, (16, 16, 16), ()): ("RGB", "RGB;16L"),(MM, 2, (1, ), 1, (16, 16, 16), ()): ("RGB", "RGB;16B"),(II, 2, (1, ), 1, (16, 16, 16, 16), ()): ("RGBA", "RGBA;16L"),(MM, 2, (1, ), 1, (16, 16, 16, 16), ()): ("RGBA", "RGBA;16B"),(II, 2, (1, ), 1, (16, 16, 16, 16), (0, )): ("RGBX", "RGBX;16L"),(MM, 2, (1, ), 1, (16, 16, 16, 16), (0, )): ("RGBX", "RGBX;16B"),(II, 2, (1, ), 1, (16, 16, 16, 16), (1, )): ("RGBA", "RGBa;16L"),(MM, 2, (1, ), 1, (16, 16, 16, 16), (1, )): ("RGBA", "RGBa;16B"),(II, 2, (1, ), 1, (16, 16, 16, 16), (2, )): ("RGBA", "RGBA;16L"),(MM, 2, (1, ), 1, (16, 16, 16, 16), (2, )): ("RGBA", "RGBA;16B"),(II, 3, (1, ), 1, (1, ), ()): ("P", "P;1"),(MM, 3, (1, ), 1, (1, ), ()): ("P", "P;1"),(II, 3, (1, ), 2, (1, ), ()): ("P", "P;1R"),(MM, 3, (1, ), 2, (1, ), ()): ("P", "P;1R"),(II, 3, (1, ), 1, (2, ), ()): ("P", "P;2"),(MM, 3, (1, ), 1, (2, ), ()): ("P", "P;2"),(II, 3, (1, ), 2, (2, ), ()): ("P", "P;2R"),(MM, 3, (1, ), 2, (2, ), ()): ("P", "P;2R"),(II, 3, (1, ), 1, (4, ), ()): ("P", "P;4"),(MM, 3, (1, ), 1, (4, ), ()): ("P", "P;4"),(II, 3, (1, ), 2, (4, ), ()): ("P", "P;4R"),(MM, 3, (1, ), 2, (4, ), ()): ("P", "P;4R"),(II, 3, (1, ), 1, (8, ), ()): ("P", "P"),(MM, 3, (1, ), 1, (8, ), ()): ("P", "P"),(II, 3, (1, ), 1, (8, 8), (2, )): ("PA", "PA"),(MM, 3, (1, ), 1, (8, 8), (2, )): ("PA", "PA"),(II, 3, (1, ), 2, (8, ), ()): ("P", "P;R"),(MM, 3, (1, ), 2, (8, ), ()): ("P", "P;R"),(II, 5, (1, ), 1, (8, 8, 8, 8), ()): ("CMYK", "CMYK"),(MM, 5, (1, ), 1, (8, 8, 8, 8), ()): ("CMYK", "CMYK"),(II, 5, (1, ), 1, (8, 8, 8, 8, 8), (0, )): ("CMYK", "CMYKX"),(MM, 5, (1, ), 1, (8, 8, 8, 8, 8), (0, )): ("CMYK", "CMYKX"),(II, 5, (1, ), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ("CMYK", "CMYKXX"),(MM, 5, (1, ), 1, (8, 8, 8, 8, 8, 8), (0, 0)): ("CMYK", "CMYKXX"),(II, 5, (1, ), 1, (16, 16, 16, 16), ()): ("CMYK", "CMYK;16L"),(II, 6, (1, ), 1, (8, 8, 8), ()): ("RGB", "RGBX"),(MM, 6, (1, ), 1, (8, 8, 8), ()): ("RGB", "RGBX"),(II, 8, (1, ), 1, (8, 8, 8), ()): ("LAB", "LAB"),(MM, 8, (1, ), 1, (8, 8, 8), ()): ("LAB", "LAB") }
PREFIXES = [b"MM\x00\x2A", b"II\x2A\x00", b"MM\x2A\x00", b"II\x00\x2A"]
def _accept(prefix):
    ...

def _limit_rational(val, max_val):
    ...

def _limit_signed_rational(val, max_val, min_val):
    ...

_load_dispatch = {  }
_write_dispatch = {  }
class IFDRational(Rational):
    """ Implements a rational class where 0/0 is a legal value to match
    the in the wild use of exif rationals.

    e.g., DigitalZoomRatio - 0.00/0.00  indicates that no digital zoom was used
    """
    __slots__ = ...
    def __init__(self, value, denominator=...):
        """
        :param value: either an integer numerator, a
        float/rational/other number, or an IFDRational
        :param denominator: Optional integer denominator
        """
        ...
    
    @property
    def numerator(a):
        ...
    
    @property
    def denominator(a):
        ...
    
    def limit_rational(self, max_denominator):
        """

        :param max_denominator: Integer, the maximum denominator value
        :returns: Tuple of (numerator, denominator)
        """
        ...
    
    def __repr__(self):
        ...
    
    def __hash__(self):
        ...
    
    def __eq__(self, other):
        ...
    
    def _delegate(op):
        ...
    
    __add__ = ...
    __radd__ = ...
    __sub__ = ...
    __rsub__ = ...
    __mul__ = ...
    __rmul__ = ...
    __truediv__ = ...
    __rtruediv__ = ...
    __floordiv__ = ...
    __rfloordiv__ = ...
    __mod__ = ...
    __rmod__ = ...
    __pow__ = ...
    __rpow__ = ...
    __pos__ = ...
    __neg__ = ...
    __abs__ = ...
    __trunc__ = ...
    __lt__ = ...
    __gt__ = ...
    __le__ = ...
    __ge__ = ...
    __bool__ = ...
    __ceil__ = ...
    __floor__ = ...
    __round__ = ...


class ImageFileDirectory_v2(MutableMapping):
    """This class represents a TIFF tag directory.  To speed things up, we
    don't decode tags unless they're asked for.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v2()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        'Some Data'

    Individual values are returned as the strings or numbers, sequences are
    returned as tuples of the values.

    The tiff metadata type of each item is stored in a dictionary of
    tag types in
    `~PIL.TiffImagePlugin.ImageFileDirectory_v2.tagtype`. The types
    are read from a tiff file, guessed from the type added, or added
    manually.

    Data Structures:

        * self.tagtype = {}

          * Key: numerical tiff tag number
          * Value: integer corresponding to the data type from
                   ~PIL.TiffTags.TYPES`

    .. versionadded:: 3.0.0
    """
    def __init__(self, ifh=..., prefix: Optional[Any] = ...):
        """Initialize an ImageFileDirectory.

        To construct an ImageFileDirectory from a real file, pass the 8-byte
        magic header to the constructor.  To only set the endianness, pass it
        as the 'prefix' keyword argument.

        :param ifh: One of the accepted magic headers (cf. PREFIXES); also sets
              endianness.
        :param prefix: Override the endianness of the file.
        """
        ...
    
    prefix = ...
    offset = ...
    legacy_api = ...
    @legacy_api.setter
    def legacy_api(self, value):
        ...
    
    def reset(self):
        self.tagtype = ...
    
    def __str__(self):
        ...
    
    def named(self):
        """
        :returns: dict of name|key: value

        Returns the complete tag dictionary, with named tags where possible.
        """
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, tag):
        ...
    
    def __contains__(self, tag):
        ...
    
    def __setitem__(self, tag, value):
        ...
    
    def _setitem(self, tag, value, legacy_api):
        ...
    
    def __delitem__(self, tag):
        ...
    
    def __iter__(self):
        ...
    
    def _unpack(self, fmt, data):
        ...
    
    def _pack(self, fmt, *values):
        ...
    
    def _register_loader(idx, size):
        ...
    
    def _register_writer(idx):
        ...
    
    def _register_basic(idx_fmt_name):
        ...
    
    @_register_loader(1, 1)
    def load_byte(self, data, legacy_api: bool = ...):
        ...
    
    @_register_writer(1)
    def write_byte(self, data):
        ...
    
    @_register_loader(2, 1)
    def load_string(self, data, legacy_api: bool = ...):
        ...
    
    @_register_writer(2)
    def write_string(self, value):
        ...
    
    @_register_loader(5, 8)
    def load_rational(self, data, legacy_api: bool = ...):
        ...
    
    @_register_writer(5)
    def write_rational(self, *values):
        ...
    
    @_register_loader(7, 1)
    def load_undefined(self, data, legacy_api: bool = ...):
        ...
    
    @_register_writer(7)
    def write_undefined(self, value):
        ...
    
    @_register_loader(10, 8)
    def load_signed_rational(self, data, legacy_api: bool = ...):
        ...
    
    @_register_writer(10)
    def write_signed_rational(self, *values):
        ...
    
    def _ensure_read(self, fp, size):
        ...
    
    def load(self, fp):
        ...
    
    def tobytes(self, offset=...):
        ...
    
    def save(self, fp):
        ...
    


class ImageFileDirectory_v1(ImageFileDirectory_v2):
    """This class represents the **legacy** interface to a TIFF tag directory.

    Exposes a dictionary interface of the tags in the directory::

        ifd = ImageFileDirectory_v1()
        ifd[key] = 'Some Data'
        ifd.tagtype[key] = TiffTags.ASCII
        print(ifd[key])
        ('Some Data',)

    Also contains a dictionary of tag types as read from the tiff image file,
    `~PIL.TiffImagePlugin.ImageFileDirectory_v1.tagtype`.

    Values are returned as a tuple.

    ..  deprecated:: 3.0.0
    """
    def __init__(self, *args, **kwargs):
        ...
    
    tags = ...
    tagdata = ...
    @classmethod
    def from_v2(cls, original):
        """ Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`

        """
        ...
    
    def to_v2(self):
        """ Returns an
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`
        instance with the same data as is contained in the original
        :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v1`
        instance.

        :returns: :py:class:`~PIL.TiffImagePlugin.ImageFileDirectory_v2`

        """
        ...
    
    def __contains__(self, tag):
        ...
    
    def __len__(self):
        ...
    
    def __iter__(self):
        ...
    
    def __setitem__(self, tag, value):
        ...
    
    def __getitem__(self, tag):
        ...
    


ImageFileDirectory = ImageFileDirectory_v1
class TiffImageFile(ImageFile.ImageFile):
    format = ...
    format_description = ...
    _close_exclusive_fp_after_loading = ...
    def _open(self):
        """Open the first image in a TIFF file"""
        self.tag_v2 = ...
        self.tag = ...
    
    @property
    def n_frames(self):
        ...
    
    @property
    def is_animated(self):
        ...
    
    def seek(self, frame):
        """Select a given frame as current image"""
        self.im = ...
    
    def _seek(self, frame):
        self.fp = ...
        self.tag = ...
    
    def tell(self):
        """Return the current frame number"""
        ...
    
    def load(self):
        ...
    
    def load_end(self):
        ...
    
    def _load_libtiff(self):
        """ Overload method triggered when we detect a compressed tiff
            Calls out to libtiff """
        self.tile = ...
        self.readonly = ...
    
    def _setup(self):
        """Setup this image object based on current tags"""
        self.tile = ...
        self.use_load_libtiff = ...
    
    def _close__fp(self):
        ...
    


SAVE_INFO = { "1": ("1", II, 1, 1, (1, ), None),"L": ("L", II, 1, 1, (8, ), None),"LA": ("LA", II, 1, 1, (8, 8), 2),"P": ("P", II, 3, 1, (8, ), None),"PA": ("PA", II, 3, 1, (8, 8), 2),"I": ("I;32S", II, 1, 2, (32, ), None),"I;16": ("I;16", II, 1, 1, (16, ), None),"I;16S": ("I;16S", II, 1, 2, (16, ), None),"F": ("F;32F", II, 1, 3, (32, ), None),"RGB": ("RGB", II, 2, 1, (8, 8, 8), None),"RGBX": ("RGBX", II, 2, 1, (8, 8, 8, 8), 0),"RGBA": ("RGBA", II, 2, 1, (8, 8, 8, 8), 2),"CMYK": ("CMYK", II, 5, 1, (8, 8, 8, 8), None),"YCbCr": ("YCbCr", II, 6, 1, (8, 8, 8), None),"LAB": ("LAB", II, 8, 1, (8, 8, 8), None),"I;32BS": ("I;32BS", MM, 1, 2, (32, ), None),"I;16B": ("I;16B", MM, 1, 1, (16, ), None),"I;16BS": ("I;16BS", MM, 1, 2, (16, ), None),"F;32BF": ("F;32BF", MM, 1, 3, (32, ), None) }
def _save(im, fp, filename):
    ...

class AppendingTiffWriter:
    fieldSizes = ...
    Tags = ...
    def __init__(self, fn, new: bool = ...):
        self.beginning = ...
    
    def setup(self):
        self.whereToWriteNewIFDOffset = ...
        self.offsetOfNewPage = ...
        self.IIMM = ...
        self.isFirst = ...
    
    def finalize(self):
        ...
    
    def newFrame(self):
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, exc_type, exc_value, traceback):
        ...
    
    def tell(self):
        ...
    
    def seek(self, offset, whence=...):
        ...
    
    def goToEnd(self):
        self.offsetOfNewPage = ...
    
    def setEndian(self, endian):
        self.endian = ...
        self.longFmt = ...
        self.shortFmt = ...
        self.tagFormat = ...
    
    def skipIFDs(self):
        ...
    
    def write(self, data):
        ...
    
    def readShort(self):
        ...
    
    def readLong(self):
        ...
    
    def rewriteLastShortToLong(self, value):
        ...
    
    def rewriteLastShort(self, value):
        ...
    
    def rewriteLastLong(self, value):
        ...
    
    def writeShort(self, value):
        ...
    
    def writeLong(self, value):
        ...
    
    def close(self):
        ...
    
    def fixIFD(self):
        ...
    
    def fixOffsets(self, count, isShort: bool = ..., isLong: bool = ...):
        ...
    


def _save_all(im, fp, filename):
    ...

