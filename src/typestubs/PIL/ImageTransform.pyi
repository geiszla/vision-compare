"""
This type stub file was generated by pyright.
"""

from . import Image

class Transform(Image.ImageTransformHandler):
    def __init__(self, data):
        self.data = ...
    
    def getdata(self):
        ...
    
    def transform(self, size, image, **options):
        ...
    


class AffineTransform(Transform):
    """
    Define an affine image transform.

    This function takes a 6-tuple (a, b, c, d, e, f) which contain the first
    two rows from an affine transform matrix. For each pixel (x, y) in the
    output image, the new value is taken from a position (a x + b y + c,
    d x + e y + f) in the input image, rounded to nearest pixel.

    This function can be used to scale, translate, rotate, and shear the
    original image.

    See :py:meth:`~PIL.Image.Image.transform`

    :param matrix: A 6-tuple (a, b, c, d, e, f) containing the first two rows
        from an affine transform matrix.
    """
    method = ...


class ExtentTransform(Transform):
    """
    Define a transform to extract a subregion from an image.

    Maps a rectangle (defined by two corners) from the image to a rectangle of
    the given size. The resulting image will contain data sampled from between
    the corners, such that (x0, y0) in the input image will end up at (0,0) in
    the output image, and (x1, y1) at size.

    This method can be used to crop, stretch, shrink, or mirror an arbitrary
    rectangle in the current image. It is slightly slower than crop, but about
    as fast as a corresponding resize operation.

    See :py:meth:`~PIL.Image.Image.transform`

    :param bbox: A 4-tuple (x0, y0, x1, y1) which specifies two points in the
        input image's coordinate system. See :ref:`coordinate-system`.
    """
    method = ...


class QuadTransform(Transform):
    """
    Define a quad image transform.

    Maps a quadrilateral (a region defined by four corners) from the image to a
    rectangle of the given size.

    See :py:meth:`~PIL.Image.Image.transform`

    :param xy: An 8-tuple (x0, y0, x1, y1, x2, y2, x3, y3) which contain the
        upper left, lower left, lower right, and upper right corner of the
        source quadrilateral.
    """
    method = ...


class MeshTransform(Transform):
    """
    Define a mesh image transform.  A mesh transform consists of one or more
    individual quad transforms.

    See :py:meth:`~PIL.Image.Image.transform`

    :param data: A list of (bbox, quad) tuples.
    """
    method = ...

