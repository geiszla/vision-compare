"""
This type stub file was generated by pyright.
"""

from PIL import Image
from typing import Any, Optional

DESCRIPTION = """
pyCMS

    a Python / PIL interface to the littleCMS ICC Color Management System
    Copyright (C) 2002-2003 Kevin Cazabon
    kevin@cazabon.com
    http://www.cazabon.com

    pyCMS home page:  http://www.cazabon.com/pyCMS
    littleCMS home page:  http://www.littlecms.com
    (littleCMS is Copyright (C) 1998-2001 Marti Maria)

    Originally released under LGPL.  Graciously donated to PIL in
    March 2009, for distribution under the standard PIL license

    The pyCMS.py module provides a "clean" interface between Python/PIL and
    pyCMSdll, taking care of some of the more complex handling of the direct
    pyCMSdll functions, as well as error-checking and making sure that all
    relevant data is kept together.

    While it is possible to call pyCMSdll functions directly, it's not highly
    recommended.

    Version History:

        1.0.0 pil       Oct 2013 Port to LCMS 2.

        0.1.0 pil mod   March 10, 2009

                        Renamed display profile to proof profile. The proof
                        profile is the profile of the device that is being
                        simulated, not the profile of the device which is
                        actually used to display/print the final simulation
                        (that'd be the output profile) - also see LCMSAPI.txt
                        input colorspace -> using 'renderingIntent' -> proof
                        colorspace -> using 'proofRenderingIntent' -> output
                        colorspace

                        Added LCMS FLAGS support.
                        Added FLAGS["SOFTPROOFING"] as default flag for
                        buildProofTransform (otherwise the proof profile/intent
                        would be ignored).

        0.1.0 pil       March 2009 - added to PIL, as PIL.ImageCms

        0.0.2 alpha     Jan 6, 2002

                        Added try/except statements around type() checks of
                        potential CObjects... Python won't let you use type()
                        on them, and raises a TypeError (stupid, if you ask
                        me!)

                        Added buildProofTransformFromOpenProfiles() function.
                        Additional fixes in DLL, see DLL code for details.

        0.0.1 alpha     first public release, Dec. 26, 2002

    Known to-do list with current version (of Python interface, not pyCMSdll):

        none

"""
VERSION = "1.0.0 pil"
core = _imagingcms
INTENT_PERCEPTUAL = 0
INTENT_RELATIVE_COLORIMETRIC = 1
INTENT_SATURATION = 2
INTENT_ABSOLUTE_COLORIMETRIC = 3
DIRECTION_INPUT = 0
DIRECTION_OUTPUT = 1
DIRECTION_PROOF = 2
FLAGS = { "MATRIXINPUT": 1,"MATRIXOUTPUT": 2,"MATRIXONLY": 1 | 2,"NOWHITEONWHITEFIXUP": 4,"NOPRELINEARIZATION": 16,"GUESSDEVICECLASS": 32,"NOTCACHE": 64,"NOTPRECALC": 256,"NULLTRANSFORM": 512,"HIGHRESPRECALC": 1024,"LOWRESPRECALC": 2048,"WHITEBLACKCOMPENSATION": 8192,"BLACKPOINTCOMPENSATION": 8192,"GAMUTCHECK": 4096,"SOFTPROOFING": 16384,"PRESERVEBLACK": 32768,"NODEFAULTRESOURCEDEF": 16777216,"GRIDPOINTS": lambda n: n & 255 << 16 }
_MAX_FLAG = 0
class ImageCmsProfile:
    def __init__(self, profile):
        """
        :param profile: Either a string representing a filename,
            a file like object containing a profile or a
            low-level profile object

        """
        ...
    
    def _set(self, profile, filename: Optional[Any] = ...):
        self.profile = ...
        self.filename = ...
    
    def tobytes(self):
        """
        Returns the profile in a format suitable for embedding in
        saved images.

        :returns: a bytes object containing the ICC profile.
        """
        ...
    


class ImageCmsTransform(Image.ImagePointHandler):
    """
    Transform.  This can be used with the procedural API, or with the standard
    Image.point() method.

    Will return the output profile in the output.info['icc_profile'].
    """
    def __init__(self, input, output, input_mode, output_mode, intent=..., proof: Optional[Any] = ..., proof_intent=..., flags=...):
        self.input_mode = ...
        self.output_mode = ...
        self.output_profile = ...
    
    def point(self, im):
        ...
    
    def apply(self, im, imOut: Optional[Any] = ...):
        ...
    
    def apply_in_place(self, im):
        ...
    


def get_display_profile(handle: Optional[Any] = ...):
    """ (experimental) Fetches the profile for the current display device.
    :returns: None if the profile is not known.
    """
    ...

class PyCMSError(Exception):
    """ (pyCMS) Exception class.
    This is used for all errors in the pyCMS API. """
    ...


def profileToProfile(im, inputProfile, outputProfile, renderingIntent=..., outputMode: Optional[Any] = ..., inPlace: bool = ..., flags=...):
    """
    (pyCMS) Applies an ICC transformation to a given image, mapping from
    inputProfile to outputProfile.

    If the input or output profiles specified are not valid filenames, a
    PyCMSError will be raised.  If inPlace is True and outputMode != im.mode,
    a PyCMSError will be raised.  If an error occurs during application of
    the profiles, a PyCMSError will be raised.  If outputMode is not a mode
    supported by the outputProfile (or by pyCMS), a PyCMSError will be
    raised.

    This function applies an ICC transformation to im from inputProfile's
    color space to outputProfile's color space using the specified rendering
    intent to decide how to handle out-of-gamut colors.

    OutputMode can be used to specify that a color mode conversion is to
    be done using these profiles, but the specified profiles must be able
    to handle that mode.  I.e., if converting im from RGB to CMYK using
    profiles, the input profile must handle RGB data, and the output
    profile must handle CMYK data.

    :param im: An open PIL image object (i.e. Image.new(...) or
        Image.open(...), etc.)
    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this image, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this image, or a profile object
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param outputMode: A valid PIL mode for the output image (i.e. "RGB",
        "CMYK", etc.).  Note: if rendering the image "inPlace", outputMode
        MUST be the same mode as the input, or omitted completely.  If
        omitted, the outputMode will be the same as the mode of the input
        image (im.mode)
    :param inPlace: Boolean.  If True, the original image is modified in-place,
        and None is returned.  If False (default), a new Image object is
        returned with the transform applied.
    :param flags: Integer (0-...) specifying additional flags
    :returns: Either None or a new PIL image object, depending on value of
        inPlace
    :exception PyCMSError:
    """
    ...

def getOpenProfile(profileFilename):
    """
    (pyCMS) Opens an ICC profile file.

    The PyCMSProfile object can be passed back into pyCMS for use in creating
    transforms and such (as in ImageCms.buildTransformFromOpenProfiles()).

    If profileFilename is not a valid filename for an ICC profile, a PyCMSError
    will be raised.

    :param profileFilename: String, as a valid filename path to the ICC profile
        you wish to open, or a file-like object.
    :returns: A CmsProfile class object.
    :exception PyCMSError:
    """
    ...

def buildTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent=..., flags=...):
    """
    (pyCMS) Builds an ICC transform mapping from the inputProfile to the
    outputProfile.  Use applyTransform to apply the transform to a given
    image.

    If the input or output profiles specified are not valid filenames, a
    PyCMSError will be raised.  If an error occurs during creation of the
    transform, a PyCMSError will be raised.

    If inMode or outMode are not a mode supported by the outputProfile (or
    by pyCMS), a PyCMSError will be raised.

    This function builds and returns an ICC transform from the inputProfile
    to the outputProfile using the renderingIntent to determine what to do
    with out-of-gamut colors.  It will ONLY work for converting images that
    are in inMode to images that are in outMode color format (PIL mode,
    i.e. "RGB", "RGBA", "CMYK", etc.).

    Building the transform is a fair part of the overhead in
    ImageCms.profileToProfile(), so if you're planning on converting multiple
    images using the same input/output settings, this can save you time.
    Once you have a transform object, it can be used with
    ImageCms.applyProfile() to convert images without the need to re-compute
    the lookup table for the transform.

    The reason pyCMS returns a class object rather than a handle directly
    to the transform is that it needs to keep track of the PIL input/output
    modes that the transform is meant for.  These attributes are stored in
    the "inMode" and "outMode" attributes of the object (which can be
    manually overridden if you really want to, but I don't know of any
    time that would be of use, or would even work).

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the transform

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """
    ...

def buildProofTransform(inputProfile, outputProfile, proofProfile, inMode, outMode, renderingIntent=..., proofRenderingIntent=..., flags=...):
    """
    (pyCMS) Builds an ICC transform mapping from the inputProfile to the
    outputProfile, but tries to simulate the result that would be
    obtained on the proofProfile device.

    If the input, output, or proof profiles specified are not valid
    filenames, a PyCMSError will be raised.

    If an error occurs during creation of the transform, a PyCMSError will
    be raised.

    If inMode or outMode are not a mode supported by the outputProfile
    (or by pyCMS), a PyCMSError will be raised.

    This function builds and returns an ICC transform from the inputProfile
    to the outputProfile, but tries to simulate the result that would be
    obtained on the proofProfile device using renderingIntent and
    proofRenderingIntent to determine what to do with out-of-gamut
    colors.  This is known as "soft-proofing".  It will ONLY work for
    converting images that are in inMode to images that are in outMode
    color format (PIL mode, i.e. "RGB", "RGBA", "CMYK", etc.).

    Usage of the resulting transform object is exactly the same as with
    ImageCms.buildTransform().

    Proof profiling is generally used when using an output device to get a
    good idea of what the final printed/displayed image would look like on
    the proofProfile device when it's quicker and easier to use the
    output device for judging color.  Generally, this means that the
    output device is a monitor, or a dye-sub printer (etc.), and the simulated
    device is something more expensive, complicated, or time consuming
    (making it difficult to make a real print for color judgement purposes).

    Soft-proofing basically functions by adjusting the colors on the
    output device to match the colors of the device being simulated. However,
    when the simulated device has a much wider gamut than the output
    device, you may obtain marginal results.

    :param inputProfile: String, as a valid filename path to the ICC input
        profile you wish to use for this transform, or a profile object
    :param outputProfile: String, as a valid filename path to the ICC output
        (monitor, usually) profile you wish to use for this transform, or a
        profile object
    :param proofProfile: String, as a valid filename path to the ICC proof
        profile you wish to use for this transform, or a profile object
    :param inMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param outMode: String, as a valid PIL mode that the appropriate profile
        also supports (i.e. "RGB", "RGBA", "CMYK", etc.)
    :param renderingIntent: Integer (0-3) specifying the rendering intent you
        wish to use for the input->proof (simulated) transform

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param proofRenderingIntent: Integer (0-3) specifying the rendering intent
        you wish to use for proof->output transform

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
        they do.
    :param flags: Integer (0-...) specifying additional flags
    :returns: A CmsTransform class object.
    :exception PyCMSError:
    """
    ...

buildTransformFromOpenProfiles = buildTransform
buildProofTransformFromOpenProfiles = buildProofTransform
def applyTransform(im, transform, inPlace: bool = ...):
    """
    (pyCMS) Applies a transform to a given image.

    If im.mode != transform.inMode, a PyCMSError is raised.

    If inPlace is True and transform.inMode != transform.outMode, a
    PyCMSError is raised.

    If im.mode, transform.inMode, or transform.outMode is not supported by
    pyCMSdll or the profiles you used for the transform, a PyCMSError is
    raised.

    If an error occurs while the transform is being applied, a PyCMSError
    is raised.

    This function applies a pre-calculated transform (from
    ImageCms.buildTransform() or ImageCms.buildTransformFromOpenProfiles())
    to an image.  The transform can be used for multiple images, saving
    considerable calculation time if doing the same conversion multiple times.

    If you want to modify im in-place instead of receiving a new image as
    the return value, set inPlace to True.  This can only be done if
    transform.inMode and transform.outMode are the same, because we can't
    change the mode in-place (the buffer sizes for some modes are
    different).  The  default behavior is to return a new Image object of
    the same dimensions in mode transform.outMode.

    :param im: A PIL Image object, and im.mode must be the same as the inMode
        supported by the transform.
    :param transform: A valid CmsTransform class object
    :param inPlace: Bool.  If True, im is modified in place and None is
        returned, if False, a new Image object with the transform applied is
        returned (and im is not changed). The default is False.
    :returns: Either None, or a new PIL Image object, depending on the value of
        inPlace. The profile will be returned in the image's
        info['icc_profile'].
    :exception PyCMSError:
    """
    ...

def createProfile(colorSpace, colorTemp=...):
    """
    (pyCMS) Creates a profile.

    If colorSpace not in ["LAB", "XYZ", "sRGB"], a PyCMSError is raised

    If using LAB and colorTemp != a positive integer, a PyCMSError is raised.

    If an error occurs while creating the profile, a PyCMSError is raised.

    Use this function to create common profiles on-the-fly instead of
    having to supply a profile on disk and knowing the path to it.  It
    returns a normal CmsProfile object that can be passed to
    ImageCms.buildTransformFromOpenProfiles() to create a transform to apply
    to images.

    :param colorSpace: String, the color space of the profile you wish to
        create.
        Currently only "LAB", "XYZ", and "sRGB" are supported.
    :param colorTemp: Positive integer for the white point for the profile, in
        degrees Kelvin (i.e. 5000, 6500, 9600, etc.).  The default is for D50
        illuminant if omitted (5000k).  colorTemp is ONLY applied to LAB
        profiles, and is ignored for XYZ and sRGB.
    :returns: A CmsProfile class object
    :exception PyCMSError:
    """
    ...

def getProfileName(profile):
    """

    (pyCMS) Gets the internal product name for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised If an error occurs while trying to obtain the
    name tag, a PyCMSError is raised.

    Use this function to obtain the INTERNAL name of the profile (stored
    in an ICC tag in the profile itself), usually the one used when the
    profile was originally created.  Sometimes this tag also contains
    additional information supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal name of the profile as stored
        in an ICC tag.
    :exception PyCMSError:
    """
    ...

def getProfileInfo(profile):
    """
    (pyCMS) Gets the internal product information for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the info tag, a PyCMSError
    is raised

    Use this function to obtain the information stored in the profile's
    info tag.  This often contains details about the profile, and how it
    was created, as supplied by the creator.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    ...

def getProfileCopyright(profile):
    """
    (pyCMS) Gets the copyright for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the copyright tag, a PyCMSError
    is raised

    Use this function to obtain the information stored in the profile's
    copyright tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    ...

def getProfileManufacturer(profile):
    """
    (pyCMS) Gets the manufacturer for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the manufacturer tag, a
    PyCMSError is raised

    Use this function to obtain the information stored in the profile's
    manufacturer tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    ...

def getProfileModel(profile):
    """
    (pyCMS) Gets the model for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the model tag, a PyCMSError
    is raised

    Use this function to obtain the information stored in the profile's
    model tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in
        an ICC tag.
    :exception PyCMSError:
    """
    ...

def getProfileDescription(profile):
    """
    (pyCMS) Gets the description for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the description tag, a PyCMSError
    is raised

    Use this function to obtain the information stored in the profile's
    description tag.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: A string containing the internal profile information stored in an
        ICC tag.
    :exception PyCMSError:
    """
    ...

def getDefaultIntent(profile):
    """
    (pyCMS) Gets the default intent name for the given profile.

    If profile isn't a valid CmsProfile object or filename to a profile,
    a PyCMSError is raised.

    If an error occurs while trying to obtain the default intent, a
    PyCMSError is raised.

    Use this function to determine the default (and usually best optimized)
    rendering intent for this profile.  Most profiles support multiple
    rendering intents, but are intended mostly for one type of conversion.
    If you wish to use a different intent than returned, use
    ImageCms.isIntentSupported() to verify it will work first.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :returns: Integer 0-3 specifying the default rendering intent for this
        profile.

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :exception PyCMSError:
    """
    ...

def isIntentSupported(profile, intent, direction):
    """
    (pyCMS) Checks if a given intent is supported.

    Use this function to verify that you can use your desired
    renderingIntent with profile, and that profile can be used for the
    input/output/proof profile as you desire.

    Some profiles are created specifically for one "direction", can cannot
    be used for others.  Some profiles can only be used for certain
    rendering intents... so it's best to either verify this before trying
    to create a transform with them (using this function), or catch the
    potential PyCMSError that will occur if they don't support the modes
    you select.

    :param profile: EITHER a valid CmsProfile object, OR a string of the
        filename of an ICC profile.
    :param intent: Integer (0-3) specifying the rendering intent you wish to
        use with this profile

            ImageCms.INTENT_PERCEPTUAL            = 0 (DEFAULT)
            ImageCms.INTENT_RELATIVE_COLORIMETRIC = 1
            ImageCms.INTENT_SATURATION            = 2
            ImageCms.INTENT_ABSOLUTE_COLORIMETRIC = 3

        see the pyCMS documentation for details on rendering intents and what
            they do.
    :param direction: Integer specifying if the profile is to be used for
        input, output, or proof

            INPUT  = 0 (or use ImageCms.DIRECTION_INPUT)
            OUTPUT = 1 (or use ImageCms.DIRECTION_OUTPUT)
            PROOF  = 2 (or use ImageCms.DIRECTION_PROOF)

    :returns: 1 if the intent/direction are supported, -1 if they are not.
    :exception PyCMSError:
    """
    ...

def versions():
    """
    (pyCMS) Fetches versions.
    """
    ...

