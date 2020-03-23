"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

class PSDraw:
    """
    Sets up printing to the given file. If **fp** is omitted,
    :py:attr:`sys.stdout` is assumed.
    """
    def __init__(self, fp: Optional[Any] = ...):
        self.fp = ...
    
    def _fp_write(self, to_write):
        ...
    
    def begin_document(self, id: Optional[Any] = ...):
        """Set up printing of a document. (Write Postscript DSC header.)"""
        self.isofont = ...
    
    def end_document(self):
        """Ends printing. (Write Postscript DSC footer.)"""
        ...
    
    def setfont(self, font, size):
        """
        Selects which font to use.

        :param font: A Postscript font name
        :param size: Size in points.
        """
        ...
    
    def line(self, xy0, xy1):
        """
        Draws a line between the two points. Coordinates are given in
        Postscript point coordinates (72 points per inch, (0, 0) is the lower
        left corner of the page).
        """
        ...
    
    def rectangle(self, box):
        """
        Draws a rectangle.

        :param box: A 4-tuple of integers whose order and function is currently
                    undocumented.

                    Hint: the tuple is passed into this format string:

                    .. code-block:: python

                        %d %d M %d %d 0 Vr\n
        """
        ...
    
    def text(self, xy, text):
        """
        Draws text at the given position. You must use
        :py:meth:`~PIL.PSDraw.PSDraw.setfont` before calling this method.
        """
        ...
    
    def image(self, box, im, dpi: Optional[Any] = ...):
        """Draw a PIL image, centered in the given box."""
        ...
    


EDROFF_PS = """\
/S { show } bind def
/P { moveto show } bind def
/M { moveto } bind def
/X { 0 rmoveto } bind def
/Y { 0 exch rmoveto } bind def
/E {    findfont
        dup maxlength dict begin
        {
                1 index /FID ne { def } { pop pop } ifelse
        } forall
        /Encoding exch def
        dup /FontName exch def
        currentdict end definefont pop
} bind def
/F {    findfont exch scalefont dup setfont
        [ exch /setfont cvx ] cvx bind def
} bind def
"""
VDI_PS = """\
/Vm { moveto } bind def
/Va { newpath arcn stroke } bind def
/Vl { moveto lineto stroke } bind def
/Vc { newpath 0 360 arc closepath } bind def
/Vr {   exch dup 0 rlineto
        exch dup neg 0 exch rlineto
        exch neg 0 rlineto
        0 exch rlineto
        100 div setgray fill 0 setgray } bind def
/Tm matrix def
/Ve {   Tm currentmatrix pop
        translate scale newpath 0 0 .5 0 360 arc closepath
        Tm setmatrix
} bind def
/Vf { currentgray exch setgray fill setgray } bind def
"""
ERROR_PS = """\
/landscape false def
/errorBUF 200 string def
/errorNL { currentpoint 10 sub exch pop 72 exch moveto } def
errordict begin /handleerror {
    initmatrix /Courier findfont 10 scalefont setfont
    newpath 72 720 moveto $error begin /newerror false def
    (PostScript Error) show errorNL errorNL
    (Error: ) show
        /errorname load errorBUF cvs show errorNL errorNL
    (Command: ) show
        /command load dup type /stringtype ne { errorBUF cvs } if show
        errorNL errorNL
    (VMstatus: ) show
        vmstatus errorBUF cvs show ( bytes available, ) show
        errorBUF cvs show ( bytes used at level ) show
        errorBUF cvs show errorNL errorNL
    (Operand stargck: ) show errorNL /ostargck load {
        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL
    } forall errorNL
    (Execution stargck: ) show errorNL /estargck load {
        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL
    } forall
    end showpage
} def end
"""
