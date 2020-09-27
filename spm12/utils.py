from __future__ import print_function
import logging
import sys
from os import getenv, path, makedirs
from functools import lru_cache
from textwrap import dedent

__all__ = ["create_dir", "get_matlab"]
log = logging.getLogger(__name__)


@lru_cache()
def get_matlab(name=None):
    try:
        from matlab import engine
    except ImportError:
        raise ImportError(
            dedent(
                """\
        Please install MATLAB and its Python module.
        See https://www.mathworks.com/help/matlab/matlab_external/\
install-the-matlab-engine-for-python.html
        or
        https://www.mathworks.com/help/matlab/matlab_external/\
install-matlab-engine-api-for-python-in-nondefault-locations.html
        It's likely you need to do:

        cd "matlabroot\\extern\\engines\\python"
        {exe} setup.py build --build-base="builddir" install --user

        (Start MATLAB and type `matlabroot` in the command window to find
        the relevant directory for the above command. Also fill in any
        temporary directory name for builddir.)
        """
            ).format(exe=sys.executable)
        )
    started = engine.find_matlab()
    if not started or (name and name not in started):
        notify = True
        log.debug("Starting MATLAB")
    res = engine.connect_matlab(name=name or getenv("SPM12_MATLAB_ENGINE", None))
    if notify:
        log.debug("MATLAB started")
    return res


def create_dir(pth):
    """Equivalent of `mkdir -p`"""
    if not path.exists(pth):
        makedirs(pth)
