from __future__ import print_function
from functools import lru_cache, wraps
from os import getenv
from pkg_resources import resource_filename
from textwrap import dedent
import logging
import sys

__all__ = ["get_matlab", "ensure_spm"]
PATH_M = resource_filename(__name__, "")
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

        cd "MATLABROOT\\extern\\engines\\python"
        {exe} setup.py build --build-base="BUILDDIR" install

        - MATLABROOT: start MATLAB and type `matlabroot` in the command window
          to find the relevant directory for the above command.
        - Fill in any temporary directory name for BUILDDIR (e.g. /tmp/builddir).
        - If installation fails due to write permissions, try appending `--user`
          to the above command.
        """
            ).format(exe=sys.executable)
        )
    started = engine.find_matlab()
    if not started or (name and name not in started):
        notify = True
        log.debug("Starting MATLAB")
    eng = engine.connect_matlab(name=name or getenv("SPM12_MATLAB_ENGINE", None))
    if notify:
        log.debug("MATLAB started")
    log.debug("adding SPM (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


@wraps(get_matlab)
def ensure_spm(name=None):
    eng = get_matlab(name)
    if not eng.exist("spm_jobman"):
        raise ImportError(
            dedent(
                """\
            MATLAB cannot find SPM.
            Please follow installation instructions at
            https://en.wikibooks.org/wiki/SPM/Download
            Make sure to add SPM12 to MATLAB's path using `startup.m`
            """
            )
        )
    return eng
