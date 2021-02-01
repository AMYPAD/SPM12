import logging
from functools import wraps
from os import path
from textwrap import dedent

from miutil.fdio import extractall
from miutil.mlab import get_engine
from miutil.web import urlopen_cached
from pkg_resources import resource_filename

try:
    from functools import lru_cache
except ImportError:  # fix py2.7
    from backports.functools_lru_cache import lru_cache

__all__ = ["get_matlab", "ensure_spm"]
PATH_M = resource_filename(__name__, "")
log = logging.getLogger(__name__)


@lru_cache()
def get_matlab(name=None):
    eng = get_engine(name=name)
    log.debug("adding wrappers (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


@lru_cache()
@wraps(get_matlab)
def ensure_spm(name=None, cache="~/.spm12", version=12):
    eng = get_matlab(name)
    cache = path.expanduser(cache)
    if str(version) != "12":
        raise NotImplementedError
    addpath = path.join(cache, "spm12")
    if path.exists(addpath):
        eng.addpath(addpath)
    if not eng.exist("spm_jobman"):
        log.warning("MATLAB could not find SPM.")
        try:
            log.info("Downloading to %s", cache)
            with urlopen_cached(
                "https://www.fil.ion.ucl.ac.uk/"
                "spm/download/restricted/eldorado/spm12.zip",
                cache,
            ) as fd:
                extractall(fd, cache)
            eng.addpath(addpath)
            if not eng.exist("spm_jobman"):
                raise RuntimeError("MATLAB could not find SPM.")
            log.info("Installed")
        except:  # NOQA: E722,B001
            raise ImportError(
                dedent(
                    """\
                MATLAB could not find SPM.
                Please follow installation instructions at
                https://en.wikibooks.org/wiki/SPM/Download
                Make sure to add SPM to MATLAB's path using `startup.m`
                """
                )
            )
    return eng
