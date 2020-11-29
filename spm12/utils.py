import logging
from functools import lru_cache, wraps
from os import path
from textwrap import dedent

from miutil.mlab import get_engine
from pkg_resources import resource_filename

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
        log.warn("MATLAB could not find SPM.")
        try:
            from zipfile import ZipFile

            from brainweb import get_file

            log.info("Downloading to %s", cache)
            fname = get_file(
                "spm12.zip",
                "https://www.fil.ion.ucl.ac.uk/"
                "spm/download/restricted/eldorado/spm12.zip",
                cache,
                chunk_size=2 ** 17,
            )
            log.info("Extracting")
            with ZipFile(fname) as fd:
                fd.extractall(path=cache)
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
