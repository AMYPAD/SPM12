import logging
from functools import wraps
from os import path
from textwrap import dedent

from miutil.fdio import extractall
from miutil.mlab import get_engine
from miutil.web import urlopen_cached
from pkg_resources import resource_filename

__all__ = ["ensure_spm", "get_matlab", "spm_dir", "spm_dir_eng"]
PATH_M = resource_filename(__name__, "")
log = logging.getLogger(__name__)


def get_matlab(name=None):
    eng = get_engine(name=name)
    log.debug("adding wrappers (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


def spm_dir(cache="~/.spm12", version=12):
    """Internal SPM12 directory"""
    cache = path.expanduser(cache)
    if str(version) != "12":
        raise NotImplementedError
    return path.join(cache, "spm12")


def spm_dir_eng(name=None, cache="~/.spm12", version=12):
    """
    Computed SPM12 directory.
    Uses matlab to find SPM12 directly,
    so may prefer user-installed version to the internal `spm_dir`.
    """
    eng = ensure_spm(name=name, cache=cache, version=version)
    return path.dirname(eng.which("spm_jobman"))


@wraps(get_matlab)
def ensure_spm(name=None, cache="~/.spm12", version=12):
    eng = get_matlab(name)
    cache = path.expanduser(cache)
    addpath = spm_dir(cache=cache, version=version)
    if path.exists(addpath):
        eng.addpath(addpath)
    if not eng.exist("spm_jobman"):
        log.warning("MATLAB could not find SPM.")
        try:
            log.info("Downloading to %s", cache)
            with urlopen_cached(
                "https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip",
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
    found = eng.which("spm_jobman")
    if path.realpath(path.dirname(found)) != path.realpath(addpath):
        log.warning(
            f"Internal ({addpath}) does not match detected ({found}) SPM12.\n"
            "This means `spm_dir()` is likely to fail - use `spm_dir_eng()` instead."
        )
    return eng
