import logging
import os
from functools import wraps
from os import path
from subprocess import CalledProcessError, check_output
from textwrap import dedent

from miutil.fdio import Path, extractall, fspath
from miutil.mlab import get_engine, get_runtime
from miutil.web import urlopen_cached
from pkg_resources import resource_filename

try:
    from functools import lru_cache
except ImportError: # fix py2.7
    from backports.functools_lru_cache import lru_cache

__all__ = ["ensure_spm", "get_matlab", "spm_dir"]
PATH_M = resource_filename(__name__, "")
log = logging.getLogger(__name__)
SPM12_ZIP = "https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip"
MCR_ZIP = "https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/spm12_r7771.zip"


def env_prefix(key, dir):
    os.environ[key] = "%s%s%s" % (os.environ[key], os.pathsep, fspath(dir))


def spm_runtime(cache="~/.spm12", version=12):
    cache = Path(cache).expanduser()
    if str(version) != "12":
        raise NotImplementedError
    runtime = cache / "runtime"
    if not runtime.is_dir():
        log.info("Downloading to %s", cache)
        with urlopen_cached(MCR_ZIP, cache) as fd:
            extractall(fd, runtime)

    runner = runtime / "spm12" / "run_spm12.sh"
    runner.chmod(0o755)
    return fspath(runner)


def mcr_run(*cmd, cache="~/.spm12", version=12, mcr_version=713):
    mcr_root = fspath(get_runtime(version=mcr_version))
    runner = spm_runtime(cache=cache, version=version)
    try:
        return check_output((runner, mcr_root) + cmd).decode("U8").strip()
    except CalledProcessError as err:
        raise RuntimeError(
            dedent("""\
            {}

            See https://en.wikibooks.org/wiki/SPM/Standalone#Trouble-shooting
            """).format(err))


@lru_cache()
def get_matlab(name=None):
    eng = get_engine(name=name)
    log.debug("adding wrappers (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


def spm_dir(cache="~/.spm12", version=12):
    cache = path.expanduser(cache)
    if str(version) != "12":
        raise NotImplementedError
    return path.join(cache, "spm12")


@lru_cache()
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
            with urlopen_cached(SPM12_ZIP, cache) as fd:
                extractall(fd, cache)
            eng.addpath(addpath)
            if not eng.exist("spm_jobman"):
                raise RuntimeError("MATLAB could not find SPM.")
            log.info("Installed")
        except:                # NOQA: E722,B001
            raise ImportError(
                dedent("""\
                MATLAB could not find SPM.
                Please follow installation instructions at
                https://en.wikibooks.org/wiki/SPM/Download
                Make sure to add SPM to MATLAB's path using `startup.m`
                """))
    return eng
