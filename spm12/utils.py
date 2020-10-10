from __future__ import print_function
from ast import literal_eval
from functools import lru_cache, wraps
from os import getenv, path
from pkg_resources import resource_filename
from subprocess import CalledProcessError, STDOUT, check_output
from textwrap import dedent
import logging
import re
import sys

from amypad.utils import tmpdir

__all__ = ["get_matlab", "ensure_spm"]
PATH_M = resource_filename(__name__, "")
MATLAB_RUN = "matlab -nodesktop -nodisplay -nosplash -nojvm -r".split()
log = logging.getLogger(__name__)


class VersionError(ValueError):
    pass


@lru_cache()
def get_matlab(name=None):
    try:
        from matlab import engine
    except ImportError:
        try:
            log.warn(
                dedent(
                    """\
                Python could not find the MATLAB engine.
                Attempting to install automatically."""
                )
            )
            log.debug(_install_engine())
            log.info("installed MATLAB engine for Python")
            from matlab import engine
        except CalledProcessError:
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

                cd "{matlabroot}\\extern\\engines\\python"
                {exe} setup.py build --build-base="BUILDDIR" install

                - Fill in any temporary directory name for BUILDDIR
                  (e.g. /tmp/builddir).
                - If installation fails due to write permissions, try appending `--user`
                  to the above command.
                """
                ).format(
                    matlabroot=matlabroot(default="matlabroot"), exe=sys.executable
                )
            )
    started = engine.find_matlab()
    if not started or (name and name not in started):
        notify = True
        log.debug("Starting MATLAB")
    eng = engine.connect_matlab(name=name or getenv("SPM12_MATLAB_ENGINE", None))
    if notify:
        log.debug("MATLAB started")
    log.debug("adding wrappers (%s) to MATLAB path", PATH_M)
    eng.addpath(PATH_M, nargout=0)
    return eng


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
            from brainweb import get_file
            from zipfile import ZipFile

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
        except:  # NOQA: E722
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


def _matlab_run(command, jvm=False, auto_exit=True):
    if auto_exit and not command.endswith("exit"):
        command = command + ", exit"
    return (
        check_output(
            MATLAB_RUN + ([] if jvm else ["-nojvm"]) + [command], stderr=STDOUT
        )
        .decode("utf-8")
        .strip()
    )


def matlabroot(default=None):
    try:
        res = check_output(["matlab", "-n"]).decode("utf-8")
    except CalledProcessError:
        if default:
            return default
        raise
    return re.search(r"MATLAB\s+=\s+(\S+)\s*$", res, flags=re.M).group(1)


def _install_engine():
    src = path.join(matlabroot(), "extern", "engines", "python")
    with open(path.join(src, "setup.py")) as fd:  # check version support
        supported = literal_eval(
            re.search(r"supported_version.*?=\s*(.*?)$", fd.read(), flags=re.M).group(1)
        )
        if ".".join(map(str, sys.version_info[:2])) not in map(str, supported):
            raise VersionError(
                dedent(
                    """\
                Python version is {info[0]}.{info[1]},
                but the installed MATLAB only supports Python versions: [{supported}]
                """.format(
                        info=sys.version_info[:2], supported=", ".join(supported)
                    )
                )
            )
    with tmpdir() as td:
        cmd = [sys.executable, "setup.py", "build", "--build-base", td, "install"]
        try:
            return check_output(cmd, cwd=src).decode("utf-8")
        except CalledProcessError:
            log.warn("Normal install failed. Attempting `--user` install.")
            return check_output(cmd + ["--user"], cwd=src).decode("utf-8")
