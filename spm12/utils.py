from __future__ import print_function
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
            _install_engine()
            log.info("installed MATLAB engine for Python")
            from matlab import engine
        except RuntimeError as e:
            raise ImportError(
                str(e)
                + "\n\n"
                + dedent(
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


def _matlab_run(command, jvm=False):
    if not command.endswith("exit"):
        command = command + ", exit"
    return check_output(
        MATLAB_RUN + ([] if jvm else ["-nojvm"]) + [command], stderr=STDOUT
    ).strip()


def matlabroot(default=None):
    try:
        res = check_output(["matlab", "-n"])
        res = res.decode("utf-8")
        res = re.search(r"MATLAB\s+=\s+(\S+)\s*$", res, flags=re.M)
        return res.group(1)
    except:  # noqa: E722
        if default:
            return default
        raise


def _install_engine():
    src = path.join(matlabroot(), "extern", "engines", "python")
    with tmpdir() as td:
        cmd = [sys.executable, "setup.py", "build", "--build-base", td, "install"]
        try:
            return check_output(cmd, cwd=src)
        except CalledProcessError:
            try:
                return check_output(cmd + ["--user"], cwd=src)
            except CalledProcessError as err:
                raise RuntimeError(
                    "Could not run {cmd} in directory {src}:\n\n{err}".format(
                        cmd=" ".join(cmd), src=src, err=err
                    )
                )
