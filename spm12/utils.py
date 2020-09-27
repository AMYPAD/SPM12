from __future__ import print_function
import logging
from os import getenv
from functools import lru_cache

try:
    from matlab import engine
except ImportError:
    raise ImportError("Please install MATLAB and its Python module")
__all__ = ["get_matlab"]
log = logging.getLogger(__name__)


@lru_cache()
def get_matlab(name=None):
    started = engine.find_matlab()
    if not started or (name and name not in started):
        notify = True
        log.debug("Starting MATLAB")
    res = engine.connect_matlab(name=name or getenv("SPM12_MATLAB_ENGINE", None))
    if notify:
        log.debug("MATLAB started")
    return res
