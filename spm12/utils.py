from __future__ import print_function
from os import getenv
try:
    from matlab import engine
except ImportError:
    raise ImportError("Please install MATLAB and its Python module")
else:
    print("Starting MATLAB ... ", end="", flush=True)
    DEFAULT_ENGINE = engine.connect_matlab(name=getenv("SPM12_MATLAB_ENGINE", None))
    print("Done")
