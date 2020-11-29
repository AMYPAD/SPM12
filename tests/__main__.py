import sys
from os import chdir, path
from subprocess import check_call

chdir(path.dirname(path.dirname(path.abspath(__file__))))
check_call(
    [
        sys.executable,
        "-m",
        "pytest",
        "-v",
        "-n=4",
        "-r=xs",
        "--timeout=600",
        "--timeout_method=thread",
        "--log-level=debug",
        "--cov=spm12",
        "--durations=0",
    ]
    + sys.argv[1:],
)
