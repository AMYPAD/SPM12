from os import chdir, path
import sys
from subprocess import check_call

chdir(path.dirname(path.dirname(path.abspath(__file__))))
check_call(
    (
        "{exe} -m pytest -v -n=4 --timeout=600 --timeout_method=thread"
        " --log-level=debug --cov=spm12 {args} --durations=0"
    ).format(exe=sys.executable, args=" ".join(sys.argv[1:])),
    shell=True,
)
