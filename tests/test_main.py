from miutil.mlab import matlabroot
from pytest import mark, skip

try:
    FileNotFoundError
except NameError:  # fix py2.7
    FileNotFoundError = OSError

pytestmark = mark.filterwarnings("ignore:numpy.ufunc size changed.*:RuntimeWarning")


def test_cli():
    try:
        if matlabroot("None") == "None":
            raise FileNotFoundError
    except FileNotFoundError:
        skip("MATLAB not installed")
    from spm12.cli import main

    assert 0 == main()
