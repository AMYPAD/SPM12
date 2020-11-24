from pytest import skip
from miutil.mlab import matlabroot


def test_cli():
    if matlabroot("None") == "None":
        skip("MATLAB not installed")
    from spm12.cli import main

    assert 0 == main()
