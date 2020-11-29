from miutil.mlab import matlabroot
from pytest import skip


def test_cli():
    if matlabroot("None") == "None":
        skip("MATLAB not installed")
    from spm12.cli import main

    assert 0 == main()
