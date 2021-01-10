from miutil.mlab import matlabroot
from pytest import skip


def test_cli():
    try:
        if matlabroot("None") == "None":
            raise FileNotFoundError
    except FileNotFoundError:
        skip("MATLAB not installed")
    from spm12.cli import main

    assert 0 == main()
