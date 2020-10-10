from pytest import skip
from spm12.utils import matlabroot


def test_cli():
    if matlabroot("None") == "None":
        skip("MATLAB not installed")
    from spm12.cli import main

    assert 0 == main()
