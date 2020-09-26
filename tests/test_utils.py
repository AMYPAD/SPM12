import pytest


def test_matlab():
    pytest.importorskip("matlab.engine")
    from spm12.utils import DEFAULT_ENGINE as ENG

    assert ENG.eval("eye(3)").size == (3, 3)
