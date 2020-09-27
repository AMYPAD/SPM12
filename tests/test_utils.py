from pytest import importorskip


def test_matlab():
    engine = importorskip("matlab.engine")
    from spm12.utils import get_matlab

    assert not engine.find_matlab()
    eng = get_matlab()

    matrix = eng.eval("eye(3)")
    assert matrix.size == (3, 3)

    assert engine.find_matlab()
    eng2 = get_matlab()
    assert eng == eng2
