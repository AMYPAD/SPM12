from pytest import importorskip


def test_matlab():
    engine = importorskip("matlab.engine")
    from spm12 import utils

    assert not engine.find_matlab()
    eng = utils.get_matlab()

    matrix = eng.eval("eye(3)")
    assert matrix.size == (3, 3)

    assert engine.find_matlab()
    eng2 = utils.get_matlab()
    assert eng == eng2

    utils.ensure_spm()
    assert utils.spm_dir_eng() == utils.spm_dir()
