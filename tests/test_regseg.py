from contextlib import contextmanager
from os import path, getenv
from shutil import rmtree
from tempfile import mkdtemp
from textwrap import dedent

from amypad.imio import nii
import numpy as np
import pytest

from spm12 import regseg

HOME = getenv("DATA_ROOT", path.expanduser("~"))
DATA = path.join(HOME, "Ab_PET_mMR_test")
skip_no_data = pytest.mark.skipif(
    not path.exists(DATA),
    reason="cannot find Ab_PET_mMR_test in ${DATA_ROOT:-~} (%s)" % HOME,
)
mri = path.join(DATA, "T1w_N4", "t1_S00113_17598013_N4bias_cut.nii.gz")
pet = path.join(
    DATA,
    "testing_reference",
    "Ab_PET_mMR_ref",
    "basic",
    "17598013_t-3000-3600sec_itr-4_suvr.nii.gz",
)
mri2pet = np.array(
    [
        [0.99990508, 0.00800995, 0.01121016, -0.68164088],
        [-0.00806219, 0.99995682, 0.00462244, -1.16235105],
        [-0.01117265, -0.00471238, 0.99992648, -1.02167229],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@contextmanager
def tmpdir(*args, **kwargs):
    d = mkdtemp(*args, **kwargs)
    yield d
    rmtree(d)


def assert_equal_arrays(x, y, nmse_tol=0):
    if nmse_tol:
        if ((x - y) ** 2).mean() / (y ** 2).mean() < nmse_tol:
            return
    elif (x == y).all():
        return
    raise ValueError(
        dedent(
            """\
        Unequal arrays:x != y. min/mean/max(std):
        x: {:.3g}/{:.3g}/{:.3g}({:.3g})
        y: {:.3g}/{:.3g}/{:.3g}({:.3g})
        """
        ).format(
            x.min(),
            x.mean(),
            x.max(),
            x.std(),
            y.min(),
            y.mean(),
            y.max(),
            y.std(),
        )
    )


@skip_no_data
def test_coreg():
    with tmpdir() as outpath:
        res = regseg.coreg_spm(pet, mri, outpath=outpath)
    assert_equal_arrays(res["affine"], mri2pet, 1e-4)


@skip_no_data
def test_resample():
    with tmpdir() as outpath:
        res = regseg.resample_spm(pet, mri, mri2pet, outpath=outpath)
        res = nii.getnii(res)
        ref = nii.getnii(pet)
        assert res.shape == ref.shape
        assert not np.isnan(res).all()
