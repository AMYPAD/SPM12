from textwrap import dedent

import numpy as np
from miutil.imio import nii
from pytest import mark

from spm12 import regseg

MRI2PET = np.array(
    [
        [0.99990508, 0.00800995, 0.01121016, -0.68164088],
        [-0.00806219, 0.99995682, 0.00462244, -1.16235105],
        [-0.01117265, -0.00471238, 0.99992648, -1.02167229],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
no_matlab_warn = mark.filterwarnings("ignore:.*collections.abc:DeprecationWarning")
no_scipy_warn = mark.filterwarnings("ignore:numpy.ufunc size changed.*:RuntimeWarning")


def assert_equal_arrays(x, y, nmse_tol=0, denan=True):
    if denan:
        x, y = map(np.nan_to_num, (x, y))
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
            x.min(), x.mean(), x.max(), x.std(), y.min(), y.mean(), y.max(), y.std(),
        )
    )


@no_scipy_warn
@no_matlab_warn
def test_resample(PET, MRI, tmp_path):
    res = regseg.resample_spm(PET, MRI, MRI2PET, outpath=tmp_path / "resample")
    res = nii.getnii(res)
    ref = nii.getnii(PET)
    assert res.shape == ref.shape
    assert not np.isnan(res).all()


@no_scipy_warn
@no_matlab_warn
def test_coreg(PET, MRI, tmp_path):
    res = regseg.coreg_spm(PET, MRI, outpath=tmp_path / "coreg")
    assert_equal_arrays(res["affine"], MRI2PET, 5e-4)

    outpath = tmp_path / "resamp"
    res = regseg.resample_spm(PET, MRI, res["affine"], outpath=outpath)
    ref = regseg.resample_spm(PET, MRI, MRI2PET, outpath=outpath)
    assert_equal_arrays(nii.getnii(res), nii.getnii(ref), 1e-4)
