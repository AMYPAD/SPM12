import errno
import logging
import os
import shutil
from textwrap import dedent

import numpy as np
import scipy.ndimage as ndi
from miutil import create_dir, hasext
from miutil.imio import nii

from .utils import ensure_spm

__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")
log = logging.getLogger(__name__)


def fwhm2sig(fwhm, voxsize=2.0):
    return fwhm / (voxsize * (8 * np.log(2)) ** 0.5)


def smoothim(fim, fwhm=4, fout=""):
    """
    Smooth image using Gaussian filter with FWHM given as an option.
    """
    imd = nii.getnii(fim, output="all")
    imsmo = ndi.filters.gaussian_filter(
        imd["im"], fwhm2sig(fwhm, voxsize=imd["voxsize"]), mode="mirror"
    )

    if not fout:
        fout = "{f[0]}{f[1]}_smo{fwhm}{f[2]}".format(
            f=nii.file_parts(fim), fwhm=str(fwhm).replace(".", "-")
        )

    nii.array2nii(
        imsmo,
        imd["affine"],
        fout,
        trnsp=(
            imd["transpose"].index(0),
            imd["transpose"].index(1),
            imd["transpose"].index(2),
        ),
        flip=imd["flip"],
    )

    return {"im": imsmo, "fim": fout, "fwhm": fwhm, "affine": imd["affine"]}


def coreg_spm(
    imref,
    imflo,
    matlab_eng_name="",
    fwhm_ref=0,
    fwhm_flo=0,
    outpath="",
    fname_aff="",
    fcomment="",
    pickname="ref",
    costfun="nmi",
    sep=None,
    tol=None,
    fwhm=None,
    params=None,
    graphics=1,
    visual=0,
    del_uncmpr=True,
    save_arr=True,
    save_txt=True,
):
    sep = sep or [4, 2]
    tol = tol or [
        0.0200,
        0.0200,
        0.0200,
        0.0010,
        0.0010,
        0.0010,
        0.0100,
        0.0100,
        0.0100,
        0.0010,
        0.0010,
        0.0010,
    ]
    fwhm = fwhm or [7, 7]
    params = params or [0, 0, 0, 0, 0, 0]
    eng = ensure_spm(matlab_eng_name)  # get_matlab

    if not outpath and fname_aff and "/" in fname_aff:
        opth = os.path.dirname(fname_aff) or os.path.dirname(imflo)
        fname_aff = os.path.basename(fname_aff)
    else:
        opth = outpath or os.path.dirname(imflo)
    log.debug("output path:%s", opth)
    create_dir(opth)

    # > decompress ref image as necessary
    if hasext(imref, "gz"):
        imrefu = nii.nii_ugzip(imref, outpath=opth)
    else:
        fnm = nii.file_parts(imref)[1] + "_copy.nii"
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)

    if fwhm_ref > 0:
        smodct = smoothim(imrefu, fwhm_ref)

        # > delete the previous version (non-smoothed)
        os.remove(imrefu)
        imrefu = smodct["fim"]

        log.info(
            "smoothed the reference image with FWHM={} and saved to\n{}".format(
                fwhm_ref, imrefu
            )
        )

    # > floating
    if hasext(imflo, "gz"):
        imflou = nii.nii_ugzip(imflo, outpath=opth)
    else:
        fnm = nii.file_parts(imflo)[1] + "_copy.nii"
        imflou = os.path.join(opth, fnm)
        shutil.copyfile(imflo, imflou)

    if fwhm_flo > 0:
        smodct = smoothim(imflou, fwhm_flo)
        # > delete the previous version (non-smoothed)
        os.remove(imflou)
        imflou = smodct["fim"]

        log.info(
            "smoothed the floating image with FWHM={} and saved to\n{}".format(
                fwhm_ref, imrefu
            )
        )

    # run the matlab SPM coregistration
    import matlab as ml

    Mm, xm = eng.coreg_spm_m(
        imrefu,
        imflou,
        costfun,
        ml.double(sep),
        ml.double(tol),
        ml.double(fwhm),
        ml.double(params),
        graphics,
        visual,
        nargout=2,
    )

    # get the affine matrix
    M = np.array(Mm._data.tolist())
    M = M.reshape(4, 4).T

    # get the translation and rotation parameters in a vector
    x = np.array(xm._data.tolist())

    # delete the uncompressed files
    if del_uncmpr:
        os.remove(imrefu)
        os.remove(imflou)

    create_dir(os.path.join(opth, "affine-spm"))

    # ---------------------------------------------------------------------------
    if fname_aff == "":
        if pickname == "ref":
            faff = os.path.join(
                opth,
                "affine-spm",
                "affine-ref-" + nii.file_parts(imref)[1] + fcomment + ".npy",
            )
        else:
            faff = os.path.join(
                opth,
                "affine-spm",
                "affine-flo-" + nii.file_parts(imflo)[1] + fcomment + ".npy",
            )

    else:

        # > add '.npy' extension if not in the affine output file name
        if not fname_aff.endswith(".npy"):
            fname_aff += ".npy"

        faff = os.path.join(opth, "affine-spm", fname_aff)
    # ---------------------------------------------------------------------------

    # > safe the affine transformation
    if save_arr:
        np.save(faff, M)
    if save_txt:
        faff = os.path.splitext(faff)[0] + ".txt"
        np.savetxt(faff, M)

    return {
        "affine": M,
        "faff": faff,
        "rotations": x[3:],
        "translations": x[:3],
        "matlab_eng": eng,
    }


def resample_spm(
    imref,
    imflo,
    M,
    matlab_eng_name="",
    fwhm=0,
    intrp=1.0,
    which=1,
    mask=0,
    mean=0,
    outpath="",
    fimout="",
    fcomment="",
    prefix="r_",
    pickname="ref",
    del_ref_uncmpr=False,
    del_flo_uncmpr=False,
    del_out_uncmpr=False,
):
    log.debug(
        dedent(
            """\
        ======================================================================
         S P M  inputs:
         > ref:' {}
         > flo:' {}
        ======================================================================"""
        ).format(imref, imflo)
    )
    eng = ensure_spm(matlab_eng_name)  # get_matlab

    if not outpath and fimout:
        opth = os.path.dirname(fimout) or os.path.dirname(imflo)
    else:
        opth = outpath or os.path.dirname(imflo)
    log.debug("output path:%s", opth)
    create_dir(opth)

    # > decompress if necessary
    if hasext(imref, "gz"):
        imrefu = nii.nii_ugzip(imref, outpath=opth)
    else:
        fnm = nii.file_parts(imref)[1] + "_copy.nii"
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)

    # > floating
    if hasext(imflo, "gz"):
        imflou = nii.nii_ugzip(imflo, outpath=opth)
    else:
        fnm = nii.file_parts(imflo)[1] + "_copy.nii"
        imflou = os.path.join(opth, fnm)
        shutil.copyfile(imflo, imflou)

    if isinstance(M, str):
        if hasext(M, ".txt"):
            M = np.loadtxt(M)
            log.info("matrix M given in the form of text file")
        elif hasext(M, ".npy"):
            M = np.load(M)
            log.info("matrix M given in the form of NumPy file")
        else:
            raise IOError(
                errno.ENOENT, M, "Unrecognised file extension for the affine."
            )
    elif isinstance(M, (np.ndarray, np.generic)):
        log.info("matrix M given in the form of Numpy array")
    else:
        raise ValueError("unrecognised affine matrix format")

    # run the Matlab SPM resampling
    import matlab as ml

    eng.resample_spm_m(
        imrefu, imflou, ml.double(M.tolist()), mask, mean, intrp, which, prefix
    )

    # -compress the output
    split = os.path.split(imflou)
    fim = os.path.join(split[0], prefix + split[1])
    nii.nii_gzip(fim, outpath=opth)

    # delete the uncompressed
    if del_ref_uncmpr:
        os.remove(imrefu)
    if del_flo_uncmpr and os.path.isfile(imflou):
        os.remove(imflou)
    if del_out_uncmpr:
        os.remove(fim)

    # > the compressed output naming
    if fimout:
        fout = os.path.join(opth, fimout)
    elif pickname == "ref":
        fout = os.path.join(
            opth, "affine_ref-" + nii.file_parts(imrefu)[1] + fcomment + ".nii.gz",
        )
    elif pickname == "flo":
        fout = os.path.join(
            opth, "affine_flo-" + nii.file_parts(imflo)[1] + fcomment + ".nii.gz",
        )
    # change the file name
    os.rename(fim + ".gz", fout)

    if fwhm > 0:
        smodct = smoothim(fout, fwhm)
        log.info(
            "smoothed the resampled image with FWHM={} and saved to\n{}".format(
                fwhm, smodct["fim"]
            )
        )

    return fout
