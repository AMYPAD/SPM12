from subprocess import check_call
from textwrap import dedent
import errno
import glob
import logging
import os
import shutil

from amypad.imio import nii
from amypad.utils import create_dir, hasext
import numpy as np
import scipy.ndimage as ndi

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
            opth,
            "affine_ref-" + nii.file_parts(imrefu)[1] + fcomment + ".nii.gz",
        )
    elif pickname == "flo":
        fout = os.path.join(
            opth,
            "affine_flo-" + nii.file_parts(imflo)[1] + fcomment + ".nii.gz",
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


def realign_mltp_spm(
    fims,
    quality=1.0,
    fwhm=6,
    sep=4,
    rtm=1,
    interp=2,
    graph=0,
    outpath="",
    fcomment="",
    niicopy=False,
    niisort=False,
):
    """
    SPM realignment of multiple images through m-scripting (dynamic PET).

    Arguments:
        fims:   has to be a list of at least two files with the first one acting
                as a reference.
    """
    inpath = os.path.dirname(fims[0])
    outpath = os.path.join(outpath or inpath, "align")

    if hasext(fims[0], ".gz") or niicopy:
        tmpth = outpath  # os.path.join(outpath, 'tmp')
        rpth = tmpth
    else:
        tmpth = outpath
        rpth = inpath
    create_dir(tmpth)

    # > uncompress for SPM
    fungz = []
    for f in fims:
        if hasext(f, ".gz"):
            fun = nii.nii_ugzip(f, outpath=tmpth)
        elif os.path.isfile(f) and hasext(f, "nii"):
            if niicopy:
                fun = os.path.join(tmpth, nii.file_parts(f)[1] + "_copy.nii")
                shutil.copyfile(f, fun)
            else:
                fun = f
        else:
            log.warning("omitting file/folder: {}".format(f))
        fungz.append(fun)

    if niisort:
        niisrt = nii.niisort(
            [
                os.path.join(tmpth, f)
                for f in os.listdir(tmpth)
                if os.path.isfile(os.path.join(tmpth, f))
            ]
        )
        fsrt = niisrt["files"]
    else:
        fsrt = fungz

    P_input = [
        f + ",1" for f in fsrt if hasext(f, "nii") and f[0] != "r" and "mean" not in f
    ]

    # > maximal number of characters per line (for Matlab array)
    Pinmx = max([len(f) for f in P_input])

    # > equalise the input char array
    Pineq = [f.ljust(Pinmx) for f in P_input]

    # ---------------------------------------------------------------------------
    # > MATLAB realign flags for SPM
    flgs = []
    pw = "''"

    flgs.append("flgs.quality = " + str(quality))
    flgs.append("flgs.fwhm = " + str(fwhm))
    flgs.append("flgs.sep = " + str(sep))
    flgs.append("flgs.rtm = " + str(rtm))
    flgs.append("flgs.pw  = " + str(pw))
    flgs.append("flgs.interp = " + str(interp))
    flgs.append("flgs.graphics = " + str(graph))
    flgs.append("flgs.wrap = [0 0 0]")

    flgs.append("disp(flgs)")
    # ---------------------------------------------------------------------------

    fscript = os.path.join(outpath, "pyauto_script_realign.m")

    with open(fscript, "w") as f:
        f.write("% AUTOMATICALLY GENERATED MATLAB SCRIPT FOR SPM PET REALIGNMENT\n\n")

        f.write("%> the following flags for PET image alignment are used:\n")
        f.write("disp('m> SPM realign flags:');\n")
        for item in flgs:
            f.write("%s;\n" % item)

        f.write("\n%> the following PET images will be aligned:\n")
        f.write("disp('m> PET images for SPM realignment:');\n")

        f.write("P{1,1} = [...\n")
        for item in Pineq:
            f.write("'%s'\n" % item)
        f.write("];\n\n")
        f.write("disp(P{1,1});\n")

        f.write("spm_realign(P, flgs);")

    cmd = [
        "matlab",
        "-nodisplay",
        "-nodesktop",
        "-r",
        "run(" + "'" + fscript + "'" + "); exit",
    ]
    check_call(cmd)

    fres = glob.glob(os.path.join(rpth, "rp*.txt"))[0]
    res = np.loadtxt(fres)

    return {"fout": fres, "outarray": res, "P": Pineq, "fims": fsrt}


def resample_mltp_spm(
    fims,
    ftr,
    interp=1.0,
    which=1,
    mask=0,
    mean=0,
    graph=0,
    niisort=False,
    prefix="r_",
    outpath="",
    fcomment="",
    pickname="ref",
    copy_input=False,
    del_in_uncmpr=False,
    del_out_uncmpr=False,
):
    """
    SPM resampling of multiple images through m-scripting (dynamic PET).

    Arguments:
        fims:   has to be a list of at least two files with the first one acting
                as a reference.
    """
    if not isinstance(fims, list) and not isinstance(fims[0], str):
        raise ValueError("unrecognised list of input images")

    if not os.path.isfile(ftr):
        raise IOError(
            errno.ENOENT, ftr, "cannot open the file with translations and rotations"
        )

    opth = outpath or os.path.dirname(fims[0])
    create_dir(opth)

    # > working file names (not touching the original ones)
    _fims = []

    # > decompress if necessary
    for f in fims:
        if not os.path.isfile(f) and not nii.RE_NII.search(f):
            raise IOError(errno.ENOENT, f, "could not open file")

        if hasext(f, "gz"):
            fugz = nii.nii_ugzip(f, outpath=os.path.join(opth, "copy"))
        elif copy_input:
            fnm = nii.file_parts(f)[1] + "_copy.nii"
            fugz = os.path.join(opth, "copy", fnm)
            shutil.copyfile(f, fugz)
        else:
            fugz = f

        _fims.append(fugz)

    if niisort:
        niisrt = nii.niisort(_fims)
        _fims = niisrt["files"]

    # > maximal number of characters per line (for Matlab array)
    Pinmx = max([len(f) for f in _fims])

    # > equalise the input char array
    Pineq = [f.ljust(Pinmx) for f in _fims]

    # ---------------------------------------------------------------------------
    # > SPM reslicing (resampling) flags
    flgs = []

    flgs.append("flgs.mask = " + str(mask))
    flgs.append("flgs.mean = " + str(mean))

    flgs.append("flgs.which = " + str(which))
    flgs.append("flgs.interp = " + str(interp))
    flgs.append("flgs.graphics = " + str(graph))
    flgs.append("flgs.wrap = [0 0 0]")
    flgs.append("flgs.prefix = " + "'" + prefix + "'")

    flgs.append("disp(flgs)")
    # ---------------------------------------------------------------------------

    fsrpt = os.path.join(opth, "pyauto_script_resampling.m")
    with open(fsrpt, "w") as f:
        f.write(
            "% AUTOMATICALLY GENERATED MATLAB SCRIPT FOR SPM RESAMPLING PET IMAGES\n\n"
        )

        f.write("%> the following flags for image reslicing are used:\n")
        f.write("disp('m> SPM reslicing flags:');\n")
        for item in flgs:
            f.write("%s;\n" % item)

        f.write("\n%> the following PET images will be aligned:\n")
        f.write("disp('m> PET images for SPM reslicing:');\n")

        f.write("P{1,1} = [...\n")
        for item in Pineq:
            f.write("'%s'\n" % item)
        f.write("];\n\n")
        f.write("disp(P{1,1});\n")

        # f.write('\n%> the following PET images will be aligned'
        #         ' using the translations and rotations in X:\n')
        # f.write("X = dlmread('"+ftr+"');\n")
        # f.write('for fi = 2:'+str(len(fims))+'\n')
        # f.write("    VF = strcat(P{1,1}(fi,:),',1');\n")
        # f.write('    M_ = zeros(4,4);\n')
        # f.write('    M_(:,:) = spm_get_space(VF);\n')
        # f.write('    M = spm_matrix(X(fi,:));\n')
        # f.write('    spm_get_space(VF, M\M_(:,:));\n')
        # f.write('end\n\n')

        f.write("spm_reslice(P, flgs);\n")

    cmd = [
        "matlab",
        "-nodisplay",
        "-nodesktop",
        "-r",
        "run(" + "'" + fsrpt + "'" + "); exit",
    ]
    check_call(cmd)

    if del_in_uncmpr:
        for fpth in _fims:
            os.remove(fpth)
