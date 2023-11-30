__author__ = ("Pawel J. Markiewicz", "Casper O. da Costa-Luis")

import errno
import logging
import os
import re
import shutil
from numbers import Number
from pathlib import Path, PurePath
from textwrap import dedent

import numpy as np
import scipy.ndimage as ndi
from miutil import create_dir, hasext
from miutil.imio import nii

from .utils import ensure_spm, spm_dir
from .standalone import standalone_coreg, standalone_seg, standalone_normw
from .setup_rt import ensure_standalone


log = logging.getLogger(__name__)


def move_files(fin, opth):
    """
    Move input file path fin to the output folder opth.
    """
    fdst = os.path.join(opth, os.path.basename(fin))
    shutil.move(fin, fdst)
    return fdst


def glob_match(pttrn, pth):
    """
    glob with regular expressions
    """
    return (os.path.join(pth, f) for f in os.listdir(pth) if re.match(pttrn, f))


def fwhm2sig(fwhm, voxsize=2.0):
    return fwhm / (voxsize * (8 * np.log(2))**0.5)


def smoothim(fim, fwhm=4, fout=""):
    """
    Smooth image using Gaussian filter with FWHM given as an option.
    """
    imd = nii.getnii(fim, output="all")
    imsmo = ndi.filters.gaussian_filter(imd["im"], fwhm2sig(fwhm, voxsize=imd["voxsize"]),
                                        mode="constant")
    if not fout:
        f = nii.file_parts(fim)
        fout = os.path.join(f[0], f"{f[1]}_smo{str(fwhm).replace('.', '-')}{f[2]}")
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


def get_bbox(fnii):
    """get the SPM equivalent of the bounding box for
    NIfTI image `fnii` which can be a dictionary or file.
    """

    if isinstance(fnii, (str, PurePath)):
        niidct = nii.getnii(fnii, output="all")
    elif isinstance(fnii, dict) and "hdr" in fnii:
        niidct = fnii
    else:
        raise ValueError("incorrect input NIfTI file/dictionary")

    dim = niidct["hdr"]["dim"]
    corners = np.array([[1, 1, 1, 1], [1, 1, dim[3], 1], [1, dim[2], 1, 1], [1, dim[2], dim[3], 1],
                        [dim[1], 1, 1, 1], [dim[1], 1, dim[3], 1], [dim[1], dim[2], 1, 1],
                        [dim[1], dim[2], dim[3], 1]])

    XYZ = np.dot(niidct["affine"][:3, :], corners.T)

    # FIXME: weird correction for SPM bounding box (??)
    crr = np.dot(niidct["affine"][:3, :3], [1, 1, 1])

    # bounding box as matrix
    bbox = np.concatenate((np.min(XYZ, axis=1) - crr, np.max(XYZ, axis=1) - crr))
    bbox.shape = (2, 3)

    return bbox


def mat2array(matlab_mat):
    if hasattr(matlab_mat, '_data'): # matlab<R2022a
        return np.array(matlab_mat._data).reshape(matlab_mat.size, order='F')
    return np.array(matlab_mat)

#====================================================================
def coreg_spm(
    imref,
    imflo,
    matlab_eng_name="",
    fwhm_ref=0,
    fwhm_flo=0,
    outpath="",
    output_eng=False,
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
    modify_nii=False,
    standalone=False,
):
    """Rigid body registration using SPM `coreg` function.
    Arguments:
      imref: reference image
      imflo: floating image
      fwhm_ref: FWHM of the smoothing kernel for the reference image
      fwhm_flo: FWHM of the smoothing kernel for the reference image
      modify_nii: modifies the affine of the NIfTI file of the floating
        image according to the rigid body transformation.
      standalone: if True, uses the standalone SPM12 with MATLAB Runtime;
                  if it is not installed it will attempt installing the 
                  MATLAB Runtime and SPM12 standalone.
    """
    out = {}  # output dictionary
    sep = sep or [4, 2]

    tol = tol or [
        0.0200, 0.0200, 0.0200, 0.0010, 0.0010, 0.0010, 0.0100, 0.0100, 0.0100, 0.0010, 0.0010,
        0.0010]

    fwhm = fwhm or [7, 7]
    params = params or [0, 0, 0, 0, 0, 0]

    if standalone:
        modify_nii = True
        ensure_standalone()

    if not outpath and fname_aff and "/" in fname_aff:
        opth = os.path.dirname(fname_aff) or os.path.dirname(imflo)
        fname_aff = os.path.basename(fname_aff)
    else:
        opth = outpath or os.path.dirname(imflo)
    log.debug("output path:%s", opth)
    create_dir(opth)

    # decompress ref image as necessary
    if hasext(imref, "gz"):
        imrefu = nii.nii_ugzip(imref, outpath=opth)
    else:
        fnm = nii.file_parts(imref)[1] + "_copy.nii"
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)

    if fwhm_ref > 0:
        smodct = smoothim(imrefu, fwhm_ref)
        # delete the previous version (non-smoothed)
        os.remove(imrefu)
        imrefu = smodct["fim"]
        log.info("smoothed the reference image with FWHM=%r and saved to\n%r", fwhm_ref, imrefu)

    # floating
    if hasext(imflo, "gz"):
        imflou = nii.nii_ugzip(imflo, outpath=opth)
    else:
        fnm = nii.file_parts(imflo)[1] + "_copy.nii"
        imflou = os.path.join(opth, fnm)
        shutil.copyfile(imflo, imflou)

    if fwhm_flo > 0:
        smodct = smoothim(imflou, fwhm_flo)
        # delete the previous version (non-smoothed)
        if not modify_nii:
            os.remove(imflou)
        else:
            # save the uncompressed and unsmoothed version
            imflou_ = imflou

        imflou = smodct["fim"]

        log.info("smoothed the floating image with FWHM=%r and saved to\n%r", fwhm_flo, imflou)

    if not standalone:
        # > ensure MATLAB and SPM
        eng = ensure_spm(matlab_eng_name) # get_matlab
        import matlab as ml

        # > run registration using standard MATLAB
        Mm, xm = eng.amypad_coreg(
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

        # modify the affine of the floating image (as usually done in SPM)
        if modify_nii:
            eng.amypad_coreg_modify_affine(imflou_, Mm)
            out["freg"] = imflou_

        # get the affine matrix
        M = mat2array(Mm)

        # get the translation and rotation parameters in a vector
        x = mat2array(xm)

        #----------------------------------------
        # > save affine
        create_dir(os.path.join(opth, "affine-spm"))

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
            # add '.npy' extension if not in the affine output file name
            if not fname_aff.endswith(".npy"):
                fname_aff += ".npy"
            faff = os.path.join(opth, "affine-spm", fname_aff)
        

        # > save the affine transformation
        if save_arr:
            np.save(faff, M)
        if save_txt:
            faff = os.path.splitext(faff)[0] + ".txt"
            np.savetxt(faff, M)
        #----------------------------------------

        out["affine"] = M
        out["faff"] = faff
        out["rotations"] = x[3:]
        out["translations"] = x[:3]
        if output_eng:
            out["matlab_eng"] = eng

    else:
        # > Standalone SPM12
        if modify_nii:
            foth = imflou_
        else:
            foth = None

        out['fbatch'] = standalone_coreg(
            imrefu,
            imflou,
            foth,
            cost_fun=costfun,
            sep=sep,
            tol=tol,
            fwhm=fwhm)

        out['freg'] = imflou_
    

    # > delete the uncompressed files
    if del_uncmpr:
        os.remove(imrefu)
        os.remove(imflou)

    return out
#====================================================================


#====================================================================
def resample_spm(
    imref,
    imflo,
    M,
    matlab_eng_name="",
    fwhm=0,
    intrp=1,
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
    standalone=False
):
    log.debug(
        dedent("""\
        ======================================================================
         S P M  inputs:
         > ref:' %r
         > flo:' %r
        ======================================================================"""),
        imref,
        imflo,
    )
    eng = ensure_spm(matlab_eng_name) # get_matlab

    if not outpath and fimout:
        opth = os.path.dirname(fimout) or os.path.dirname(imflo)
    else:
        opth = outpath or os.path.dirname(imflo)
    log.debug("output path:%s", opth)
    create_dir(opth)

    # decompress if necessary
    if hasext(imref, "gz"):
        imrefu = nii.nii_ugzip(imref, outpath=opth)
    else:
        fnm = nii.file_parts(imref)[1] + "_copy.nii"
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)

    # floating
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
            raise IOError(errno.ENOENT, M, "Unrecognised file extension for the affine.")
    elif isinstance(M, (np.ndarray, np.generic)):
        log.info("matrix M given in the form of Numpy array")
    else:
        raise ValueError("unrecognised affine matrix format")

    #--------------------------------------------
    # run the Matlab SPM resampling
    import matlab as ml

    eng.amypad_resample(
        imrefu,
        imflou,
        ml.double(M.tolist()),
        mask,
        mean,
        float(intrp),
        which,
        prefix,
    )
    #--------------------------------------------

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

    # the compressed output naming
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
        log.info("smoothed the resampled image with FWHM=%r and saved to\n%r", fwhm, smodct["fim"])

    return fout


#====================================================================
def seg_spm(
    f_mri,
    spm_path=None,
    matlab_eng_name="",
    outpath=None,
    store_nat_gm=False,
    store_nat_wm=False,
    store_nat_csf=False,
    store_nat_bon=False,
    store_fwd=False,
    store_inv=False,
    visual=False,
    standalone=False,
):
    """
    Normalisation/segmentation using SPM12.
    Args:
      f_mri: file path to the T1w MRI file
      spm_path(str): SPM path
      matlab_eng_name: name of the Python engine for Matlab.
      outpath: output folder path for the normalisation file output
      store_nat_*: stores native space segmentation output for either
        grey matter, white matter or CSF
      sotre_fwd/inv: stores forward/inverse normalisation definitions
      visual: shows the Matlab window progress
    """

    # > output dictionary
    out = {}                          

    if not standalone:

        # > run SPM normalisation/segmentation using standard MATLAB
        # > get Matlab engine or use the provided one
        eng = ensure_spm(matlab_eng_name) 
        if not spm_path:
            spm_path = spm_dir()

        param, invdef, fordef = eng.amypad_seg(
            f_mri,
            str(spm_path),
            float(store_nat_gm),
            float(store_nat_wm),
            float(store_nat_csf),
            float(store_fwd),
            float(store_inv),
            float(visual),
            nargout=3,
        )

        if outpath is not None:
            create_dir(outpath)
            out["param"] = move_files(param, outpath)
            out["invdef"] = move_files(invdef, outpath)
            out["fordef"] = move_files(fordef, outpath)

            # move each tissue type to the output folder
            for c in glob_match(r"c\d*", os.path.dirname(param)):
                nm = os.path.basename(c)[:2]
                out[nm] = move_files(c, outpath)
        else:
            out["param"] = param
            out["invdef"] = invdef
            out["fordef"] = fordef

            for c in glob_match(r"c\d*", os.path.dirname(param)):
                nm = os.path.basename(c)[:2]
                out[nm] = c

    else:
        # > run standalone SPM12 using MATLAB Runtime (no license needed)
        out = standalone_seg(
            f_mri,
            outpath=outpath,
            nat_gm=store_nat_gm,
            nat_wm=store_nat_wm,
            nat_csf=store_nat_csf,
            nat_bn=store_nat_bon,
            biasreg=0.001,
            biasfwhm=60,
            mrf_cleanup=1,
            cleanup=1,
            regulariser=[0, 0.001, 0.5, 0.05, 0.2],
            affinereg='mni',
            fwhm=0,
            sampling=3,
            store_fwd=store_fwd,
            store_inv=store_inv)

    
    return out
#====================================================================


#====================================================================
def normw_spm(
    f_def,
    files4norm,
    outpath=None,
    voxsz=2,
    intrp=4,
    bbox=None,
    matlab_eng_name="",
    standalone=False):

    """
    Write normalisation output to NIfTI files using SPM12.
    Args:
      f_def:    NIfTI file of definitions for non-rigid normalisation
      files4norm: list or single Path/string of input NIfTI file 
                path(s)
      voxsz:    voxel size of the output (normalised) images
      intrp:    interpolation level used for the normalised images
                (4: B-spline, default)
      matlab_eng_name: name of the Python engine for Matlab.
      outpath:  output folder path for the normalisation files
    """

    if isinstance(files4norm, (str, PurePath)):
        files4norm = str(files4norm)
    elif isinstance (files4norm, list):
        files4norm = [str(f) for f in files4norm]
    else:
        raise ValueError('unknown input type for `files4norm` (only strings, Paths or list of Paths/strings is accepted)')

    if not standalone:
        import matlab as ml

        list4norm = [f+',1' for f in files4norm]  

        if bbox is None:
            bb = ml.double([[np.NaN, np.NaN, np.NaN], [np.NaN, np.NaN, np.NaN]])
        elif isinstance(bbox, np.ndarray) and bbox.shape == (2, 3):
            bb = ml.double(bbox.tolist())
        elif isinstance(bbox, list) and len(bbox) == 2:
            bb = ml.double(bbox)
        else:
            raise ValueError("unrecognised format for bounding box")

        if isinstance(voxsz, Number):
            voxsz = ml.double([voxsz] * 3)
        elif isinstance(voxsz, (np.ndarray, list)):
            if len(voxsz) != 3:
                raise ValueError(f"voxel size ({voxsz}) should be scalar or 3-vector")
            voxsz = ml.double(np.float64(voxsz))
        else:
            raise ValueError(f"voxel size ({voxsz}) should be scalar or 3-vector")

        eng = ensure_spm(matlab_eng_name) # get_matlab
        eng.amypad_normw(f_def, list4norm, voxsz, float(intrp), bb)
        out = []                          # output list

        if outpath is not None:
            create_dir(outpath)
            for f in files4norm:
                fpth = f #.split(",")[0]
                out.append(
                    move_files(
                        os.path.join(os.path.dirname(fpth), "w" + os.path.basename(fpth)),
                        outpath,
                    ))
        else:
            out.append("w" + os.path.basename(f.split(",")[0]))

    # > Standalone SPM12
    else:

        out = standalone_normw(
            f_def,
            files4norm,
            bbox=bbox,
            voxsz=voxsz,
            intrp=intrp,
            prfx='w',
            outpath=outpath)

    return out
#====================================================================
