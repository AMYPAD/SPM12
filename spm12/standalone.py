'''
Standalone SPM12 functionalities running on MATLAB Runtime
'''
__author__ = "Pawel Markiewicz"
__copyright__ = "Copyright 2023"



import os
from pathlib import Path, PurePath
import subprocess

from niftypet import nimpa
import spm12

import logging
log = logging.getLogger(__name__)



fmri = Path('D:/data/reg_test/04000177_MRI_T1_N4_N4bias_com-modified.nii')
fpet = Path('D:/data/reg_test/UR-aligned_4-summed-frames_DY_MRAC_20MIN__PETBrain_static_com-modified.nii')

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def standalone_coreg(
    fref,
    fflo,
    foth=None,
    cost_fun='nmi',
    sep=[4, 2],
    tol=[0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001],
    fwhm=[7, 7],
    ):

    ''' Run SPM12 coregistration using SPM12 standalone on MATLAB Runtime
        Arguments:
        fref:   file path to the reference image (uncompressed NIfTI)
        fflo:   file path to the floating image (uncompressed NIfTI)
    '''

    if not spm12.check_standalone():
        log.error('MATLAB Runtime or standalone SPM12 has not been correctly installed.\nAttempting installation... ')
        response = input('Do you want to install MATLAB Runtime? [y/n]')
        if response in ['y', 'Y', 'yes']:
            spm12.install_standalone()
    else:
        fspm = spm12.standalone_path()


    # > reference and floating images
    fref = str(fref)
    fflo = str(fflo)

    # > other image
    if foth is None:
        foth = ''
    else:
        foth = str(foth)

    # > change the parameters to strings for MATLAB scripting
    sep_str = ' '.join([str(s) for s in sep])
    tol_str = ' '.join([str(s) for s in tol])
    fwhm_str= ' '.join([str(s) for s in fwhm])

    # > form full set of commands for SPM12 coregistration
    coreg_batch_txt = "spm('defaults', 'PET');\nspm_jobman('initcfg');\n\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.ref = {{'{fref},1'}};\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.source = {{'{fflo},1'}};\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.other = {{'{foth},1'}};\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.eoptions.cost_fun = '{cost_fun}';\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.eoptions.sep = [{sep_str}];\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.eoptions.tol = [{tol_str}];\n"
    coreg_batch_txt += f"matlabbatch{{1}}.spm.spatial.coreg.estimate.eoptions.fwhm = [{fwhm_str}];\n\n"
    coreg_batch_txt += "spm_jobman('run', matlabbatch);"

    fcoreg = fspm.parent.parent/'spm_coreg_runtime.m'

    with open(fcoreg, 'w') as f:
        f.write(coreg_batch_txt)

    try:
        print('Running MATLAB Runtime SPM12 Coregistration...')
        subprocess.run([fspm, 'batch', fcoreg], check=True)
        print("SPM12 coregistration started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Coregistration error: {e}")


    return fcoreg
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+



#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# > Segmentation
def standalone_seg(
    fmri,
    outpath=None,
    nat_gm=True,
    nat_wm=True,
    nat_csf=True,
    nat_bn=False,
    biasreg=0.001,
    biasfwhm=60,
    mrf_cleanup=1,
    cleanup=1,
    regulariser=[0, 0.001, 0.5, 0.05, 0.2],
    affinereg='mni',
    fwhm=0,
    sampling=3,
    store_fwd=True,
    store_inv=True):

    ''' Segment MRI NIfTI image using standalone SPM12 with normalisation.
        Arguments:
        fmri:   input T1w MRI image.
        nat_{gm,wm,csf,bn}: output native space grey matter, white matter
                            CSF or bone segmentations.
        store_{fwd,inv}:    store forward and/or inverse deformation
                            fields definitions.
    '''


    if not spm12.check_standalone():
        log.error('MATLAB Runtime or standalone SPM12 has not been correctly installed.\nAttempting installation... ')
        response = input('Do you want to install MATLAB Runtime? [y/n]')
        if response in ['y', 'Y', 'yes']:
            spm12.install_standalone()
    else:
        fspm = spm12.standalone_path()


    # > the path to the input T1w MRI image
    fmri  = Path(fmri)
    f_mri = str(fmri)

    # > path to the TPM.nii internal file
    tpm_pth = str(fspm.parent/'spm12_mcr'/'spm12'/'spm12'/'tpm'/'TPM.nii')

    # > regulariser string
    rglrsr_str= ' '.join([str(s) for s in regulariser])

    # > writing deformations (forward and inverse)
    wrt_dfrm = [int(store_fwd), int(store_inv)]
    wrtdfrm_str= ' '.join([str(s) for s in wrt_dfrm])

    nat1 = '[{} 0]'.format(int(nat_gm))
    nat2 = '[{} 0]'.format(int(nat_wm))
    nat3 = '[{} 0]'.format(int(nat_csf))
    nat4 = '[{} 0]'.format(int(nat_bn))

    seg_batch_txt = "spm('defaults', 'PET');\nspm_jobman('initcfg');\n\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{f_mri},1'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = {biasreg};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = {biasfwhm};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(1).tpm = {{'{tpm_pth},1'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(1).ngaus = 1;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(1).native = {nat1};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(1).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(2).tpm = {{'{tpm_pth},2'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(2).ngaus = 1;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(2).native = {nat2};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(2).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(3).tpm = {{'{tpm_pth},3'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(3).ngaus = 2;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(3).native = {nat3};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(3).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(4).tpm = {{'{tpm_pth},4'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(4).ngaus = 3;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(4).native = {nat4};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(4).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(5).tpm = {{'{tpm_pth},5'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(5).ngaus = 4;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(5).native = [0 0];\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(5).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(6).tpm = {{'{tpm_pth},6'}};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(6).ngaus = 2;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(6).native = [0 0];\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.tissue(6).warped = [0 0];\n"

    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = {mrf_cleanup};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = {cleanup};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [{rglrsr_str}];\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = '{affinereg}';\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = {fwhm};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.samp = {sampling};\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.write = [{wrtdfrm_str}];\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.vox = NaN;\n"
    seg_batch_txt += f"matlabbatch{{1}}.spm.spatial.preproc.warp.bb = [NaN NaN NaN;NaN NaN NaN];\n"

    seg_batch_txt += "spm_jobman('run', matlabbatch);"

    fseg = fspm.parent.parent/'spm_seg_runtime.m'

    with open(fseg, 'w') as f:
        f.write(seg_batch_txt)


    # > attempt SPM12 segmentation
    try:
        print('Running MATLAB Runtime SPM12 segmentation...')
        subprocess.run([fspm, 'batch', fseg], check=True)
        print("SPM12 segmentation started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Segmentation error: {e}")


    # > output path
    if not outpath is None:
        opth = Path(outpath)
    else:
        opth = fmri.parent
    nimpa.create_dir(opth)


    fmri_fldr = fmri.parent

    outdct = {}
    for f in fmri_fldr.iterdir():

        if f.name[:2]=='c1':
            outdct['c1'] = opth/f.name
        elif f.name[:2]=='c2':
            outdct['c2'] = opth/f.name
        elif f.name[:2]=='c3':
            outdct['c3'] = opth/f.name
        elif f.name[:2]=='c4':
            outdct['c4'] = opth/f.name
        elif f.name[-8:]=='seg8.mat':
            outdct['param'] = opth/f.name
        elif f.name[:2]=='y_':
            outdct['fordef'] = opth/f.name
        elif f.name[:3]=='iy_':
            outdct['invdef'] = opth/f.name
        else:
            continue
        
        os.replace(f, opth/f.name)

    outdct['fbatch'] = fseg

    return outdct
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+



#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def standalone_normw(
    fdef,
    list4norm,
    bbox=None,
    voxsz=[2,2,2],
    intrp=4,
    prfx='w',
    outpath=None):

    ''' Write out normalised NIfTI images using definitions `fdef`.
    '''

    if not spm12.check_standalone():
        log.error('MATLAB Runtime or standalone SPM12 has not been correctly installed.\nAttempting installation... ')
        response = input('Do you want to install MATLAB Runtime? [y/n]')
        if response in ['y', 'Y', 'yes']:
            spm12.install_standalone()
    else:
        fspm = spm12.standalone_path()

    # > the path to the input T1w MRI image
    fdef = str(fdef)

    if isinstance(list4norm, (PurePath, str)):
        list4norm = [str(list4norm)]
    elif isinstance(list4norm, list):
        # > ensure paths are in strings
        list4norm = [str(s) for s in list4norm]

    # > form list of flies to be normalised
    lst2nrm = '\n'.join([f"'{s},1'" for s in list4norm])


    if isinstance(voxsz, (int, float)):
        voxsz = [voxsz, voxsz, voxsz]
    voxstr = ' '.join([str(v) for v in voxsz])

    if bbox is None:
        bbxstr = "NaN NaN NaN; NaN NaN NaN"
    elif isinstance(bbox, np.ndarray) and bbox.shape == (2, 3):
        bbxstr = ''
        for i in range(len(bbox)):
            bbxstr += ' '.join([str(e) for e in bbox[i]])
            bbxstr += '; '

    elif isinstance(bbox, list) and len(bbox) == 2:
        bbxstr = ''
        for i in range(len(bbox)):
            bbxstr += ' '.join([str(e) for e in bbox[i]])
            bbxstr += '; '
    else:
        raise ValueError("unrecognised format for bounding box")

    wnrm_batch_txt = "spm('defaults', 'PET');\nspm_jobman('initcfg');\n\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.subj.def = {{'{fdef}'}};\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.subj.resample = {{{lst2nrm}}};\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.woptions.bb = [{bbxstr}];\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.woptions.vox = [{voxstr}];\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.woptions.interp = {intrp};\n"
    wnrm_batch_txt += f"matlabbatch{{1}}.spm.spatial.normalise.write.woptions.prefix = '{prfx}';\n"

    wnrm_batch_txt += "spm_jobman('run', matlabbatch);"

    fwnrm = fspm.parent.parent/'spm_writenorm_runtime.m'

    with open(fwnrm, 'w') as f:
        f.write(wnrm_batch_txt)


    # > attempt spm12 segmentation
    try:
        print('running MATLAB runtime SPM12 normalisation writing...')
        subprocess.run([fspm, 'batch', fwnrm], check=True)
        print("SPM12 write norm started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"write normalisation error: {e}")

    # > output path
    if not outpath is None:
        opth = Path(outpath)
    else:
        opth = Path(list4norm[0]).parent

    nimpa.create_dir(opth)

    fwnrm_out = []
    for f in list4norm:
        f = Path(f)
        fout = opth/(prfx+f.name)
        os.replace(f.parent/(prfx+f.name), fout)
        fwnrm_out.append(fout)

    return fwnrm_out
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
