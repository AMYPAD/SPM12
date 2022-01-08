function [param,invdef,fordef] = amypad_seg(f_mri, spm_path, nat_gm, nat_wm, nat_csf, store_fwd, store_inv, visual)
    job.channel.vols = {strcat(f_mri,',1')};
    job.channel.biasreg = 0.001;
    job.channel.biasfwhm = 60;
    job.channel.write = [0, 0];
    job.tissue(1).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,1']};
    job.tissue(1).ngaus = 1;
    job.tissue(1).native = [nat_gm, 0];
    job.tissue(1).warped = [0, 0];
    job.tissue(2).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,2']};
    job.tissue(2).ngaus = 1;
    job.tissue(2).native = [nat_wm, 0];
    job.tissue(2).warped = [0, 0];
    job.tissue(3).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,3']};
    job.tissue(3).ngaus = 2;
    job.tissue(3).native = [nat_csf, 0];
    job.tissue(3).warped = [0, 0];
    job.tissue(4).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,4']};
    job.tissue(4).ngaus = 3;
    job.tissue(4).native = [0, 0];
    job.tissue(4).warped = [0, 0];
    job.tissue(5).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,5']};
    job.tissue(5).ngaus = 4;
    job.tissue(5).native = [0, 0];
    job.tissue(5).warped = [0, 0];
    job.tissue(6).tpm = {[spm_path, filesep, 'tpm', filesep, 'TPM.nii,6']};
    job.tissue(6).ngaus = 2;
    job.tissue(6).native = [0, 0];
    job.tissue(6).warped = [0, 0];
    job.warp.mrf = 1;
    job.warp.cleanup = 1;
    job.warp.reg = [0, 0.001, 0.5, 0.05, 0.2];
    job.warp.affreg = 'mni';
    job.warp.fwhm = 0;
    job.warp.samp = 3;
    job.warp.write = [store_fwd, store_inv];
    if visual>0
        Finter = spm_figure('GetWin','Interactive');
    end
    spm_jobman('initcfg');
    segout = spm_preproc_run(job);
    param  = segout.param{1};
    invdef = segout.invdef{1};
    fordef = segout.fordef{1};
    %disp(segout);
end
