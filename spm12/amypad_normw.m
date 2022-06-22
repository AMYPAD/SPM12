function out = amypad_normw(def_file, flist4norm, voxsz)
    job.subj.def = {def_file};
    job.subj.resample = flist4norm;
    job.woptions.bb = [NaN, NaN, NaN; NaN, NaN, NaN];
    job.woptions.vox = [voxsz, voxsz, voxsz];
    job.woptions.interp = 4;
    job.woptions.prefix = 'w';
    spm_run_norm(job);
    out=0;
end
