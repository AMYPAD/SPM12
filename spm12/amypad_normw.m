function out = amypad_normw(def_file, flist4norm)
    job.subj.def = {def_file};
    job.subj.resample = flist4norm;
    job.woptions.bb = [NaN, NaN, NaN; NaN, NaN, NaN];
    job.woptions.vox = [2, 2, 2];
    job.woptions.interp = 4;
    job.woptions.prefix = 'w';
    spm_run_norm(job);
    out=0;
end
