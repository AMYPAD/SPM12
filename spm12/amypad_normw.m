function out = amypad_normw(def_file, flist4norm, voxsz, intrp, bbox)
    job.subj.def = {def_file};
    job.subj.resample = flist4norm;
    %job.woptions.bb = [NaN, NaN, NaN; NaN, NaN, NaN];
    job.woptions.bb = bbox;
    job.woptions.vox = [voxsz, voxsz, voxsz];
    job.woptions.interp = intrp;
    job.woptions.prefix = 'w';
    spm_run_norm(job);
    out=0;
end
