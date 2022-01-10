function out = amypad_coreg_modify_affine(imflo, M)
    VF = strcat(imflo,',1');
    MM = zeros(4,4);
    MM(:,:) = spm_get_space(VF);
    spm_get_space(VF, M\MM(:,:));
    out = 0;
end
