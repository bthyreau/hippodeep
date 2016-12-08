subjname=$1
###########################
# Post-script, ANTs version
###########################
antsApplyTransforms -i affcrop_${subjname}_outseg_L.nii.gz -r ${subjname}.nii.gz -o ${subjname}_mask_L.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear
FSLOUTPUTTYPE=NIFTI_GZ fslmaths ${subjname}_mask_L.nii.gz -thr 128 ${subjname}_mask_L.nii.gz -odt char
fslstats ${subjname}_mask_L.nii.gz -V | cut -d" " -f 2 > ${subjname}_hippo_vol_LR.txt

antsApplyTransforms -i affcrop_${subjname}_outseg_R.nii.gz -r ${subjname}.nii.gz -o ${subjname}_mask_R.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear
FSLOUTPUTTYPE=NIFTI_GZ fslmaths ${subjname}_mask_R.nii.gz -thr 128 ${subjname}_mask_R.nii.gz -odt char
fslstats ${subjname}_mask_R.nii.gz -V | cut -d" " -f 2 >> ${subjname}_hippo_vol_LR.txt
