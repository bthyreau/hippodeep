subjname=$1
##########################
# Post-script, FSL version
##########################
flirt -in affcrop_${subjname}_outseg_L.nii.gz -ref ${subjname}.nii.gz -o ${subjname}_mask_L.nii.gz  -applyxfm -init crop2natif_${subjname}.xfm -interp trilinear
FSLOUTPUTTYPE=NIFTI_GZ fslmaths ${subjname}_mask_L.nii.gz -thr 128 ${subjname}_mask_L.nii.gz -odt char
fslstats ${subjname}_mask_L.nii.gz -V | cut -d" " -f 2 > ${subjname}_hippo_vol_LR.txt

flirt -in affcrop_${subjname}_outseg_R.nii.gz -ref ${subjname}.nii.gz -o ${subjname}_mask_R.nii.gz  -applyxfm -init crop2natif_${subjname}.xfm -interp trilinear
FSLOUTPUTTYPE=NIFTI_GZ fslmaths ${subjname}_mask_R.nii.gz -thr 128 ${subjname}_mask_R.nii.gz -odt char
fslstats ${subjname}_mask_R.nii.gz -V | cut -d" " -f 2 >> ${subjname}_hippo_vol_LR.txt

