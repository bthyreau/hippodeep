subjname=$1
###########################
# Post-script, ANTs version
###########################
antsApplyTransforms -i affcrop_${subjname}_outseg_L.nii.gz -r ${subjname}.nii.gz -o ${subjname}_mask_L.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear

antsApplyTransforms -i affcrop_${subjname}_outseg_R.nii.gz -r ${subjname}.nii.gz -o ${subjname}_mask_R.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear

scriptpath=$(dirname $0)
python $scriptpath/postproc_thresh.py ${subjname} > ${subjname}_hippo_vol_LR.txt
