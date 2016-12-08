scriptpath=$(dirname $0)
subjname=$1

##########################
# Pre-script, ANTs version
##########################
ResampleImage 3 ${subjname}.nii.gz r_T1_${subjname}.nii 4x4x4 0 0
# it looks like adding more threads to ANTS seems to make results more robust (?? more descent paths ?)
antsRegistrationSyNQuick.sh -f $scriptpath/MNI152_4mm.nii.gz -m r_T1_${subjname}.nii -o affine_${subjname} -n 4 -t a
antsApplyTransforms -d 3 -i ${subjname}.nii.gz -o affcrop_tmp_${subjname}.nii -r $scriptpath/pre_crop.nii.gz -t affine_${subjname}0GenericAffine.mat --float
#antsRegistrationSyNQuick.sh -f $scriptpath/pre_crop.nii.gz -m affcrop_tmp_${subjname}.nii -o affine2_${subjname} -n 1 -t a -j 1
antsRegistrationSyNQuick.sh -f $scriptpath/croproi_MNI152_1mm.nii.gz -m affcrop_tmp_${subjname}.nii -o affine2_${subjname} -n 4 -t a -j 1
antsApplyTransforms -d 3 -i ${subjname}.nii.gz -o affcrop_${subjname}.nii.gz -r ${scriptpath}/croproi_MNI152_1mm.nii.gz -t affine2_${subjname}0GenericAffine.mat -t affine_${subjname}0GenericAffine.mat --float

rm r_T1_${subjname}.nii affine_${subjname}Warped.nii.gz affine_${subjname}InverseWarped.nii.gz affcrop_tmp_${subjname}.nii affine2_${subjname}Warped.nii.gz affine2_${subjname}InverseWarped.nii.gz

