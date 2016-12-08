scriptpath=$(dirname $0)
subjname=$1

#########################
# Pre-script, FSL version
#########################

#cat fslmat_mni2precrop.mat 1 0 0 -27 0 1 0 -58 0 0 1 -18 0 0 0 1
#cat fslmat_mni2crop.mat 1 0 0 -36 0 1 0 -67 0 0 1 -27 0 0 0 1
#cat fslmat_precrop2crop.mat 1 0 0 -8.79 0 1 0 -8.84 0 0 1 -8.86 0 0 0 1

echo "Downsampling (if ANTS available)"
# firt command is useful only if second command fail
gunzip -c  ${subjname}.nii.gz > r_T1_${subjname}.nii
ResampleImage 3 ${subjname}.nii.gz r_T1_${subjname}.nii 2x2x2 0 0

echo "FSL FLIRT pre-registrations"
flirt -in r_T1_${subjname}.nii -ref $scriptpath/MNI152_T1_2mm.nii.gz -minsampling 8 -dof 7 -bins 64 -searchrx -60 60 -searchry -60 60 -searchrz -60 60 -omat natif2mni_${subjname}.xfm
convert_xfm  -omat natif2precrop_${subjname}.xfm -concat $scriptpath/fslmat_mni2precrop.mat natif2mni_${subjname}.xfm
flirt -in ${subjname}.nii.gz -o affcrop_tmp_${subjname}.nii.gz -ref $scriptpath/pre_crop.nii -applyxfm -init natif2precrop_${subjname}.xfm

### Warp on pre_crop then resample on target (croproi)
flirt -in affcrop_tmp_${subjname}.nii.gz -ref $scriptpath/pre_crop.nii.gz -finesearch 2 -searchrx -5 5 -searchry -5 5 -searchrz -5 5 -dof 12 -bins 256 -omat precrop2precrop_${subjname}.xfm
convert_xfm -omat natif2precrop2_${subjname}.xfm -concat precrop2precrop_${subjname}.xfm natif2precrop_${subjname}.xfm
convert_xfm -omat natif2crop_${subjname}.xfm -concat  $scriptpath/fslmat_precrop2crop.mat natif2precrop2_${subjname}.xfm
flirt -in ${subjname}.nii.gz -o affcrop_${subjname}.nii.gz -ref $scriptpath/croproi_MNI152_1mm.nii.gz  -applyxfm -init natif2crop_${subjname}.xfm

convert_xfm -omat crop2natif_${subjname}.xfm -inverse natif2crop_${subjname}.xfm 

rm r_T1_${subjname}.nii affcrop_tmp_${subjname}.nii.gz
#rm natif2mni_${subjname}.xfm natif2precrop_${subjname}.xfm precrop2crop_${subjname}.xfm
rm natif2mni_${subjname}.xfm natif2precrop_${subjname}.xfm natif2precrop2_${subjname}.xfm precrop2precrop_${subjname}.xfm 
echo "done" 
