export FSLOUTPUTTYPE=NIFTI_GZ

scriptpath=$(dirname $0)

a=$1
ba=$(basename $a)
a=$(basename $a .gz)
a=$(basename $a .nii)
a=$(basename $a .img)
a=$(basename $a .hdr)
a=$(basename $a .mgz)
a=$(basename $a .mgh)
pth=$(dirname $1)

#if [ $(basename $a) != $a ]; then echo "no path allowed in image filename"; exit; fi
#if [ ! -f ${a}.nii.gz ]; then echo "input T1 image must be .nii.gz format"; exit; fi

which antsApplyTransforms
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

cd $pth

export THEANO_FLAGS="device=cpu,floatX=float32"
$scriptpath/quickaffineonly.sh $ba

subjname=$a
if [ -f "aff_${subjname}0GenericAffine.mat" ]; then
	cp aff_${subjname}0GenericAffine.mat affine_${subjname}0GenericAffine.mat
else
	echo "Affine registration failed - aborting. Try deepseg2.sh or deepseg3.sh"
	exit;
fi

antsApplyTransforms -d 3 -i rescaled_${a}.nii -o affcrop_tmp_${subjname}.nii -r $scriptpath/pre_crop.nii.gz -t affine_${subjname}0GenericAffine.mat --float
#antsRegistrationSyNQuick.sh -f $scriptpath/pre_crop.nii.gz -m affcrop_tmp_${subjname}.nii -o affine2_${subjname} -n 1 -t a -j 1
antsRegistrationSyNQuick.sh -f $scriptpath/croproi_MNI152_1mm.nii.gz -m affcrop_tmp_${subjname}.nii -o affine2_${subjname} -n 4 -t a -j 1
antsApplyTransforms -d 3 -i rescaled_${a}.nii -o affcrop_${subjname}.nii.gz -r ${scriptpath}/croproi_MNI152_1mm.nii.gz -t affine2_${subjname}0GenericAffine.mat -t affine_${subjname}0GenericAffine.mat --float

rm r_T1_${subjname}.nii affine_${subjname}Warped.nii.gz affine_${subjname}InverseWarped.nii.gz affcrop_tmp_${subjname}.nii affine2_${subjname}Warped.nii.gz affine2_${subjname}InverseWarped.nii.gz
rm rescaled_${a}.nii


python $scriptpath/applyseg_unique.py affcrop_${a}.nii.gz

antsApplyTransforms -i affcrop_${subjname}_outseg_L.nii.gz -r $ba -o ${subjname}_mask_L.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear
antsApplyTransforms -i affcrop_${subjname}_outseg_R.nii.gz -r $ba -o ${subjname}_mask_R.nii.gz -t [affine_${subjname}0GenericAffine.mat,1] -t [affine2_${subjname}0GenericAffine.mat,1]  --float -n Linear
python $scriptpath/postproc_thresh.py ${subjname} > ${subjname}_hippo_vol_LR.txt

/bin/rm affcrop_${a}_outseg_[LR].nii.gz affcrop_${a}.nii.gz
#echo fslview affcrop_${a}.nii.gz affcrop_${a}_outseg_[LR].nii.gz
/bin/rm affine2_${a}0GenericAffine.mat affine_${a}0GenericAffine.mat
/bin/rm aff_${a}0GenericAffine.mat
echo
echo
echo "Hippocampal Volume (Left, Right):"
cat ${a}_hippo_vol_LR.txt
echo "(values saved in ${a}_hippo_vol_LR.txt)"
echo "Approximate intra-cranial-volume:"
cat ${a}_eTIV.txt
echo "(value saved in ${a}_eTIV.txt)"
echo "you can check the results with:"
#echo fslview $1 ${a}_mask_L.nii.gz ${a}_mask_R.nii.gz
echo fslview $pth/$ba $pth/${a}_mask_L.nii.gz $pth/${a}_mask_R.nii.gz
cd -
