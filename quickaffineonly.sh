#!/bin/bash
#
# This script computes an _approximate_ registration of the image to MNI space
#
# It subsamples& extract the gray-matter using a convnet, then
# use that for affine registration using ANTS
#
scriptpath=$(dirname $0)
if [ ! -f "$1" ]; then echo input file not found $1; exit; fi

# try to drop a few differents suffix names
a=$1
a=$(basename $a .gz)
a=$(basename $a .nii)
a=$(basename $a .img)
a=$(basename $a .hdr)
a=$(basename $a .mgz)
a=$(basename $a .mgh)
pth=$(dirname $1)

which antsApplyTransforms > /dev/null
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

cd $pth

#ImageMath 3 rescaled_${a}.nii RescaleImage ${1} 0 1000
# Rescaling (mostly useful for some old misconverted AOBA data)
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
N4BiasFieldCorrection -d 3 -i ${1} -o rescaled_${a}.nii -s 4 -c [5x5x4x4x4x4]
ImageMath 3 rescaled_${a}.nii RescaleImage rescaled_${a}.nii 0 1000


echo "head priors computation"

ResampleImage 3 rescaled_${a}.nii res64_rescaled_${a}.nii 64x64x64 1 0
THEANO_FLAGS="device=cpu" python $scriptpath/model_apply_6tissues_mini.py res64_rescaled_${a}.nii | grep -A 1 volume | tail -1 > ${a}_eTIV.txt

#antsRegistrationSyNQuick.sh -m res64_rescaled_${a}_tissues0.nii.gz -f ${scriptpath}/atlas/b64_TPM_mask.nii.gz  -m res64_rescaled_${a}_tissues1.nii.gz -f ${scriptpath}/atlas/b64_TPM_c1.nii.gz -t a -o aff_${a} -n 1
antsRegistrationSyNQuick.sh -m res64_rescaled_${a}_tissues1.nii.gz -f ${scriptpath}/atlas/b64_TPM_c1.nii.gz -t a -o aff_${a} -n 1

echo $PWD
echo "ANTS affine matrix saved as aff_${a}0GenericAffine.mat"
##echo "for a quick view of the results, try:"
##echo " fslview aff_${a}Warped.nii.gz"

rm res64_rescaled_${a}.nii

rm aff_${a}InverseWarped.nii.gz

rm aff_${a}Warped.nii.gz

##rm aff_${a}0GenericAffine.mat

#rm rescaled_${a}.nii
rm res64_rescaled_${a}_tissues0.nii.gz
rm res64_rescaled_${a}_tissues1.nii.gz
cd -
