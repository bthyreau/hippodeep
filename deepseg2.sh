export FSLOUTPUTTYPE=NIFTI_GZ
scriptpath=$(dirname $0)
a=${1%%.nii.gz}

if [ $(basename $a) != $a ]; then echo "no path allowed in image filename"; exit; fi
if [ ! -f ${a}.nii.gz ]; then echo "input T1 image must be .nii.gz format"; exit; fi

which antsApplyTransforms
if [ $? -eq "1" ]; then echo "ANTs scripts not in path"; exit; fi

export THEANO_FLAGS="device=cpu"

echo "Testing python environment"
python $scriptpath/test_import.py
echo no check
if [ $? -eq "1" ]; then
        echo "Failure to load the python Theano/Lasagne environment"
	    echo "Theano >= 0.9 is required"
        echo "python $scriptpath/test_import.py"
        exit
fi

$scriptpath/prerun2.sh $a
python $scriptpath/applyseg_unique.py affcrop_${a}.nii.gz
$scriptpath/backrun2.sh $a

# Uncomment the following to compute the volume in template space (and scale accordingly)

# antsTransformInfo affine2_${a}0GenericAffine.mat affine_${a}0GenericAffine.mat | /bin/grep -A 3 "Matrix:" | grep "[0-9]" | python -c 'import numpy as np; X=np.loadtxt("/dev/stdin"); print(np.linalg.det(X[:3]) * np.linalg.det(X[3:]))' >  det_${a}0GenericAffine.txt
# python -c 'import sys, nibabel; print([(nibabel.load(x).get_data() >= 128).sum() * float(open(sys.argv[1]).read()) for x in sys.argv[2:]])' det_${a}0GenericAffine.txt affcrop_${a}_outseg_L.nii.gz affcrop_${a}_outseg_R.nii.gz > ${a}_hippo_vol_templatespace_LR.txt
# echo "Hippocampal Volume (template space)(Left, Right):"
# cat ${a}_hippo_vol_templatespace_LR.txt

/bin/rm affcrop_${a}_outseg_[LR].nii.gz affcrop_${a}.nii.gz
echo fslview affcrop_${a}.nii.gz affcrop_${a}_outseg_[LR].nii.gz
/bin/rm affine2_${a}0GenericAffine.mat affine_${a}0GenericAffine.mat
echo "Hippocampal Volume (Left, Right):"
cat ${a}_hippo_vol_LR.txt
echo "(values saved in ${a}_hippo_vol_LR.txt)"
echo "you can check the results with:"
echo fslview ${a}.nii.gz ${a}_mask_L.nii.gz ${a}_mask_R.nii.gz
