export FSLOUTPUTTYPE=NIFTI_GZ
scriptpath=$(dirname $0)
a=${1%%.nii.gz}

if [ $(basename $a) != $a ]; then echo "no path allowed in image filename"; exit; fi
if [ ! -f ${a}.nii.gz ]; then echo "input T1 image must be .nii.gz format"; exit; fi

which fslmaths
if [ $? -eq "1" ]; then echo "fsl executable (e.g. fslmaths) not in path"; exit; fi

export THEANO_FLAGS="device=cpu"

echo "Testing python environment"
python $scriptpath/test_import.py
if [ $? -eq "1" ]; then
        echo "Failure to load the python Theano/Lasagne environment"
	    echo "Theano >= 0.9 is required"
        echo "python $scriptpath/test_import.py"
        exit
fi


$scriptpath/prerun3.sh $a
python $scriptpath/applyseg_unique.py affcrop_${a}.nii.gz
$scriptpath/backrun3.sh $a

# Uncomment the following to compute the volume in template space (and scale accordingly)

# python -c "import sys, nibabel, numpy as np; print([(nibabel.load(x).get_data() >= 128).sum() * np.linalg.det(np.loadtxt(sys.argv[1])) for x in sys.argv[2:]])" crop2natif_${a}.xfm affcrop_${a}_outseg_L.nii.gz affcrop_${a}_outseg_R.nii.gz > ${a}_hippo_vol_templatespace_LR.txt
# echo "Hippocampal Volume (template space)(Left, Right):"
# cat ${a}_hippo_vol_templatespace_LR.txt

/bin/rm affcrop_${a}_outseg_[LR].nii.gz affcrop_${a}.nii.gz
/bin/rm natif2crop_${a}.xfm crop2natif_${a}.xfm
echo "Hippocampal Volume (Left, Right):"
cat ${a}_hippo_vol_LR.txt

echo fslview ${a}.nii.gz ${a}_mask_L.nii.gz ${a}_mask_R.nii.gz
