import nibabel
import numpy as np
import sys

subjname = sys.argv[1] #"example_brain_t1"

for fname in [ "{}_mask_L.nii.gz".format(subjname), "{}_mask_R.nii.gz".format(subjname) ]:
    img = nibabel.load( fname )
    d = img.get_data(caching="unchanged")
    d[d < 128] = 0

    outimg = nibabel.Nifti1Image( d.astype("uint8"), img.affine )
    outimg.to_filename( fname )

    vol = (d > 0).sum() * np.abs(np.linalg.det(img.affine))
    print("%4.6f" % vol)
