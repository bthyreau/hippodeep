from __future__ import print_function

from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer, NonlinearityLayer
from lasagne.nonlinearities import rectify, leaky_rectify, elu
import sys, os, time

import nibabel
import numpy as np
import scipy.ndimage
import theano
import theano.tensor as T

import lasagne

# Note that Conv3DLayer and .Conv3DLayer have opposite filter-fliping defaults
from lasagne.layers import Conv3DLayer, MaxPool3DLayer
from lasagne.layers import Upscale3DLayer

from lasagne.layers import *

import pickle
import theano.misc.pkl_utils 

cachefile = os.path.dirname(os.path.realpath(__file__)) + "/model6tissues.pkl"

if not os.path.exists(cachefile):

    l = InputLayer(shape = (None, 1, 64, 64, 64), name="input")
    l_input = l

    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = batch_norm(l)
    li0 = l

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 24, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = Conv3DLayer(l, num_filters = 32, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = batch_norm(l)
    li1 = l

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 48, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = Conv3DLayer(l, num_filters = 64, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
    l = batch_norm(l)
    li2 = l

    l = MaxPool3DLayer(l, pool_size = 2, name ='maxpool')
    l = Conv3DLayer(l, num_filters = 96, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = Conv3DLayer(l, num_filters = 64, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 32, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = ConcatLayer([l, li2])
    l = Conv3DLayer(l, num_filters = 32, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = batch_norm(l)
    l = ConcatLayer([l, li1])
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name ='conv', nonlinearity=elu)
    l = batch_norm(l)

    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)
    l = ConcatLayer([l, li0])
    l = Conv3DLayer(l, num_filters = 16, filter_size = (3,3,3), pad = 'same', name="conv", nonlinearity=elu)

    l = Conv3DLayer(l, num_filters = 7, filter_size = 1, pad = "same", name="conv1x", nonlinearity = lasagne.nonlinearities.sigmoid )
    lastl = l
    network = l

    def reload_fn(fn):
        with np.load(fn) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(lastl, param_values)

    reload_fn(os.path.dirname(os.path.realpath(__file__)) + "/params_01560_00000.npz")

    print("Compiling")

    input_var = l_input.input_var
    prediction = lasagne.layers.get_output(lastl)
    getout = theano.function([input_var], prediction)
    print("Pickling")
    if 1:
        pickle.dump(getout, open(cachefile,"wb"))
else:
    print("Loading from cache")
    getout = pickle.load(open(cachefile,"rb"))


if __name__ == "__main__":
    fname = sys.argv[1]
    outfilename = sys.argv[1].replace(".nii.gz", ".nii").replace(".nii", "_tiv.nii.gz")
    img = nibabel.load(fname)

    d = img.get_data().astype(np.float32)
    d = (d - d.mean()) / d.std()
    
    o1 = nibabel.orientations.io_orientation(img.affine)
    o2 = np.array([[ 0., -1.], [ 1.,  1.], [ 2.,  1.]])
    trn = nibabel.orientations.ornt_transform(o1, o2)
    d_orr = nibabel.orientations.apply_orientation(d, trn)

    print("Starting inference")    
    T = time.time()
    out1 = getout(d_orr[None,None])
    output = out1[0,0].astype("float32")
    print("Inferrence in " + str(time.time() - T))

    out_cc, lab = scipy.ndimage.label(output > .01)
    output *= (out_cc == np.bincount(out_cc.flat)[1:].argmax()+1)

    vol = (output[output > .5]).sum() * np.abs(np.linalg.det(img.affine))
    print("Estimated intra-cranial volume (mm^3):\n%4.2f" % vol)

    trn = nibabel.orientations.ornt_transform(o2, o1)
    for i in range(2):
        output = out1[0,i].astype("float32")
        out = nibabel.orientations.apply_orientation(output, trn)
        nibabel.Nifti1Image(out, img.affine, img.header).to_filename(outfilename.replace("_tiv", "_tissues%d" % i))
