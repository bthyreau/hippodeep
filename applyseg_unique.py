from __future__ import print_function
import time
ct = time.time()


from lasagne.layers import get_output, InputLayer, DenseLayer, ReshapeLayer, NonlinearityLayer
from lasagne.nonlinearities import rectify, leaky_rectify
from lasagne.updates import nesterov_momentum, rmsprop, adamax

import sys, os, time
scriptpath = os.path.dirname(__file__)

import nibabel
import numpy as np
import theano
import theano.tensor as T

import lasagne

# Note that Conv3DLayer and dnn.Conv3DDNNLayer have opposite filter-fliping defaults
from lasagne.layers import Conv3DLayer, MaxPool3DLayer
from lasagne.layers import Upscale3DLayer
from lasagne.layers import *


from lasagne.layers import Layer
from lasagne.utils import as_tuple

import pickle
import theano.misc.pkl_utils

cachefile = os.path.dirname(os.path.realpath(__file__)) + "/model_hippo.pkl"

if not os.path.exists(cachefile):


    # This broadcast-enabled layer is required to apply a 3d-mask along the feature-dimension
    # From GH PR #633
    class ElemwiseMergeLayerBroadcast(MergeLayer):
        """
        This layer performs an elementwise merge of its input layers.
        It requires all input layers to have the same output shape.
        Parameters
        ----------
        incomings : Unless `cropping` is given, all shapes must be equal, except
            for dimensions that are undefined (``None``) or broadcastable (``1``).
        merge_function : callable
            the merge function to use. Should take two arguments and return the
            updated value. Some possible merge functions are ``theano.tensor``:
            ``mul``, ``add``, ``maximum`` and ``minimum``.
        cropping : None or [crop]
            Cropping for each input axis. Cropping is described in the docstring
            for :func:`autocrop`
        See Also
        --------
        ElemwiseSumLayer : Shortcut for sum layer.
        """

        def __init__(self, incomings, merge_function, cropping=None, **kwargs):
            super(ElemwiseMergeLayerBroadcast, self).__init__(incomings, **kwargs)
            self.merge_function = merge_function
            self.cropping = cropping
            self.broadcastable = None

        def get_output_shape_for(self, input_shapes):
            input_shapes = autocrop_array_shapes(input_shapes, self.cropping)

            input_dims = [len(shp) for shp in input_shapes]
            if not all(input_dim == input_dims[0] for input_dim in input_dims):
                raise ValueError('Input dimensions must be the same but were %s' %
                                 ", ".join(map(str, input_shapes)))

            def broadcasting(input_dim):
                # Identify dimensions that will be broadcasted.
                sorted_dim = sorted(input_dim,
                                    key=lambda x: x if x is not None else -1)
                if isinstance(sorted_dim[-1], int) and sorted_dim[-1] != 1 \
                        and all([d == 1 for d in sorted_dim[:-1]]):
                    size_after_broadcast = sorted_dim[-1]
                    broadcast = [True if d == 1 else None for d in input_dim]
                    return ((size_after_broadcast,)*len(input_dim), broadcast)
                else:
                    return (input_dim, [None]*len(input_dim))

            # if the dimension is broadcastable we replace 1's with the size
            # after broadcasting.
            input_dims, broadcastable = list(zip(
                *[broadcasting(input_dim)for input_dim in zip(*input_shapes)]))

            self.broadcastable = list(zip(*broadcastable))
            input_shapes = list(zip(*input_dims))

            # Infer the output shape by grabbing, for each axis, the first
            # input size that is not `None` (if there is any)
            output_shape = tuple(next((s for s in sizes if s is not None), None)
                                 for sizes in zip(*input_shapes))

            def match(shape1, shape2):
                return (len(shape1) == len(shape2) and
                        all(s1 is None or s2 is None or s1 == s2
                            for s1, s2 in zip(shape1, shape2)))

            # Check for compatibility with inferred output shape
            if not all(match(shape, output_shape) for shape in input_shapes):
                raise ValueError("Mismatch: not all input shapes are the same")
            return output_shape

        def get_output_for(self, inputs, **kwargs):
            inputs = autocrop(inputs, self.cropping)
            # modify broadcasting pattern.
            if self.broadcastable is not None:
                for n, broadcasting_dim in enumerate(self.broadcastable):
                    for dim, broadcasting in enumerate(broadcasting_dim):
                        if broadcasting:
                            inputs[n] = T.addbroadcast(inputs[n], dim)

            output = None
            for input in inputs:
                if output is not None:
                    output = self.merge_function(output, input)
                else:
                    output = input
            return output


    # Definition of the network
    conv_num_filters = 48
    l = InputLayer(shape = (None, 1, 48, 72, 64), name="input")
    l_input = l

    # # # #
    # encoding
    # # # #
    l = Conv3DLayer(l, flip_filters=False, num_filters = 16, filter_size = (1,1,3), pad = 'valid', name="conv")
    l = Conv3DLayer(l, flip_filters=False, num_filters = 16, filter_size = (1,3,1), pad = 'valid', name="conv")
    l_conv_0 = l = Conv3DLayer(l, flip_filters=False, num_filters = 16, filter_size = (3,1,1), pad = 'valid', name="conv")

    l = l_conv_f1 = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters, filter_size = 3, pad = 'valid', name="conv_f1")

    l = l_maxpool1 = MaxPool3DLayer(l, pool_size = 2, name ='maxpool1')
    l = BatchNormLayer(l, name="batchnorm")

    l = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters, filter_size = (3,3,3), pad = "same", name="conv")
    l = l_convout1 = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters, filter_size = (3, 3, 3), pad = 'same', name ='convout1', nonlinearity = None)
    l = ElemwiseSumLayer(incomings = [l_maxpool1, l_convout1], name="sum_1s")
    l = NonlinearityLayer(l, nonlinearity = rectify, name="relu")


    conv_num_filters2 = 48
    l = l_maxpool2 = MaxPool3DLayer(l, pool_size = 2, name = 'maxpool2')
    l_maxpool2_conv = l
    l = BatchNormLayer(l, name="batchnorm")
    l = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters2, filter_size = (3,3,3), pad = "same", name="conv")
    l = l_convout2 = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters2, filter_size = (3, 3, 3), pad = 'same', name ='convout2', nonlinearity = None)
    l = ElemwiseSumLayer(incomings = [l_maxpool2_conv, l_convout2], name="sum_2s")
    l = NonlinearityLayer(l, nonlinearity = rectify, name="relu")

    # # # #
    # segmentation
    # # # #
    l_middle = l
    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = Conv3DLayer(l_middle, flip_filters=False, num_filters = conv_num_filters, filter_size = 3, pad = "same", name="conv")
    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l = l_convout1 = Conv3DLayer(l, flip_filters=False, num_filters = conv_num_filters, filter_size = 3, pad = 1, name="conv")
    l = Upscale3DLayer(l, scale_factor = 2, name="upscale")
    l_upscale = l
    l_convout2 = Conv3DLayer(l_upscale, flip_filters=False, num_filters = 16, filter_size = 3, pad = 1, name="conv")

    # Original (before refinement) output
    l_output1 = Conv3DLayer(l_convout2, flip_filters=False, num_filters = 1, filter_size = 1, pad = 'same', name="conv_1x", nonlinearity =lasagne.nonlinearities.sigmoid )

    # # #
    # refinement
    # # #
    ## The next output is reusing masked original filters to temptatively improve the network
    l_blur = Conv3DLayer(l_output1, flip_filters=False, num_filters=1, filter_size=7, stride=1, pad='same', W=lasagne.init.Constant(1.), b=lasagne.init.Constant(-7*7*7.*.10), nonlinearity=lasagne.nonlinearities.sigmoid)
    # in the above, *10 is : threshold at 10% of the smoothed mask (the higher, the smaller the mask)
    for x in l_blur.params.values():
        x.remove("trainable") # never train this, this is for downsampling

    l_masked_f1 = ElemwiseMergeLayerBroadcast([l_blur, l_conv_f1], merge_function=T.mul)
    l_extract = Conv3DLayer(l_masked_f1, flip_filters=False, num_filters = 47, filter_size = 3, pad = 1, name="extractconv", nonlinearity=leaky_rectify)
    l_concat = l = ConcatLayer([l_output1, l_extract], axis=1)
    l_mix = Conv3DLayer(l_concat, flip_filters=False, num_filters = 16, filter_size = 3, pad = 1, name="mixconv", nonlinearity=rectify)
    l_output2 = Conv3DLayer(l_mix, flip_filters=False, num_filters = 1, filter_size = 1, pad = 'same', name="conv_1x", nonlinearity =lasagne.nonlinearities.sigmoid )

    # Final output
    network = l_output2

    l_out = ConcatLayer([l_output2, l_output1])

    with np.load(os.path.join(scriptpath, "modelparams.npz")) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    #print ("weights loaded on %s (init took %4.2f sec)" % (time.ctime(), time.time() - ct))

    print("Compiling")

    fn_get_output = theano.function([l_input.input_var], get_output(l_out, deterministic=True))

    try:
        print("Pickling")
        pickle.dump(fn_get_output, open(cachefile,"wb"))
    except:
        print("Pickling failed")
        pass
else:
    print("Loading from cache")
    fn_get_output = pickle.load(open(cachefile,"rb"))


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        print ("Running %s" % (fn))
        img = nibabel.load(fn)
        d = img.get_data().astype(np.float32)
        d -= d.mean()
        d /= d.std()
        # split Left and Right (flipping Right)
        d_in = np.vstack([d[None, None, 6: 54:+1,: ,2:-2 ], d[None, None,-7:-55:-1,: ,2:-2 ]])
        out= fn_get_output(d_in)

        if 1:
            output = np.zeros((107, 72, 68, 2), np.uint8)
            output[-7:-55:-1,: ,2:-2, 0 ][2:-2,2:-2,2:-2] = np.clip(out[1,0] * 256, 0, 255)#* maskL
            output[6: 54:+1,: ,2:-2, 1 ][2:-2,2:-2,2:-2] = np.clip(out[0,0] * 256, 0, 255) # * maskR
            outputfn = fn.replace(".nii.gz", "_outseg_L.nii.gz")
            nibabel.Nifti1Image(output[...,0], img.get_affine()).to_filename(outputfn)
            outputfn = fn.replace(".nii.gz", "_outseg_R.nii.gz")
            nibabel.Nifti1Image(output[...,1], img.get_affine()).to_filename(outputfn)

        if 0: # l_output1 (for debugging)
            output = np.zeros((107, 72, 68, 2), np.uint8)
            output[-7:-55:-1,: ,2:-2, 0 ][2:-2,2:-2,2:-2] = np.clip(out[1,1] * 256, 0, 255)#* maskL
            output[6: 54:+1,: ,2:-2, 1 ][2:-2,2:-2,2:-2] = np.clip(out[0,1] * 256, 0, 255) # * maskR
            outputfn = fn.replace(".nii.gz", "_outseg_output1_L.nii.gz")
            nibabel.Nifti1Image(output[...,0], img.get_affine()).to_filename(outputfn)
            outputfn = fn.replace(".nii.gz", "_outseg_output1_R.nii.gz")
            nibabel.Nifti1Image(output[...,1], img.get_affine()).to_filename(outputfn)

    print("Elapsed: %4.2fs" %  (time.time() - ct))
