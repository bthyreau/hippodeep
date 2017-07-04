# hippodeep
Brain Hippocampus Segmentation

This program can quickly segment (<30s) the Hippocampus of a brain T1 image, using Deep Convolutional NN.

![screenshot](blink.gif?raw=True)

It relies on a Convolutional Neural Network trained on thousands of images from multiple large cohorts, and is therefore quite robust. (see also manuscript (*under review*))

## Requirement
*(new)* This program requires Theano >= 0.9.0 and Lasagne
No GPU is required, as CPU inference is now possible with low speed penalty.

Either the FSL tools or the ANTs tools are also required for the initial low-res registration. (Or both for improved robustness)

Tested on Linux CentOS 6.8 and 7.1

## Installation

The code uses the Scipy stack and a recent Theano. It also requires the nibabel library (for nifti loading) and the Lasagne library.

To setup a working FSL environment, download from fMRIB.
To setup a ANTs environment, get it from http://stnava.github.io/ANTs/ (or alternatively, from a docker container such as http://www.mindboggle.info/ )

The simplest way to install the rest from scratch is to use a Anaconda environment, then
* install scipy and Theano >=0.9.0 (`conda install theano`)
* nibabel is available on pip (`pip install nibabel`)
* Lasagne (version >=0.2 If not available, it should be probably pulled from the github repo `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`)


## Usage:
To use the program, simply

Using ANTs tools: `deepseg2.sh example_brain_t1.nii.gz`.

or, if you want to use the pre-registration from FSL instead:

Using FSL tools `deepseg3.sh example_brain_t1.nii.gz`,

The script can be called with its full path, but the image file must be in current, writable, directory.

The resulting segmentation should be stored as `example_brain_t1_mask_L.nii.gz` (or R for right).

The volumes values are stored in `example_brain_t1_hippo_vol_LR.txt`
