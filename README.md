# hippodeep
Brain Hippocampus Segmentation

This program can segment quickly the Hippocampus of a brain T1 image, using Deep Convolutional NN.

![screenshot](blink.gif?raw=True)

A more complete description of the process is coming soon.

##Requirement
The program requires an NVIDIA card with CUDA capabilities >= 3.0 (common on card sold since 2012)
This code uses NVIDIA's cudnn >5.0 library through Theano/Lasagne.
The FSL tools are also required
Optionally, the ANTs warping tools can be used if installed
Tested on Linux CentOS6

##Installation

This code uses the Scipy stack and a recent Theano. It also requires the nibabel library (for nifti loading) and the Lasagne library.

To setup a working CUDA+CuDNN environment, download from NVIDIA and follow installation instruction
To setup a working FSL environment, download from fMRIB.

The simplest way to install the rest from scratch is to use a Anaconda environment, then
* install scipy and Theano >=0.8.2
* nibabel is available on pip (`pip install nibabel`)
* Lasagne should be probably pulled from the github repo


##Usage:
To use the program, simply
Using only FSL tools `deepseg3.sh brain_t1.nii.gz`
or Using ANTs tools: `deepseg2.sh brain_t1.nii.gz`

The resulting segmentation should be stored as `brain_t1_mask_L.nii.gz` (or R for right)
The volumes values are stored in `brain_t1_hippo_vol_LR.txt`
