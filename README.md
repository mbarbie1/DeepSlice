# DeepSlice

Trial project to use deep learning for fluorescent brain slice region segmentation. DeepSlice is not meant to achieve brain region segmentation with a higher accuracy than manual annotation, but its focus is rather on being robust under slice distortions, illumination artifacts, etc.

*Word of warning* for those who bump into this repo: Since this is a test project, documentation of both the code and the project itself is limited, or uhm, let's say "non-existant" :-).
Moreover, you might notice that the code is mainly a bunch of Keras and Tensorflow tutorial code snippets, cherry picked and adapted to fit the workflow of our interest.

## Installation

The current working version of DeepSlice is implemented in Python. 
The current workflow uses heavily downscaled/downgraded images of brain slices as input and output.
The source code in the folder DeepSlice/python/keras_small contains all python code, 
and the folder DeepSlice/python/keras_small/data contains the input data of almost 60 annotated slices at different scales 128, 256, and 512 times binned (as Numpy data).

Main dependencies (these might also have there own dependencies):
- numpy
- matplotlib
- scikit-image
- rasterio
- tensorflow
- keras

## Running

To run DeepSlice, run the main function which can be found in *DeepSlice/python/keras_small/main_training.py*
Parameters can be found in this main file and should be adapted to ones own need.

## Current status

There is a running pipeline on a small data set.

Following steps/improvements should be still undertaken:

- Debugging the workflow which uses data augmentation, getting faulty results. Solution is probably in generating less augmented data, e.g. only mirroring, or train on a larger subset of all the augmented data.
- Implementing the alpha-SMD metric as a custom loss function.
- Adding layers to the network depending on the resolution of image data provided.
- Adding a second patch-based workflow and combine both workflows.

