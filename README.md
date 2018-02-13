# DeepSlice

Trial project to use deep learning for fluorescent brain slice region segmentation. DeepSlice is not meant to achieve brain region segmentation with a higher accuracy than manual annotation, but its focus is rather on being robust under slice distortions, illumination artifacts, etc.

![image](https://github.com/mbarbie1/DeepSlice/blob/master/python/keras_small/data/example_prediction/img_0.png)
![prediction](https://github.com/mbarbie1/DeepSlice/blob/master/python/keras_small/data/example_prediction/pred_0.png)
![manual segmentation](https://github.com/mbarbie1/DeepSlice/blob/master/python/keras_small/data/example_prediction/manual_0.png)
![difference](https://github.com/mbarbie1/DeepSlice/blob/master/python/keras_small/data/example_prediction/difference_0.png)

*Figure: from left to right: Sample, prediction, manual segmentation, difference image (blue is wrongly labeled)*

*Word of warning* for those who bump into this repo: Since this is a test project, documentation of both the code and the project itself is limited, or uhm, let's say "non-existant" :-).
Moreover, you might notice that the code is mainly a bunch of Keras and Tensorflow tutorial code snippets, cherry picked and adapted to fit the workflow of our interest.

## Installation

The current working version of DeepSlice is implemented in Python. 
The current workflow uses heavily downscaled/downgraded images of brain slices as input and output.
The source code in the folder DeepSlice/python/keras_small contains all python code, 
and the folder DeepSlice/python/keras_small/data contains the input data of almost 60 annotated slices at different scales 128, 256, and 512 times binned (as Numpy data).
To have your own data you can look at the folder DeepSlice/python/keras_small/data/example_raw_data where you can find a raw image (already downscaled) with a corresponding zip-file containing ImageJ ROIs.

Main dependencies (these might also have there own dependencies):
- numpy
- matplotlib
- scikit-image
- rasterio
- tensorflow
- keras

## Running

To run the DeepSlice training, run the main function which can be found in *DeepSlice/python/keras_small/main_training.py*
Parameters can be found in this main file and should be adapted to ones own need.
To run DeepSlice for prediction, run the main function which can be found in *DeepSlice/python/keras_small/main_prediction.py*

## Current status

There is a running pipeline on a small data set.

Following steps/improvements should be still undertaken:

- Debugging the workflow which uses data augmentation, getting faulty results. Solution is probably in generating less augmented data, e.g. only mirroring, or train on a larger subset of all the augmented data.
- Implementing the alpha-SMD metric as a custom loss function.
- Adding layers to the network depending on the resolution of image data provided.
- Adding a second patch-based workflow and combine both workflows.

