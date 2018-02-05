# DeepSlice

Trial project to use deep learning for fluorescent brain slice region segmentation.

Word of warning for those who bump into this repo: Since this is a test project, documentation of both the code and the project itself is limited, or uhm, let's say "non-existant" :-).
Moreover, you might notice that the code is mainly a bunch of Keras and Tensorflow tutorial code snippets, cherry picked and adapted to fit the workflow of our interest.  

## Main goal of the project

DeepSlice is not meant to achieve brain region segmentation with a higher accuracy than manual annotation, but its focus is rather on being robust under slice distortions, illumination artifacts, etc.   

## Current status

There is a running pipeline on a small data set.

Following steps/improvements should be still undertaken:

- Debugging the workflow which uses data augmentation, getting faulty results. Solution is probably in generating less augmented data, e.g. only mirroring, or train on a larger subset of all the augmented data.
- Implementing the alpha-SMD metric as a custom loss function.
- Adding layers to the network depending on the resolution of image data provided.
- Adding a second patch-based workflow and combine both workflows.
