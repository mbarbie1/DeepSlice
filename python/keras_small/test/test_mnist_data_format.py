''' Checking the format of the MNIST dataset.
'''
from __future__ import print_function

""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import module_data as md

def normalize( x ):
    """
    Normalise data to [0, 1] range
    """
    x /= np.max(x) 
    return x

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(type(x_train))
print(x_train)
print(x_train.shape)


# server
imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
#imageFolder = "/home/mbarbier/Documents/data/test_ref_small/DAPI/reference_images"
#roisFolder = "/home/mbarbier/Documents/data/test_ref_small/DAPI/reference_rois"
imageFormat = "png"

# laptop platsmurf
#imageFolder = "/home/mbarbier/Documents/prog/SliceMap/dataset/input/reference_images"
#roisFolder = "/home/mbarbier/Documents/prog/SliceMap/dataset/input/reference_rois"
#imageFormat = "tif"

binning = 64
regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data/features_labels_2d"

# Load pre-generated data if it exists else generate and save it
contains = ""#"01"
reduceDimension = False

features, labels = md.lazyGenerateSmall( imageFolder, roisFolder, dataFolder, imageFormat, contains, binning, regionList, reduceDimension )
ratioTrain = 0.8
nSamples = features.shape[0]
nX = features.shape[1]
nY = features.shape[2]
nTrain = round(ratioTrain * nSamples)
x_train = features[0:(nTrain-1)]
x_test = features[nTrain:(nSamples-1)]
train_labels = {}
test_labels = {}
for region in regionList:
    train_labels[region] = labels[region][0:(nTrain-1)]
    test_labels[region] = labels[region][nTrain:(nSamples-1)]

print(K.image_data_format())

img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = normalize(x_train)
x_test = normalize(x_test)


