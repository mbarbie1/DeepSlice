''' Checking the format of the MNIST dataset.
'''
from __future__ import print_function

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


def loadData( ratioTrain, binning ):
    # server
    imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
    roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
    imageFormat = "png"

    regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
    dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data/features_labels_2d_" + str(binning)

    # Load pre-generated data if it exists else generate and save it
    contains = ""#"01"
    reduceDimension = False

    features, labels = md.lazyGenerateSmall( imageFolder, roisFolder, dataFolder, imageFormat, contains, binning, regionList, reduceDimension )
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
    y_train = train_labels
    y_test = test_labels
    id_test = list( range( nTrain,(nSamples-1) ) )
    id_train = list( range( 0,(nTrain-1) ) )

    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
#    if K.image_data_format() == 'channels_first':
#        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#        input_shape = (1, img_rows, img_cols)
#    else:
#        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#        input_shape = (img_rows, img_cols, 1)
    
    x_train = normalize(x_train)
    x_train = ( x_train * 255.).astype(np.uint8)
    x_test = normalize(x_test)
    x_test = (x_test * 255.).astype(np.uint8)


    for region in regionList:
        y_train[region] = ( y_train[region] ).astype(np.uint8)
        y_test[region] = ( y_test[region] ).astype(np.uint8)
    
    return x_train, x_test, y_train, y_test, id_test, id_train



def loadDataParams( flag ):
    # server
    imageFolder = flag.original_image_dir
    roisFolder = flag.original_rois_dir
    imageFormat = flag.image_format
    maskFolder = flag.image_rest_region_dir

    regionList = flag.region_list
    dataFolder = flag.original_data_dir
    ratioTrain = flag.ratio_train
    binning = flag.binning
    

    # Load pre-generated data if it exists else generate and save it
    contains = ""#"01"
    reduceDimension = False

    features, labels = md.lazyGenerateSmall( imageFolder, roisFolder, dataFolder, imageFormat, contains, binning, regionList, reduceDimension )
    labels_rest = md.lazyGenerateRestRegion( imageFolder, imageFormat, contains, maskFolder, labels, dataFolder, binning, regionList )
    for region in regionList:
        labels_rest[labels[region]>0] = 0
    labels["rest"] = labels_rest
    #regionList.append("rest")
    flag.region_list.append("rest")
    nSamples = features.shape[0]
    nX = features.shape[1]
    nY = features.shape[2]
    nTrain = round(ratioTrain * nSamples)
    x_train = features[0:(nTrain-1)]
    x_test = features[nTrain:(nSamples-1)]
    train_labels = {}
    test_labels = {}
    for region in flag.region_list:
        train_labels[region] = labels[region][0:(nTrain-1)]
        test_labels[region] = labels[region][nTrain:(nSamples-1)]
    y_train = train_labels
    y_test = test_labels
    id_test = list( range( nTrain,(nSamples-1) ) )
    id_train = list( range( 0,(nTrain-1) ) )

    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
#    if K.image_data_format() == 'channels_first':
#        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#        input_shape = (1, img_rows, img_cols)
#    else:
#        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#        input_shape = (img_rows, img_cols, 1)
    
    x_train = normalize(x_train)
    x_train = ( x_train * 255.).astype(np.uint8)
    x_test = normalize(x_test)
    x_test = (x_test * 255.).astype(np.uint8)


    for region in flag.region_list:
        y_train[region] = ( y_train[region] ).astype(np.uint8)
        y_test[region] = ( y_test[region] ).astype(np.uint8)
    
    return x_train, x_test, y_train, y_test, id_test, id_train
