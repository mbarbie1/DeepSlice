# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:05:56 2018

@author: mbarbier
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
from module_data import loadImage
import argparse
from module_metrics_weight import weighted_categorical_crossentropy
#from module_train import train
from module_utilities import writeDictToCsv
import time
from keras import losses
import numpy as np

from keras import backend as K
#from module_model_unet import unet
from module_model import getModel, preprocess
from module_utilities import writeData
#from module_model_segnet import segnet
from module_callbacks import trainCheck
import matplotlib.pyplot as plt
import os
from module_metrics import getMetricFunctions
from keras.utils.np_utils import to_categorical
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json, load_model

def loadNeubias():
    # Load raw data
    imageFolder = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/neubias"
    imageNames = ["img1.tif"]
    probNames = ["prob1.tif"]
    refNames = ['ref1.tif']
    
    
    for imageName in imageNames:
        imagePath = os.path.join( imageFolder, imageName )
        img_ori = loadImage( imagePath )
        img = np.transpose(img_ori, (0, 2, 3, 1) )
        img = img[ :,:,:,0 ]
        
    
    for probName in probNames:
        imagePath = os.path.join( imageFolder, imageName )
        #img = np.transpose(img, (0, 3, 1, 2) )
        prob_ori = loadImage( imagePath )
        prob = np.transpose(prob_ori, (0, 2, 3, 1) )
        
 
    return img, prob
    
def probToMask(prob):
    mask = np.zeros( prob.shape, dtype='uint8' )
    mask[prob > 128] = 1
    #mask = mask[:,0,:,:]
    return mask    





def train(image_size, epochs ):
    
    output_masks_feed_dir = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/neubias"
    
    output_images_feed_dir = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/neubias"
    img_rows = image_size
    img_cols = image_size
    n_epochs = epochs

    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    img, prob = loadNeubias()
    mask = probToMask(prob)

    images = preprocess(img, img_rows, img_cols )
    for i in range(mask.shape[3]):    
        temp = np.expand_dims( mask[:,:,:,i], axis = 3 )
        mask[:,:,:,i] = np.squeeze(preprocess(temp, img_rows, img_cols ), axis=3)
    masks = mask
    nImages = images.shape[0]
    

    print('-'*30)
    print('Save masks as images')
    print('-'*30)
    for imageIndex in range( nImages ):
        fileName = 'mask_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( output_masks_feed_dir, fileName )
        maskData = np.transpose( masks[imageIndex, ...], (2, 0, 1) )
        maskData = np.expand_dims( maskData, axis = 3 )
        writeData( filePath, maskData )

    print('-'*30)
    print('Save (preprocessed) images as images')
    print('-'*30)
    for imageIndex in range( nImages ):
        fileName = 'image_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( output_images_feed_dir, fileName )
        imageData = np.transpose( images[imageIndex, ...], (2, 0, 1) )
        imageData = np.expand_dims( imageData, axis = 3 )
        writeData( filePath, imageData )

    print('-'*30)
    print('Load the model')
    print('-'*30)
    metrics = getMetricFunctions()
    nClasses = 2
    optimizer = "adam"
    loss_metric = "categorical_crossentropy"
    activation = "sigmoid"
    learning_rate = 1e-5
    batch_size = 18
    model = getModel( "unet", nClasses, optimizer, activation, loss_metric, metrics, learning_rate, image_size )

    #show_pred_masks = trainCheck(flag)
    history = model.fit( images, masks, batch_size=batch_size, epochs=n_epochs, verbose=1, shuffle=True,
                     validation_split=0.2)#, callbacks=[show_pred_masks])

    print('-'*30)
    print('Saving model and training data (weights)')
    print('-'*30)

    #path.join( flag.output_models_weights_dir, flag.run_id + ".h5" ) )
    #weights_path = os.path.join( flag.output_weights_dir, flag.run_id + ".h5" )
    #model.save_weights( weights_path )
    #json_path = os.path.join( flag.output_models_dir, flag.model_id + ".json" )
    #json_string = model.to_json()
    #with open( json_path, 'w') as json_file:
    #    json_file.write( json_string )

    #plotVarList = getMetrics( history )
    #showTrainingHistory( flag, history, plotVarList )
#

imageFormat = "png"
regionList = [ "fg"]
extended_regionList = [ "bg", "fg" ]
train( 192, 2 )

#train( flag )

