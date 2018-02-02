# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:25:25 2018

@author: mbarbier
"""
from data_small import loadDataParams
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


def getParameterString( history):
    return "parameterString"

def getMetrics( history ):
    keys = list(history.history.keys())
    metrics = []
    for key in keys:
        if not key.startswith("val_"):
            metrics.append(key)
    return metrics
    #return ['dice_coef', 'loss', 'pixelwise_binary_ce', 'pixelwise_l2_loss']
    #return ['loss', 'acc']

def plotMetric( history, plotVar, parameterString, output_dir ):
    plt.figure()
    plt.plot(history.history[plotVar])
    plt.plot(history.history['val_' + plotVar])
    plt.title('Model ' + plotVar)
    plt.ylabel(plotVar)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig( os.path.join( output_dir, "Plot_" + plotVar + "_" + parameterString + "_" + '.png' ) )

# summarize history for accuracy
def showTrainingHistory( flag, history, plotVarList):
    parameterString = getParameterString( history )
    plotVar = getMetrics( history )
    for plotVar in plotVarList:
        plotMetric( history, plotVar, parameterString, flag.output_plots_dir )

from module_utilities import removeFolderContents
from keras.preprocessing.image import ImageDataGenerator

def train_generator( image_generator, mask_generator ):
        while True:
            yield( next(image_generator)[0], next(mask_generator)[0])

def dataAugmentation( flag, images, masks ):

    print('-'*30)
    print('Data augmentation')
    print('-'*30)
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    # Compute the statistics for data augmentation purposes (centering, normalization)
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)
    
    # The folders to save the extra images and corresponding masks of the data augmentation
    print('-'*30)
    print('Save data augmentation images')
    print('-'*30)
    flag.augm_masks_dir = os.path.join( flag.output_run_dir, "augm_masks" )
    flag.augm_images_dir = os.path.join( flag.output_run_dir, "augm_images" )
    if not os.path.isdir( flag.augm_masks_dir ):
        os.mkdir( flag.augm_masks_dir )
    if not os.path.isdir( flag.augm_images_dir ):
        os.mkdir( flag.augm_images_dir )

    fake_y = np.zeros( images.shape[0] )
    image_generator = image_datagen.flow(
        x = images,
        y = fake_y,
        save_to_dir = flag.augm_images_dir,
        save_prefix = 'augm_image',
        save_format = 'png',
        seed=seed)
    mask_generator = mask_datagen.flow(
        x = masks,
        y = fake_y,
        seed=seed)
    # Mask of dimension 7 cannot be saved as image 
    #        save_to_dir = flag.augm_masks_dir,
    #        save_prefix = 'augm_mask',
    #        save_format = 'png',

    # Empty the dirs
    removeFolderContents(flag.augm_images_dir)
    removeFolderContents(flag.augm_masks_dir)

    # combine generators into one which yields image and masks
    #train_generator = zip(image_generator, mask_generator)
    
    return train_generator(image_generator, mask_generator)

def splitDataSet( data, ratio ):
    
    n1 = round(ratio * data.shape[0])
    part1 = data[0:n1,...]
    part2 = data[(n1+1):,...]
    
    return part1, part2


def train( flag ):
    
    img_rows = flag.image_size
    img_cols = flag.image_size
    n_epochs = flag.epochs

    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    
    imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadDataParams( flag )
    images = imgs_train#np.expand_dims( imgs_train, axis = 3 )
    #images = np.expand_dims( images, axis = 3 )
    images = preprocess(images, img_rows, img_cols )
    masks_all = imgs_mask_train_all
    #loadDataParams( flag )

    print('-'*30)
    print('Fuse masks to single multi-label image')
    print('-'*30)
    regionList = flag.region_list
    regionIdList = range(1, len(regionList)+1)
    nClasses = len(regionList)+1
    #masks = masks_all[regionList[0]]
    s = masks_all[regionList[0]].shape
    nImages = s[0]
    masks = np.zeros( (nImages, img_rows, img_cols, nClasses) )
    masks_reshape = np.zeros( (nImages, img_rows * img_cols, nClasses) )
    #one_temp = np.ones( (s[0], img_rows, img_cols) )
    masks[:,:,:,0] = np.ones( (nImages, img_rows, img_cols) )
    for regionIndex in range(len(regionList)):
        # put the mask layer of the region index to 1
        masks_region = masks_all[regionList[regionIndex]]
        masks_region = preprocess(masks_region, img_rows, img_cols )
        temp = masks_region[:,:,:,0]
        masks[:,:,:,regionIndex+1] = temp
        # and the background layer to 0, for every region pixel
        masks[:,:,:,0] = (1 - temp) * masks[:,:,:,0]
        # Reshape (we don't need/use this???)
        temp = temp.reshape((nImages,img_rows * img_cols))
        masks_reshape[:,:,regionIndex+1] = temp
    

    print('-'*30)
    print('Save masks as images')
    print('-'*30)
    for imageIndex in range( nImages ):
        fileName = 'mask_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( flag.output_masks_feed_dir, fileName )
        maskData = np.transpose( masks[imageIndex, ...], (2, 0, 1) )
        maskData = np.expand_dims( maskData, axis = 3 )
        writeData( filePath, maskData )

    print('-'*30)
    print('Save (preprocessed) images as images')
    print('-'*30)
    for imageIndex in range( nImages ):
        fileName = 'image_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( flag.output_images_feed_dir, fileName )
        imageData = np.transpose( images[imageIndex, ...], (2, 0, 1) )
        imageData = np.expand_dims( imageData, axis = 3 )
        writeData( filePath, imageData )

    #masks = np.expand_dims( masks, axis = 3 )

    print('-'*30)
    print('Load the model')
    print('-'*30)
    metrics = getMetricFunctions()
    
    if ( flag.load_model_weights_file != "" ):
        model = load_model( flag.load_model_weights_file )
    elif ( flag.load_model_file != "" ):
        json_string = open( flag.load_model_file ).read()
        model = model_from_json( json_string )
    else:
        model = getModel( flag.network, nClasses, flag.optimizer, flag.activation, flag.loss_metric, metrics, flag.learning_rate, flag.image_size )

    if ( flag.load_weights_file != "" ):
        model.load_weights( flag.load_weights_file )

    model_checkpoint = ModelCheckpoint( os.path.join( flag.output_run_dir, 'weights.{epoch:03d}.h5'), period=n_epochs//10)

    show_pred_masks = trainCheck(flag)
    if flag.data_augmentation:
        #steps_per_epoch = len(train_generator) / flag.batch_size
        steps_per_epoch = 10
        images, imagesValidation = splitDataSet( images, 0.8 )
        masks, masksValidation = splitDataSet( masks, 0.8)
        train_generator = dataAugmentation( flag, images, masks )
        history = model.fit_generator( train_generator, 
                           validation_data = (imagesValidation, masksValidation), 
                          steps_per_epoch = steps_per_epoch,
                        epochs=n_epochs, verbose=1, shuffle=True,
                        callbacks=[model_checkpoint,show_pred_masks])
#        for e in range(flag.epochs):
#            print('Epoch', e)
#            batches = 0
#            for x_batch, y_batch in train_generator:
#                history = model.fit( x_batch[0], y_batch[0], validation_data = (imagesValidation, masksValidation), callbacks=[show_pred_masks])
#                print('batch ' + str(batches) + ' : do nothing, save something?')
#                #model.fit(x_batch, y_batch)
#                batches += 1
#                if batches >= len(images) / 32:
#                    # we need to break the loop by hand because
#                    # the generator loops indefinitely
#                    break
    
    else:
        #categorical_labels = to_categorical(int_labels, num_classes=None)
#        if model.loss is "categorical_crossentropy":
#            masks_cat = to_categorical(masks, num_classes=None)
        history = model.fit( images, masks, batch_size=flag.batch_size, epochs=n_epochs, verbose=1, shuffle=True,
                     validation_split=0.2, callbacks=[model_checkpoint,show_pred_masks])

    print('-'*30)
    print('Saving model and training data (weights)')
    print('-'*30)

    model.save( os.path.join( flag.output_models_weights_dir, flag.run_id + ".h5" ) )
    weights_path = os.path.join( flag.output_weights_dir, flag.run_id + ".h5" )
    model.save_weights( weights_path )
    json_path = os.path.join( flag.output_models_dir, flag.model_id + ".json" )
    json_string = model.to_json()
    with open( json_path, 'w') as json_file:
        json_file.write( json_string )

    plotVarList = getMetrics( history )
    showTrainingHistory( flag, history, plotVarList )
