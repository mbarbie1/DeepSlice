# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:36:48 2018
@author: mbarbier
"""
""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
import argparse
from module_train import train
from module_utilities import writeDictToCsv
import time
from keras import losses
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def multi_slice_viewer(volume):
    volume = np.transpose( volume, (2,0,1) )
    #remove_keymap_conflicts({'j', 'i'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    #slider = Slider(ax, 'slice', 0, volume.shape[0], valinit=0)
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'i':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def imageShow( y ):
    img = Image.fromarray(255*y/y.max(),"F")
    img.resize( (100,100), resample=Image.NEAREST )
    plt.imshow(img, cmap="Greys")
    #plt.colorbar()


def splitLabels( yint, nClasses ):
    nRows = yint.shape[0]
    nCols = yint.shape[1]
    y = np.zeros( (nRows, nCols, nClasses) )
    y[:,:,0] = np.ones( (nRows, nCols) )
    for regionIndex in range( 1, nClasses ):
        # put the mask layer of the region index to 1
        y_region = np.zeros( (nRows, nCols) )
        y_region[ yint == regionIndex ] = 1
        y[:,:,regionIndex] = y_region
        # and the background layer to 0, for every region pixel
        y[:,:,0] = (1 - y_region) * y[:,:,0]
    return y
    


def compare( flag ):

    print('-'*30)
    print('Masks as single channel with multiple 1, .. , n labels')
    print('-'*30)

    y = np.zeros( (3, 3), np.float32 )
    y[1,1] = 2
    y[1,2] = 2
    y[0,2] = 1
    y_pred = np.zeros( (3, 3), np.float32 )
    y_pred[1,1] = 1
    y_pred[2,2] = 2
    y_pred[1,2] = 2

    nClasses = int(y.max()) + 1
    y_cat = splitLabels( y, nClasses )
    y_pred_cat = splitLabels( y_pred, nClasses )

    multi_slice_viewer(y_cat)
    plt.figure()
    multi_slice_viewer(y_pred_cat)

#    imageShow( y )
#    imageShow( y_pred )
#    multi_slice_viewer(y_cat)

    #confusion_matrix_np( y_true, y_pred , nClasses )
    
    #img.save('my.png')
    #img.show()

def init():
    regionList = [ "1", "2", "3" ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="output directory", default="./output_loss" )
    parser.add_argument("--loss_metric", help="Metric used as loss, if None, the network default metric is used", default=None )
    parser.add_argument("--region_list", help="List of ROIs", default=regionList )
    parser.add_argument("--image_format", help="Format of input and output images", default="png" )
    flag = parser.parse_args()
    return flag

def run( flag ):
    makeDirs( flag )
    compare( flag )


def makeDirs( flag ):
    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )

def main():
    flag = init()
    run( flag )

if __name__ == '__main__':
    main()




"""    
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
    #model = get_unet( img_rows, img_cols )
    metrics = getMetricFunctions()
    model = getModel( flag.network, nClasses, flag.optimizer, flag.activation, flag.loss_metric, metrics, flag.learning_rate, flag.image_size )
    #model = unet( nClasses = nClasses, optimizer = None, img_rows = img_rows, img_cols = img_cols )
    #model = segnet( nClasses = nClasses, optimizer = None, img_rows = img_rows, img_cols = img_cols )


    model_checkpoint = ModelCheckpoint( os.path.join( flag.output_run_dir, 'weights.{epoch:03d}.h5'), period=n_epochs//10+1)

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
                 

    # list all data in history
    #print(history.history.keys())              
    plotVarList = getMetrics( history )
    showTrainingHistory( flag, history, plotVarList )
    #showTrainingHistoryMultiClass(history)
    """