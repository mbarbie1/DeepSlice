# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 12:42:59 2018

@author: mbarbier
"""
""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
import cv2
import argparse
from module_train import train
import time
from keras import losses
import numpy as np
from data_small import loadDataParams
from module_model import getModel, preprocess
from module_data import mergeLabelImage
from keras.models import load_model
from module_metrics_weight import weighted_categorical_crossentropy
from module_metrics import getMetricFunctions

def init():
    imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
    roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
    imageFormat = "png"
    regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
    extended_regionList = [ "bg", "cb", "hp", "cx", "th", "mb", "bs", "other" ]
    binning = 16
    dataFolder = "./data/features_labels_2d_" + str(binning)
    restRegionFolder = "./data/rest_region_labels_2d"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_rest_region_dir", help="Rest region masks path", default=restRegionFolder)
    parser.add_argument("--original_image_dir", help="training images path", default=imageFolder)
    parser.add_argument("--original_rois_dir", help="training ROIs path", default=roisFolder)
    parser.add_argument("--original_data_dir", help="training processing data path", default=dataFolder )
    parser.add_argument("--data_dir", help="training processing data path", default="./data" )
    parser.add_argument("--binning", help="image binning", default=binning, type=int )
    parser.add_argument("--region_list", help="List of ROIs", default=regionList )
    parser.add_argument("--image_format", help="Format of input and output images", default=imageFormat )
    parser.add_argument("--load_model_weights_file", help="Path to load the model and weights (if the path exists the model will be loaded, h5 file expected)", default="" )
    parser.add_argument("--image_size", help="image size", default=192, type=int )
    parser.add_argument("--batch_size", help="batch size", default=32, type=int )
    parser.add_argument("--ratio_train", help="ratio training samples", default=0.8, type=float )
    parser.add_argument("--network", help="Network name [unet, segnet]", default="segnet" )
    parser.add_argument("--optimizer", help="Optimizer of the network [adam, sgd]", default="adam" )
    parser.add_argument("--learning_rate", help="Learning rate of the optimizer", default=1e-5, type=float )
    parser.add_argument("--loss_metric", help="Metric used as loss, if None, the network default metric is used", default=None )
    parser.add_argument("--activation", help="Activation function of the output layer [sigmoid,softmax]", default="sigmoid" )

    parser.add_argument("--output_dir", help="output directory", default="./output" )
    parser.add_argument("--output_predict_subdir", help="output directory", default="peter_predictions" )

    flag = parser.parse_args()
    
    folder_model_weights = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output/Models_Weights"
    file_model_weights = "peter_unet_adam_categorical_crossentropy_epochs-2_batch-size-32_image-size-192_lr-1e-05_data-augm-0_prim-augm-.h5"
    #file_model_weights = "augm_small_unet_adam_categorical_crossentropy_epochs-500_batch-size-32_image-size-192_lr-1e-05_data-augm-0_prim-augm-rotation_3.h5"
    flag.load_model_weights_file = os.path.join( folder_model_weights, file_model_weights )

    folder_weights = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output/Weights"
    flag.load_weights_file = os.path.join( folder_weights, file_model_weights )
    
    return flag


def toMultiLabelMask( masks_all, nClasses, regionList, img_rows, img_cols ):

    s = masks_all[regionList[0]].shape
    nImages = s[0]
    masks = np.zeros( (nImages, img_rows, img_cols, nClasses) )
    masks[:,:,:,0] = np.ones( (nImages, img_rows, img_cols) )
    for regionIndex in range(len(regionList)):
        # put the mask layer of the region index to 1
        masks_region = masks_all[regionList[regionIndex]]
        masks_region = preprocess(masks_region, img_rows, img_cols )
        temp = masks_region[:,:,:,0]
        masks[:,:,:,regionIndex+1] = temp
        # and the background layer to 0, for every region pixel
        masks[:,:,:,0] = (1 - temp) * masks[:,:,:,0]

    return masks


def predict( flag ):
    
    flag.network = "unet"
    flag.data_augmentation = 0
    flag.primitive_augmentation = ""
    flag.epochs = 2
    flag.learning_rate = 1e-5
    flag.batch_size = 32
    flag.loss_metric = "categorical_crossentropy"

    img_rows = flag.image_size
    img_cols = flag.image_size

    print('-'*30)
    print('Loading testing data...')
    print('-'*30)
    
    imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadDataParams( flag )
    images = imgs_test
    images = preprocess(images, img_rows, img_cols )
   
    regionList = flag.region_list
    nClasses = len(regionList)+1

    masks = toMultiLabelMask( imgs_mask_test_all, nClasses, regionList, img_rows, img_cols  )

    metrics = getMetricFunctions()
    model = getModel( flag.network, nClasses, flag.optimizer, flag.activation, flag.loss_metric, metrics, flag.learning_rate, flag.image_size )
    model.load_weights( flag.load_weights_file )
    #model = load_model( flag.load_model_weights_file )

    pred = model.predict( images, batch_size=flag.batch_size, verbose=0, steps=None)
    
    return images, pred, masks

def saveLabelImage( mask, output_path ):
    y = mergeLabelImage(mask).astype( np.float32 )
    y = y + 0.9
    y /= y.max()
    maskOri = (y*255).astype( np.uint8 )
    maskOriColor = cv2.applyColorMap( maskOri, cv2.COLORMAP_HSV )
    maskColor = np.flip(maskOriColor, 2)
    #maskColor = maskOriColor
    cv2.imwrite( output_path, maskColor )

def makeOverlay( img, mask, output_path ):
    
    y = mergeLabelImage(mask).astype( np.float32 )
    y = y + 0.9
    y /= y.max()
    maskOri = (y*255).astype( np.uint8 )
    maskOriColor = cv2.applyColorMap( maskOri, cv2.COLORMAP_HSV )
    scaleIntensity = 500
    img = img.astype( np.float32 )
    img /= img.max()
    img = np.fmin( 255.0 * np.ones(img.shape, np.float32 ), scaleIntensity * img )
    img = img.astype( np.uint8 )
    maskColor = np.flip(maskOriColor, 2)    
    imgColor = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
    imgOverlay = cv2.addWeighted(imgColor, 0.9, maskColor, 0.4, 0.0)
    cv2.imwrite( output_path, imgOverlay )

def saveImage( img, output_path ):
    
    scaleIntensity = 500
    img = img.astype( np.float32 )
    img /= img.max()
    img = np.fmin( 255.0 * np.ones(img.shape, np.float32 ), scaleIntensity * img )
    img = img.astype( np.uint8 )
    img = np.squeeze(img)
    #imgColor = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
    cv2.imwrite( output_path, img )

def saveImageRaw( img, output_path ):

    #scaleIntensity = 500
    #img = img.astype( np.float32 )
    #img /= img.max()
    #img = np.fmin( 255.0 * np.ones(img.shape, np.float32 ), scaleIntensity * img )
    img = img.astype( np.uint8 )
    img = np.squeeze(img)
    #imgColor = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
    cv2.imwrite( output_path, img )


def makeOverlayMaskDifference( img, mask_ref, mask, output_path ):
    
    y = mergeLabelImage(mask).astype( np.float32 )
    y_ref = mergeLabelImage(mask_ref).astype( np.float32 )
    diff = np.zeros( y.shape )
    diff[ y != y_ref ] = 1
    diff /= diff.max()
    diff = (diff*255).astype( np.uint8 )
    diffColor = cv2.applyColorMap( diff, cv2.COLORMAP_JET )
    scaleIntensity = 500
    img = img.astype( np.float32 )
    img /= img.max()
    img = np.fmin( 255.0 * np.ones(img.shape, np.float32 ), scaleIntensity * img )
    img = img.astype( np.uint8 )
    diffColor = np.flip( diffColor, 2)
    imgColor = cv2.cvtColor( img, cv2.COLOR_GRAY2BGR )
    imgOverlay = cv2.addWeighted(imgColor, 0.9, diffColor, 0.4, 0.0)
    cv2.imwrite( output_path, imgOverlay )


def dice_coef_single( y_true, y_pred ):
    """
    Dice coefficient of real and predicted mask
    """
    smooth = 1.
    intersection = np.sum(y_true * y_pred )
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from module_utilities import writeDictToCsv
def saveClassificationMetrics( preds, masks, regionList, output_path ):
    #print( classification_report( y_true, y_pred ) )
    nImages = preds.shape[0]
    nClasses = preds.shape[-1]
    d = np.zeros( ( nImages, nClasses) )
    for i in range( nImages ):
        ys_true = preds[i,...]
        ys_pred = masks[i,...]
        nClasses = ys_true.shape[-1]
        for m in range( nClasses ):
            y_true = ys_true[:,:,m]
            y_pred = ys_pred[:,:,m]
            d[i,m] = dice_coef_single( y_true, y_pred )

    for i in range( nImages ):
        dic = {}
        for m in range( nClasses ):
            dic[regionList[m]] = str(d[i,m])
        writeDictToCsv( output_path, dic )



def run( flag ):
    """
    This is the main function
    """

    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )
    flag.output_predict_dir = os.path.join( flag.output_dir, flag.output_predict_subdir)
    if not os.path.isdir( flag.output_predict_dir ):
        os.mkdir( flag.output_predict_dir )

    images, pred, masks = predict( flag )

    outputName = 'metrics_table.csv'
    outputPath = os.path.join( flag.output_predict_dir, outputName  )
    flag.region_list = ["bg"] + flag.region_list
    saveClassificationMetrics( pred, masks, flag.region_list, outputPath )

    nImages = images.shape[0]
    for i in range( nImages ):
        img = images[i,...]
        mask = pred[i,...]
        mask_ref = masks[i,...]

        outputName = 'pred_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        makeOverlay( img, mask, outputPath )

        outputName = 'img_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        saveImage( img, outputPath )

        outputName = 'img-raw_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        saveImageRaw( img, outputPath )

        outputName = 'manual_mask_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        saveLabelImage( mask_ref, outputPath )        

        outputName = 'pred_mask_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        saveLabelImage( mask, outputPath )        

        outputName = 'manual_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        makeOverlay( img, mask_ref, outputPath )

        outputName = 'difference_%d.png' % ( i )
        outputPath = os.path.join( flag.output_predict_dir, outputName  )
        makeOverlayMaskDifference( img, mask_ref, mask, outputPath )
        

def main():
    """
    This is the main function
    """
    flag = init()
    run( flag )


if __name__ == '__main__':
    main()
