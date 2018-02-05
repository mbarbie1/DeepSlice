# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:36:33 2018

@author: mbarbier
"""
from data_small import loadDataParams
from module_model import preprocess
from module_utilities import writeData
from module_data import mergeLabelImage
import numpy as np
import os
import cv2
import argparse

imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
imageFormat = "png"
regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
binning = 16
dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data/features_labels_2d_" + str(binning)

parser = argparse.ArgumentParser()
parser.add_argument("--original_image_dir", help="training images path", default=imageFolder)
parser.add_argument("--original_rois_dir", help="training ROIs path", default=roisFolder)
parser.add_argument("--original_data_dir", help="training processing data path", default=dataFolder )
parser.add_argument("--data_dir", help="training processing data path", default="./data" )
parser.add_argument("--output_dir", help="output directory", default="./output_pres" )
parser.add_argument("--output_plots_subdir", help="output plot subdirectory", default="plots" )
parser.add_argument("--output_images_subdir", help="output images subdirectory", default="images" )
parser.add_argument("--output_masks_feed_subdir", help="output images subdirectory", default="masks_feed" )
parser.add_argument("--output_images_feed_subdir", help="output images subdirectory", default="images_feed" )
parser.add_argument("--network", help="Network name [unet, segnet]", default="segnet" )
parser.add_argument("--optimizer", help="Optimizer of the network [adam, sgd]", default="adam" )
parser.add_argument("--learning_rate", help="Learning rate of the optimizer", default=1e-5, type=float )
parser.add_argument("--loss_metric", help="Metric used as loss, if None, the network default metric is used", default=None )
parser.add_argument("--activation", help="Activation function of the output layer [sigmoid,softmax]", default="sigmoid" )
parser.add_argument("--data_augmentation", help="do data augmentation [0=false, 1=true]", default=0, type=int )
parser.add_argument("--image_size", help="image size", default=192, type=int )
parser.add_argument("--batch_size", help="batch size", default=32, type=int )
parser.add_argument("--epochs", help="number of epochs", default=2, type=int )
parser.add_argument("--ratio_train", help="ratio training samples", default=0.8, type=float )
parser.add_argument("--binning", help="image binning", default=binning, type=int )
parser.add_argument("--region_list", help="List of ROIs", default=regionList )
parser.add_argument("--image_format", help="Format of input and output images", default=imageFormat )

flag = parser.parse_args()
parameterCsvPath = os.path.join( flag.output_dir, "run_parameters.csv")

def run( flag ):
    flag.run_id = 'save_images_image-size-%d' % ( flag.image_size )
    flag.output_run_dir = os.path.join( flag.output_dir, flag.run_id )
    makeDirs( flag )
    showData( flag )

def showData( flag ):
    
    img_rows = flag.image_size
    img_cols = flag.image_size

    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    
    imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadDataParams( flag )
    images = imgs_train
    images = preprocess(images, img_rows, img_cols )
    masks_all = imgs_mask_train_all

    print('-'*30)
    print('Fuse masks to single multi-label image')
    print('-'*30)
    regionList = flag.region_list
    nClasses = len(regionList)+1
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
    

    print('-'*30)
    print('Save masks as images')
    print('-'*30)
    scaleIntensity = 1
    for imageIndex in range( nImages ):
        fileName = 'mask_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( flag.output_masks_feed_dir, fileName )
        maskData = np.transpose( masks[imageIndex, ...], (2, 0, 1) )
        maskData = np.expand_dims( maskData, axis = 3 )
        writeData( filePath, maskData )
    
        y = masks[imageIndex,...]
        y = mergeLabelImage(y).astype( np.float32 )
        y /= y.max()
        x = images[imageIndex,...].astype( np.float32 )
        x = np.fmin( 255.0 * np.ones(x.shape, np.float32 ), scaleIntensity * x )
        x = x.astype( np.uint8 )
        x = np.squeeze(x)
        print(x.shape)
        #x /= x.max()
        #img = (x*255).astype( np.uint8 )
        imgColor = cv2.cvtColor( x, cv2.COLOR_GRAY2BGR )
        imgMaskOri = (y*255).astype( np.uint8 )
        imgMaskOriColor = cv2.applyColorMap( imgMaskOri, cv2.COLORMAP_JET )
        imgOverlay = cv2.addWeighted(imgColor, 0.9, imgMaskOriColor, 0.4, 0.0)
        output_path_mask_ori = os.path.join( flag.output_images_dir, 'mask_index-%04d.png' % ( imageIndex) )
        output_path_overlay = os.path.join( flag.output_plots_dir, 'overlay_index-%04d.png' % ( imageIndex) )
        output_path_image = os.path.join( flag.output_plots_dir, 'image_index-%04d.png' % ( imageIndex) )
        cv2.imwrite( output_path_image, imgColor )
        cv2.imwrite( output_path_mask_ori, imgMaskOriColor )
        cv2.imwrite( output_path_overlay, imgOverlay )

    print('-'*30)
    print('Save (preprocessed) images as images')
    print('-'*30)
    for imageIndex in range( nImages ):
        fileName = 'image_sample_%d.tif' % ( imageIndex )
        filePath = os.path.join( flag.output_images_feed_dir, fileName )
        imageData = scaleIntensity * np.transpose( images[imageIndex, ...], (2, 0, 1) )
        imageData = np.expand_dims( imageData, axis = 3 )
        writeData( filePath, imageData )


def makeDirs( flag ):
    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )
    if not os.path.isdir( flag.output_run_dir ):
        os.mkdir( flag.output_run_dir )

    flag.output_plots_dir = os.path.join( flag.output_run_dir, flag.output_plots_subdir)
    flag.output_images_dir = os.path.join( flag.output_run_dir, flag.output_images_subdir)
    flag.output_images_feed_dir = os.path.join( flag.output_run_dir, flag.output_images_feed_subdir)
    flag.output_masks_feed_dir = os.path.join( flag.output_run_dir, flag.output_masks_feed_subdir)
    flag.output_plots_dir = os.path.join( flag.output_run_dir, flag.output_plots_subdir)
    flag.images_csv_path = os.path.join( flag.output_run_dir, flag.run_id + '.csv' )

    if not os.path.isdir( flag.output_plots_dir ):
        os.mkdir( flag.output_plots_dir )

    if not os.path.isdir( flag.output_images_dir ):
        os.mkdir( flag.output_images_dir )

    if not os.path.isdir( flag.output_images_feed_dir ):
        os.mkdir( flag.output_images_feed_dir )

    if not os.path.isdir( flag.output_masks_feed_dir ):
        os.mkdir( flag.output_masks_feed_dir )

    if not os.path.isdir( flag.data_dir ):
        os.mkdir( flag.data_dir )

def main():

    run( flag )
    #train_op = Unet_train.TrainModel(flag)
    #train_op.train_unet()

if __name__ == '__main__':
    main()
