# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:15:31 2018

@author: mbarbier
"""
""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
import argparse
from module_metrics_weight import weighted_categorical_crossentropy
from module_train import train
from module_utilities import writeDictToCsv
import time
from keras import losses
import numpy as np
#import Unet_train
#from Unet_test import *

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
parser.add_argument("--output_dir", help="output directory", default="./output" )
parser.add_argument("--output_models_subdir", help="output directory", default="Models" )
parser.add_argument("--output_weights_subdir", help="output directory", default="Weights" )
parser.add_argument("--output_models_weights_subdir", help="output directory", default="Models_Weights" )
parser.add_argument("--output_plots_subdir", help="output plot subdirectory", default="plots" )
parser.add_argument("--output_images_subdir", help="output images subdirectory", default="images" )
parser.add_argument("--output_masks_feed_subdir", help="output images subdirectory", default="masks_feed" )
parser.add_argument("--output_images_feed_subdir", help="output images subdirectory", default="images_feed" )
parser.add_argument("--network", help="Network name [unet, segnet]", default="segnet" )
parser.add_argument("--optimizer", help="Optimizer of the network [adam, sgd]", default="adam" )
parser.add_argument("--learning_rate", help="Learning rate of the optimizer", default=1e-5, type=float )
parser.add_argument("--loss_metric", help="Metric used as loss, if None, the network default metric is used", default=None )
parser.add_argument("--activation", help="Activation function of the output layer [sigmoid,softmax]", default="sigmoid" )
parser.add_argument("--load_model_weights_file", help="Path to load the model and weights (if the path exists the model will be loaded, h5 file expected)", default="" )
parser.add_argument("--load_model_file", help="Path to load the model architecture only, without the weights (if the path exists the model will be loaded, json file expected)", default="" )
parser.add_argument("--load_weights_file", help="Path to load the weights of a previous run (network should be the same, h5 file expected)", default="" )
parser.add_argument("--primitive_augmentation", help="Data augmentation using some primitive transformations (this is for testing only) [rotation_1, rotation_2, rotation_3]", default="" )
parser.add_argument("--data_augmentation", help="Do data augmentation [0=false, 1=true]", default=0, type=int )
parser.add_argument("--image_size", help="image size", default=192, type=int )
parser.add_argument("--batch_size", help="batch size", default=32, type=int )
parser.add_argument("--epochs", help="number of epochs", default=2, type=int )
parser.add_argument("--ratio_train", help="ratio training samples", default=0.8, type=float )
parser.add_argument("--binning", help="image binning", default=binning, type=int )
parser.add_argument("--region_list", help="List of ROIs", default=regionList )
parser.add_argument("--extended_region_list", help="List of ROIs", default=extended_regionList )
parser.add_argument("--image_format", help="Format of input and output images", default=imageFormat )
parser.add_argument("--show_metrics", help="Whether the computed metrics through training are shown (otherwise they are saved in the output plots subdirectory), 1 means also show, 0 means only save plot [1,0]", default=0, type=int )
parser.add_argument("--run_id_prefix", help="Prefix of the run_id which will determine the naming of the output folders and files", default="peter" )
parser.add_argument("--run_function", help="What to run? Training, loop training, prediction, both? [train,predict]", default="peter" )


flag = parser.parse_args()
parameterCsvPath = os.path.join( flag.output_dir, "run_parameters.csv")

def run_train_loop( flag ):
    """
    Runs the training of the network for all defined parameters. 
    For each parameter-set a training will be performed and output written in a different folder.
    This is also the place to define your parameters.
    """
    
    # List of neural networks to train:
    nns = ["unet"]

    # Use data augmentation (on the fly generation):
    flag.data_augmentation = 0
    # Simple data augmentation to test whether it would be useful (this is for testing only, be aware that the validation set is only split afterwards, so this can bias your training results).
    # Only useful when data augmentation is 0
    #flag.primitive_augmentation = "rotation_1"
    pas = [""] #["rotation_1", "rotation_2", "rotation_3"]

    # Number of epochs to train for:
    flag.epochs = 5

    # File to load pre-computed weights from (if commented no pre-computed weights are used)
    flag.load_weights_file = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output/Weights/save_unet_adam_categorical_crossentropy_epochs-1200_batch-size-18_image-size-192_lr-1e-05_data-augm-0.h5"
    
    # List of learning rates to train for:
    lrs = [1e-5]
    
    # List of batch-sizes to train for:
    bsz = [ 32 ]
    
    # List of loss metrics to train for:    
    lms = [ "categorical_crossentropy" ]
    
    # List of image sizes to train for (images are scaled to this resolution):
    imszs = [ 192 ]

    # starting to loop over all above parameter list given above:
    for nn in nns:
        for imsz in imszs:
            for lr in lrs:
                for bs in bsz:
                    for lm in lms:
                        for pa in pas:
                            flag.primitive_augmentation = pa
                            flag.network = nn
                            flag.image_size = imsz
                            flag.loss_metric = lm
                            flag.learning_rate = lr
                            flag.batch_size = bs
                            run_train( flag )


def run_train( flag ):
    """
    Runs the training of the network for all defined parameters. 
    For each parameter-set a training will be performed and output written in a different folder.
    This is also the place to define your parameters.
    """

    flag.run_id = '%s_%s_%s_%s_epochs-%d_batch-size-%d_image-size-%d_lr-%3.3g_data-augm-%d_prim-augm-%s' % ( flag.run_id_prefix, flag.network, flag.optimizer, flag.loss_metric, flag.epochs, flag.batch_size, flag.image_size, flag.learning_rate, flag.data_augmentation, flag.primitive_augmentation )
    flag.model_id = 'model_%s_%s_%s_image-size-%d_lr-%3.3g_prim-augm-%s' % ( flag.network, flag.optimizer, flag.loss_metric, flag.image_size, flag.learning_rate, flag.primitive_augmentation )
    flag.output_run_dir = os.path.join( flag.output_dir, "runs", flag.run_id )
    makeDirs( flag )
    t_begin = time.time()
    train( flag )
    t_end = time.time()
    t_training = t_end - t_begin
    print('-'*30)
    print('Training duration: %d s' % (t_training) )
    print('-'*30)
    flag.time_training = t_training
    # still needs to be sorted
    writeDictToCsv( parameterCsvPath, vars(flag) )


def run_predict( flag ):
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


def makeDirs( flag ):
    """
    Generate the necessary output folders    
    """
    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )
    if not os.path.isdir( flag.output_run_dir ):
        os.mkdir( flag.output_run_dir )

    flag.output_models_dir = os.path.join( flag.output_dir, flag.output_models_subdir)
    flag.output_models_weights_dir = os.path.join( flag.output_dir, flag.output_models_weights_subdir)
    flag.output_weights_dir = os.path.join( flag.output_dir, flag.output_weights_subdir)
    flag.output_plots_dir = os.path.join( flag.output_run_dir, flag.output_plots_subdir)
    flag.output_images_dir = os.path.join( flag.output_run_dir, flag.output_images_subdir)
    flag.output_images_feed_dir = os.path.join( flag.output_run_dir, flag.output_images_feed_subdir)
    flag.output_masks_feed_dir = os.path.join( flag.output_run_dir, flag.output_masks_feed_subdir)
    flag.output_plots_dir = os.path.join( flag.output_run_dir, flag.output_plots_subdir)
    flag.images_csv_path = os.path.join( flag.output_run_dir, flag.run_id + '.csv' )

    if not os.path.isdir( flag.output_models_dir ):
        os.mkdir( flag.output_models_dir )

    if not os.path.isdir( flag.output_models_weights_dir ):
        os.mkdir( flag.output_models_weights_dir )

    if not os.path.isdir( flag.output_weights_dir ):
        os.mkdir( flag.output_weights_dir )

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
    """
    This is the main function
    """

    run( flag )
    #train_op = Unet_train.TrainModel(flag)
    #train_op.train_unet()

if __name__ == '__main__':
    main()
