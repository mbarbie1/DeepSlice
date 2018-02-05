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
dataFolder = "data/features_labels_2d_" + str(binning)
restRegionFolder = "data/rest_region_labels_2d"

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
parser.add_argument("--data_augmentation", help="do data augmentation [0=false, 1=true]", default=0, type=int )
parser.add_argument("--image_size", help="image size", default=192, type=int )
parser.add_argument("--batch_size", help="batch size", default=32, type=int )
parser.add_argument("--epochs", help="number of epochs", default=2, type=int )
parser.add_argument("--ratio_train", help="ratio training samples", default=0.8, type=float )
parser.add_argument("--binning", help="image binning", default=binning, type=int )
parser.add_argument("--region_list", help="List of ROIs", default=regionList )
parser.add_argument("--extended_region_list", help="List of ROIs", default=extended_regionList )
parser.add_argument("--image_format", help="Format of input and output images", default=imageFormat )

flag = parser.parse_args()
parameterCsvPath = os.path.join( flag.output_dir, "run_parameters.csv")

def run( flag ):
    nns = ["unet"]
    #flag.data_augmentation = 1
    flag.epochs = 20
    #/home/mbarbier/Documents/prog/DeepSlice/
    #flag.load_weights_file = "python/keras_small/output/Weights/save_unet_adam_categorical_crossentropy_epochs-1200_batch-size-18_image-size-192_lr-1e-05_data-augm-0.h5"
    lrs = [1e-5]
    bsz = [ 18 ]
    
    lms = [ "categorical_crossentropy" ]
    imszs = [ 192 ]
    for nn in nns:
        for imsz in imszs:
            for lr in lrs:
                for bs in bsz:
                    for lm in lms:
                        flag.network = nn
                        flag.image_size = imsz
                        flag.loss_metric = lm
                        flag.learning_rate = lr
                        flag.batch_size = bs
                        flag.run_id = 'save_%s_%s_%s_epochs-%d_batch-size-%d_image-size-%d_lr-%3.3g_data-augm-%d' % ( flag.network, flag.optimizer, flag.loss_metric, flag.epochs, flag.batch_size, flag.image_size, flag.learning_rate, flag.data_augmentation )
                        flag.model_id = 'model_%s_%s_%s_image-size-%d_lr-%3.3g' % ( flag.network, flag.optimizer, flag.loss_metric, flag.image_size, flag.learning_rate )
                        flag.output_run_dir = os.path.join( flag.output_dir, flag.run_id )
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


def makeDirs( flag ):
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

    run( flag )
    #train_op = Unet_train.TrainModel(flag)
    #train_op.train_unet()

if __name__ == '__main__':
    main()
