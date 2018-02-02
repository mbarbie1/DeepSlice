# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:43:50 2018

@author: mbarbier
"""
from __future__ import print_function

import numpy as np
from skimage.transform import resize
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from module_model_unet import unet
from module_model_segnet import segnet
from keras import losses
from module_metrics import mean_acc_loss, mean_jaccard_index_loss, my_sparse_categorical_crossentropy, dice_coef_loss, pixelwise_l2_loss

def getOptimizer( optimizerName, learning_rate ):

    optimizer = None
    lr = learning_rate

    if optimizerName == "adam":
        optimizer = Adam(lr=lr)
    if optimizerName == "sgd":
        optimizer = SGD(lr=lr)
    if optimizerName == "rmsprop":
        optimizer = RMSprop(lr=lr)
    if optimizerName == "adagrad":
        optimizer = Adagrad(lr=lr)
    if optimizerName == "adadelta":
        optimizer = Adadelta(lr=lr)
    if optimizerName == "adamax":
        optimizer = Adamax(lr=lr)
    if optimizerName == "nadam":
        optimizer = Nadam(lr=lr)

    return optimizer

def getLoss( lossName ):

    loss = None

    if lossName == "categorical_crossentropy":
        loss = losses.categorical_crossentropy
    if lossName == "sparse_categorical_crossentropy":
        loss = losses.sparse_categorical_crossentropy
    if lossName == "dice_coef":
        loss = dice_coef_loss
    if lossName == "mean_jaccard_index":
        loss = mean_jaccard_index_loss
    if lossName == "mean_acc":
        loss = mean_acc_loss

    return loss


def getModel( modelName, nClasses, optimizerName, activationName, lossName, metrics, learning_rate, image_size ):

    optimizer = getOptimizer( optimizerName, learning_rate )    
    
    loss=getLoss( lossName )
    
    if modelName == "unet":
        return unet( nClasses=nClasses , optimizer=optimizer, activationName='softmax', loss=loss, metrics = metrics, img_rows=image_size, img_cols=image_size )
    if modelName == "segnet":
        return segnet( nClasses=nClasses , optimizer=optimizer, activationName='softmax', loss=loss, metrics = metrics, img_rows=image_size, img_cols=image_size )

    return segnet( nClasses=nClasses , optimizer=optimizer, activationName='softmax', loss=loss, metrics = metrics, img_rows=image_size, img_cols=image_size )


def preprocess(imgs, img_rows, img_cols ):
    """
    Does the preprocessing of the data images, this belongs to another module?
    At this moment does not perform any intensity scaling, only resizes the image and converts to uint8 format
    """
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8) #"uint8"
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p
