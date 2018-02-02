# -*- coding: utf-8 -*-
"""
The metrics for accuracy and loss calculations

Created on Tue Jan  9 14:12:52 2018

@author: mbarbier
"""
from keras import backend as K
import numpy as np



""" 
---------------------------------------------------------------------------
    Metrics functions and helper functions for calculation of the loss function
--------------------------------------------------------------------------- 
""" 
nClasses = 2


def dice_coef( y_true, y_pred ):
    """
    Dice coefficient of real and predicted mask
    """
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_acc( y_true, y_pred ):
    """
    Accuracy averaged over the labels, computing the general confusion matrix: is this too slow?
    """
    smooth = 1.
    #nClasses = 7#K.eval(y_true).shape[3]
    axis_pred = 1
    axis_gt = 0
    c = confusion_matrix( y_true, y_pred )
    
    JI = np.zeros( (nClasses) )
    cj = c.sum(axis=axis_gt)
    for k in range( nClasses ):
        JI[k] = ( c(k,k) + smooth ) / ( cj[k] + smooth )

    return np.sum( JI )

def mean_jaccard_index( y_true, y_pred ):
    """
    Accuracy averaged over the labels, computing the general confusion matrix: is this too slow?
    """
    smooth = 1.
    # TODO this is undefined Tensor type
    #nClasses = 7
    print( "Shape of the y_true = " + str( K.int_shape(y_true) ) )
    axis_pred = 1
    axis_gt = 0
    c = confusion_matrix( y_true, y_pred, nClasses )
    
    JI = np.zeros( (nClasses) )
    cj = c.sum(axis=axis_gt)
    ci = c.sum(axis=axis_pred)
    for k in range( nClasses ):
        JI[k] = ( c(k,k) + smooth ) / ( cj[k] + ci[k] - c(k,k) + smooth )

    return np.sum( JI )


def confusion_matrix( y_true, y_pred, nClasses ):
    """
    Multi-labels: computing the general confusion matrix: is this too slow?
    """

    # TODO this is undefined Tensor type
    print(nClasses)
    c = K.zeros( shape = ( nClasses, nClasses ) )
    smooth = K.ones( shape=(1,1) )
    for i in range( nClasses ):
        y_true_f = y_true[:,:,:,i]
        for j in range( nClasses ):
            y_pred_f = y_pred[:,:,:,j]
            c[i,j] = K.sum( K.flatten( y_true_f * y_pred_f ) )
    JI = K.zeros( (nClasses) )
    cj = K.sum( c, axis=axis_gt)
    ci = K.sum( c, axis=axis_pred)
    for k in range( nClasses ):
        JI[k] = ( c[k,k] + smooth ) / ( cj[k] + ci[k] - c(k,k) + smooth )

    return K.sum( JI )


def confusion_matrix_np( y_true, y_pred , nClasses ):
    """
    Numpy equivalent for testing: Multi-labels: computing the general confusion matrix?
    """

    c = np.zeros( shape = ( nClasses, nClasses ), dtype = np.float32 )
    smooth = 1.0
    for i in range( nClasses ):
        y_true_f = y_true[:,:,:,i].flatten()
        for j in range( nClasses ):
            y_pred_f = y_pred[:,:,:,j].flatten()
            c[i,j] = np.sum( y_true_f * y_pred_f )
    JI = np.zeros( (nClasses) )
    cj = np.sum( c, axis=axis_gt)
    ci = np.sum( c, axis=axis_pred)
    for k in range( nClasses ):
        JI[k] = ( c[k,k] + smooth ) / ( cj[k] + ci[k] - c(k,k) + smooth )

    return np.sum( JI ) / nClasses


""" 
---------------------------------------------------------------------------
    Loss functions from metrics
--------------------------------------------------------------------------- 
""" 

def my_sparse_categorical_crossentropy( y_true, y_pred ):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)


def pixelwise_binary_ce(y_true, y_pred):
    """
    Pixelwise cross-entropy between real and predicted mask
    (The y values should be [0,1])
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.mean( K.binary_crossentropy(y_pred, y_true) )


def dice_coef_loss( y_true, y_pred ):
    """
    Loss calculated from the dice coefficient
    """
    return -dice_coef(y_true, y_pred)


def pixelwise_l2_loss(y_true, y_pred):
    """
    Pixelwise L2 metric distance between real and predicted mask
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean( K.square(y_true_f - y_pred_f) )


def mean_acc_loss( y_true, y_pred ):
    """
    Loss calculated from the mean accuracy
    """
    return -mean_acc(y_true, y_pred)


def mean_jaccard_index_loss( y_true, y_pred ):
    """
    Loss calculated from the mean accuracy
    """
    return -mean_jaccard_index(y_true, y_pred)

def getMetricFunctions():
    return [ 'accuracy', dice_coef, pixelwise_l2_loss, pixelwise_binary_ce ]
    #return [ 'accuracy', my_sparse_categorical_crossentropy, mean_jaccard_index, mean_acc, dice_coef, pixelwise_l2_loss, pixelwise_binary_ce ]

    