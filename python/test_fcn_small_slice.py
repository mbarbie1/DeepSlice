# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
A full connected network on downscaled slices for segmentation

Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""
import tensorflow as tf

def deepnn( x, nX, nY, nChannels ):
    """
    deepnn builds the graph for a deep net for classifying the different regions for each pixel in an image.

    Args:
        x: an input tensor with the dimensions (nImages, nPixels).

    Returns:
        A tuple (y, keepProb). y is a tensor of shape (nPixels, nRegions), with values
        equal to the logits of classifying the pixel into one of nRegion classes (the
        regions brainstem, midbrain, ...). keepProb is a scalar placeholder for the probability of
        dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        xImage = tf.reshape(x, [-1, nX, nY, nChannels])
    
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    nFeatures1 = 32
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, nFeatures1])
        b_conv1 = bias_variable([nFeatures1])
        h_conv1 = tf.nn.relu(conv2d(xImage, W_conv1) + b_conv1)    
    
    nFeatures2 = 32
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([nX * nY * nChannels, nFeatures2])
        b_fc1 = bias_variable([nFeatures2])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    return y, keepProb


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
