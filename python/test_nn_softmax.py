# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
A neural network on small images for segmentation

Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""
import argparse
import sys

import tensorflow as tf
from scipy import misc
import numpy as np

FLAGS = None

def loadRegions( filePath ):
    """
    Read in regions for an image
    """
    roisArray = read_roi_zip( filePath )
    rois = {}
    for el in roisArray:
        rois[el[0].replace(".roi","")] = np.fliplr( el[1] )

    return rois

def loadImage( imagePath ):
    stack = misc.imread( imagePath )
    images = np.asarray(images)
    return images

def loadLabeledImage( imagePath, roisPath ):
    images = loadImage( imagePath )
    rois = loadRegions( roisPath )
    return images, rois


def main(_):
    """
    Builds the graph for a neural network for classifying the different regions for each pixel in an image.

    Args:
        x: an input tensor with the dimensions (nImages, nPixels).

    Returns:
        A tuple (yOut, keepProb). yOut is a tensor of shape (nPixels, nRegions), with values
        equal to the logits of classifying the pixel into one of nRegion classes (the
        regions brainstem, midbrain, ...).
    """
    
    dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"
  
    
    # Load the training data into two NumPy arrays, for example using `np.load()`.
    
    
    with np.load("/var/data/training_data.npy") as data:
        features = data["features"]
        labels = data["labels"]

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    # [Other transformations on `dataset`...]
    # dataset = ...
    iterator = dataset.make_initializable_iterator()

    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

    
    nPixels = 64
    nClasses = 2
    nBatch = 10
    nIteration = 20
    nOut = nPixels * nClasses
    x = tf.placeholder( tf.float32, [None, nPixels])
    W = tf.Variable( tf.truncated_normal([nPixels, nOut], stddev=0.1) )
    b = tf.Variable( tf.constant(0.1, shape=[nOut]) )
    y = tf.matmul( x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, nOut])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(nIteration):
        batch_xs, batch_ys = slices1.train.next_batch(nBatch)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: slices1.test.images,
                                      y_: slices1.test.labels}))


#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--data_dir', type=str, default='/home/mbarbier/Documents/prog/DeepSlice/data',
#                      help='Directory for storing input data')
#    FLAGS, unparsed = parser.parse_known_args()
#    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
tf.app.run( main=main )
  