# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
A neural network on small images for segmentation

Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""
import tensorflow as tf

def nn():
    """
    nn builds the graph for a neural network for classifying the different regions for each pixel in an image.

    Args:
        x: an input tensor with the dimensions (nImages, nPixels).

    Returns:
        A tuple (yOut, keepProb). yOut is a tensor of shape (nPixels, nRegions), with values
        equal to the logits of classifying the pixel into one of nRegion classes (the
        regions brainstem, midbrain, ...).
    """
    nPixels = 64
    nClasses = 2
    nBatch = 10
    nIteration = 20
    nOut = nPixels * nClasses
    x = tf.placeholder( tf.float32, [None, nPixels])
    W = tf.Variable( tf.truncated_normal([nPixels, nOut], stddev=0.1) )
    b = tf.Variable( tf.constant(0.1, shape=[nOut]) )
    yOut = tf.matmul( x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, nOut])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(nIteration):
        batch_xs, batch_ys = mnist.train.next_batch(nBatch)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

tf.app.run( main=nn, argv=[sys.argv[0]] + unparsed)
