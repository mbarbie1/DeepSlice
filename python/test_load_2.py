# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
A neural network on small images for segmentation

Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""

#%%
from __future__ import division

""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import argparse
import sys
import re
import os

import tensorflow as tf
from scipy import misc
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import img_as_float,img_as_uint
from skimage.filters import gaussian
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from shapely.geometry import Polygon
import shapely
from ijroi import read_roi_zip as read_roi_zip_ijroi
from read_roi import read_roi_zip as read_roi_zip_read_roi
from PIL import Image
import math

FLAGS = None

def xyKey():
    return "xy"

def polyRegionKey():
    return "polygon"


def loadRegionsStack( filePath, prefix ):
    """
    Read in regions for an image stack
    """
    roisDefs = read_roi_zip_read_roi( filePath )
    roisArray = read_roi_zip_ijroi( filePath )
    slices = {}
    for el in roisArray:
        roiNameOri = el[0].replace(".roi","")
        roiName = re.sub( r"-.*$", "", roiNameOri )
        roi = {}
        roi["z"] = roisDefs[roiNameOri]["position"]
        imageId = prefix + "-" + str(roi["z"]).zfill(2)
        roi["image_id"] = imageId
        xy = np.fliplr( el[1] )
        roi["xy"] = xy
        if not (imageId in slices):
            slices[imageId] = {}
        slices[imageId][roiName] = roi
    return slices

def convertRegionsToPolygons( rois ):
    roisPoly = {}
    for key in rois.keys():
        roisPoly[key] = rois[key]
        roisPoly[key][polyRegionKey()] = Polygon(rois[key][xyKey()])
    return roisPoly

def loadRegionsImage( filePath, imageId ):
    """
    Read in regions for an image
    """
    roisDefs = read_roi_zip_read_roi( filePath )
    roisArray = read_roi_zip_ijroi( filePath )
    rois = {}
    for el in roisArray:
        roiNameOri = el[0].replace(".roi","")
        # remove any enumeration from the rois (roiName-1, roiName-2, ...)
        roiName = re.sub( r"-.*$", "", roiNameOri )
        roi = {}
        roi["z"] = roisDefs[roiNameOri]["position"]
        roi["image_id"] = imageId
        xy = np.fliplr( el[1] )
        roi["xy"] = xy
        rois[roiName] = roi
    return rois


def loadImage( imagePath ):
    """
    Read in an image
    """
    images = io.imread( imagePath )
    return images


def loadLabeledImage( imagePath, roisPath, imageId ):
    """
    Read in an image and its regions
    """
    images = loadImage( imagePath )
    rois = loadRegionsImage( roisPath, imageId )
    return images, rois


def showOverlayRegions( image, rois, nFig ):
    """
    shows the regions and returns the matplotlib axes
    """
    fig = plt.figure(nFig, dpi=90)
    plt.imshow( image, cmap='gray' )
    plt.axis('equal')
    ax = fig.add_subplot(111)
    for key in rois.keys():
        xy = rois[key]["xy"]
        region = Polygon( xy )
        x, y = region.exterior.xy
        plt.plot(x, y)
    return fig, ax


def rasterizePolygon( image, polygon ):
    mask = rasterio.features.rasterize(
        [(polygon, 1)],
        out_shape=image.shape,
        fill=0,
        all_touched=True,
        dtype=np.uint8)
    return mask


def maskImage( image, mask ):
    maskedImage = np.ma.array( data=image, mask=mask.astype(bool))
    return maskedImage


def imageSize( imagePath ):
    with Image.open( imagePath ) as img:
        width, height = img.size
        return width, height


def findFiles( folderPath, ext, containString ):
    fileList = []
    for root, directories, fileNameList in os.walk( folderPath ):
        fileNameList.sort();
        for fileName in fileNameList:
            # Check for file extension
            if not fileName.endswith(ext):
                continue
            # Check for file name pattern
            if containString not in fileName:
                continue
            filePath = os.path.join( folderPath, fileName)
            fileList.append( filePath )
    return fileList


def imageSizeList( imagePathList ):
    sizeList = []
    for imagePath in imagePathList:
        width, height = imageSize( imagePath )
        sizeList.append( (width, height) )
    return sizeList


def maxImageSize( sizeList ):
    maxWidth = 0
    maxHeight = 0
    for size in sizeList:
        maxWidth = max( maxWidth, size[0] )
        maxHeight = max( maxHeight, size[1] )
    return maxWidth, maxHeight

def upperLeftOrigin( largeSize, smallSize ):
    """
    The upper left coordinate (tuple) of a small rectangle in a larger rectangle (centered)
    """
    origin = tuple( map( lambda x: int( ( (x[0]-x[1])/2 ) ), zip( largeSize, smallSize )) )
    return origin

def extendImageNumpy( img, side ):
    """
    return a extended (zero-padded) image (numpy array) having the img in the center
    """
    size = (side, side)
    imgResized = np.zeros( size, img.dtype )
    imgSize = img.shape
    pasteOrigin = upperLeftOrigin( size, imgSize )
    imgResized[pasteOrigin[0]:(pasteOrigin[0]+imgSize[0]), pasteOrigin[1]:(pasteOrigin[1]+imgSize[1])] = img
    return imgResized


def extendImagePillow( img, side ):
    """
    return a extended (zero-padded) image (pillow) having the img in the center
    """
    size = ( side, side )
    imgResized = Image.new( img.mode, size )
    pasteOrigin = upperLeftOrigin( size, img.size )
    imgResized.paste(img, pasteOrigin )
    return imgResized

def extendRegions( rois, sizeResized, sizeOri ):
    newRois = rois
    newOrigin = upperLeftOrigin( sizeResized, sizeOri )
    for key in rois.keys():
        shift = np.array( [(newOrigin[0], newOrigin[1])] )
        newRois[key][xyKey()] = rois[key][xyKey()] + shift
        #rois[key] = shapely.affinity.translate( roi , xoff=newOrigin[0], yoff=newOrigin[1], zoff=0.0)
    return newRois

def convertRegionsToMasks( rois, im ):
    masks = {}    
    for key in rois.keys():
        polygon = Polygon( rois[key][xyKey()] )
        mask = rasterizePolygon( im, polygon )
        masks[key] = mask
    return masks

def imageIdFromFileName( imagePath ):
    """
    TODO
    """
    imageId = "fakeID"    
    return imageId

def loadImages( imageFolder, imageFormat, contains ):
    imagePathList = findFiles( imageFolder, imageFormat, contains )
    sizeList = imageSizeList( imagePathList )
    maxWidth, maxHeight = maxImageSize( sizeList )
    side = max( maxWidth, maxHeight )
    imageList = {}
    roisList = {}
    for imagePath in imagePathList:
        image, rois = loadLabeledImage( imagePath, roisPath, imageId )
        roisList[imageId] = rois
        imageList[imageId] = image

def preLoadLabeledImages( imageFolder, imageFormat, contains ):
    """
    List all image paths and sizes and rois but do not load the images into memory    
    Open images but don't load into memory
    """
    imagePathList = findFiles( imageFolder, imageFormat, contains )
    sizeList = imageSizeList( imagePathList )
    maxWidth, maxHeight = maxImageSize( sizeList )
    side = max( maxWidth, maxHeight )
    return imagePathList, roisList


import math
def featureAndLabels( imageFolder, imageFormat, roisFolder, binning, regionList, contains ):
    """
    Generates feature vectors and label vectors from the ROIs, saves as a (nImages, nPixels) numpy arrays
        features = single array (only intensity)
        labels = dictionary with for each region the masks as values (e.g. labels["cb"] = single array)
    """

    # Get all the images in the imageFolder
    imagePathList = findFiles( imageFolder, imageFormat, contains )

    # Obtain the list of image sizes and their maximum
    sizeList = imageSizeList( imagePathList )
    maxWidth, maxHeight = maxImageSize( sizeList )
    side = max( maxWidth, maxHeight )

    # Find the scaled maximum size of the images    
    scale = 1.0 / float(binning)
    scaledSide = int( math.ceil( scale * side ) )
    nPixels = scaledSide**2

    # Prepare the feature and label arrays
    imageIdList = list( map( lambda x: os.path.splitext(os.path.basename(x))[0], imagePathList ) )
    nImages = len( imageIdList )
    features = np.zeros( (nImages, nPixels), float )
    labels = {}
    for region in regionList:
        labels[region] = np.zeros( (nImages, nPixels), int )

    # For every image:
    #   load images, ROIs,
    #   scale them
    #   Extend to max. size
    #   Make a feature vector from them
    #   Generate a mask from the ROIs
    arrayIndex = 0
    for imageId in imageIdList:

        # Print imageId
        print(imageId)

        # Load image
        image = loadImage( imageFolder + "/" + imageId + "." + imageFormat )
        # Scale the image
        image = gaussian(image, sigma=(1.0/scale)/2.0)
        image_downscaled = downscale_local_mean( image, (binning, binning) )
        img = Image.fromarray( image_downscaled )
        # Extend the images with a black border
        imgExtended = extendImagePillow( img, scaledSide )
        im = array( img )
        imExtended = extendImageNumpy( im, scaledSide )
        nPixels = imExtended.shape[0] * imExtended.shape[1]
        features1 = np.reshape( imExtended, (1, nPixels) )
        features[arrayIndex] = features1

        # Load ROIs
        rois = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
        roisPoly = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
        # Scale the ROIs with a black border
        for key in rois.keys():
            rois[key]["xy"] = rois[key]["xy"] * scale
        # Extend the ROIs
        regionsExtended = extendRegions( rois, ( scaledSide, scaledSide ), img.size )
        # Get the mask from the extended images
        masksExtended = convertRegionsToMasks( regionsExtended, imExtended )
        for region in regionList:
            try:
                labels1 = np.reshape( masksExtended[region], (1, nPixels) )
                labels[region][arrayIndex] = labels1
            except:
                print("Error")
        arrayIndex = arrayIndex + 1

# Test images
        #fig1 = plt.figure(1, dpi=90)
        #plt.imshow( array(imExtended) )
        #showOverlayRegions( imExtended, regionsExtended, 5 )
        #fig2 = plt.figure(2, dpi=90)
        #plt.imshow( masksExtended["mb"], cmap='gray' )

    # Assume that each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels["cb"].shape[0]

    return features, labels



""" ----------------------------------------------------------------------- """
""" Setting up the network """
""" ----------------------------------------------------------------------- """

#import dataset

from skimage.viewer import ImageViewer

class Network(object):

    def __init__(self, nPixels):
        """
        
        """
        self.nPixels = nPixels
        nClasses = 1
        nOut = nPixels * nClasses
        # Input x
        self.x = tf.placeholder( tf.float32, shape = [None, nPixels], name = "x_nn" )
        # Predicted y
        self.y_ = tf.placeholder( tf.float32, shape = [None, nOut], name = "y_nn")

        # Linear model
        W = tf.Variable( tf.truncated_normal([nPixels, nOut], stddev=0.1) )
        b = tf.Variable( tf.constant(0.1, shape = [ 1 , nOut]) )
        self.y_conv = tf.matmul( self.x, W ) + b

        # Train and Evaluate the Model
        #
        # Cost is defined by the cross-entropy between predicted and real y
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits = self.y_conv, labels = self.y_))
        # Optimizer minimizes the cost
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        #self.perc_correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        
        self.x_image = tf.reshape(self.x, [-1, int(math.sqrt(self.nPixels)), int(math.sqrt(self.nPixels)), 1])
        self.y_conv_image = tf.reshape(self.y_conv, [-1, int(math.sqrt(self.nPixels)), int(math.sqrt(self.nPixels)), 1])
        self.mse = tf.reduce_sum( tf.square( tf.cast( tf.equal( self.y_conv, self.y_ ), tf.float32 ) ), reduction_indices = 1 ) 
        self.accuracy = tf.reduce_mean( tf.cast( tf.equal( self.y_conv, self.y_ ), tf.float32 ), reduction_indices = 1 )
        self.distance = tf.sqrt( tf.cast(self.mse, tf.float32 ) )
        #self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("Network initialization done")

    def show( self, training_data, training_labels, outputFolder ):
        
        for imageIndex in range(2):
            x = training_data[imageIndex].reshape(1,nPixels)
            y = training_labels[imageIndex].reshape(1,nPixels)
            image = self.x_image.eval(session=self.sess, feed_dict={self.x: x, self.y_: y})
            image = np.squeeze(image, axis=0)
            image = np.squeeze(image, axis=2)
            print(image.shape)
            print(image)
            simage = img_as_uint(image)
            #viewer = ImageViewer(simage)
            #viewer.show()
            imageSavePath = os.path.join( outputFolder, "save_" + str(imageIndex) + ".png" )
            io.imsave( imageSavePath, simage)
#            plt.show()


        #Image.fromarray(np.asarray(simage)).show()


    def train( self, training_data, training_labels, training_epochs, display_step, outputFolder ):
        
        nImagesTraining = training_data.shape[0]
        for epoch in range(training_epochs):
            for imageIndex in range(nImagesTraining):
                x = training_data[imageIndex].reshape(1,nPixels)
                y = training_labels[imageIndex].reshape(1,nPixels)
                self.train_accuracy = self.accuracy.eval(session=self.sess, feed_dict={self.x: x, self.y_: y})
                self.train_mse = self.mse.eval(session=self.sess, feed_dict={self.x: x, self.y_: y})
                self.train_distance = self.distance.eval(session=self.sess, feed_dict={self.x: x, self.y_: y})
                image = self.y_conv_image.eval(session=self.sess, feed_dict={self.x: x, self.y_: y})
                image = np.squeeze(image, axis=0)
                image = np.squeeze(image, axis=2)
                imageSavePath = os.path.join( outputFolder, "train_prob_" + str(imageIndex) + "_epoch_" + str(epoch) + ".png" )
                io.imsave( imageSavePath, image)
                #io.imshow(image)
                #print(image.shape)
                #print(image)
                #Image.fromarray(np.asarray(img_as_uint(image))).show()
                print( "Epoch: " + str(epoch) + ", image: " + str(imageIndex) + ", Accuracy: " + str(self.train_accuracy[0])  + ", MSE: " + str(self.train_mse[0]) + ", Accuracy: " + str(self.train_distance[0]) )

"""    
    def predict( self, data ):

        nImagesPredict = data.shape[0]
            for imageIndex in range(nImagesPredict):

                test_cost = []
                test_accuracy = []
                for batch_num, (input_X, labels_y) in enumerate(epoch):
                    cost, accuracy = model.step(sess,
                                                input_X, labels_y,
                                                dropout_value_conv=0.0,
                                                dropout_value_hidden=0.0,
                                                forward_only=True)
                    test_cost.append(cost)
                    test_accuracy.append(accuracy)
 
                print "Validation:"
                print "Epoch: %i, batch: %i, cost: %.3f, accuarcy: %.3f" % (
                    epoch_num, batch_num,
                    np.mean(test_cost), np.mean(test_accuracy))
"""     
"""
        # tf Graph Input
        X = tf.placeholder( tf.float32, shape=[None, nPixels], name="X_train" )
        Y = tf.placeholder( tf.float32, shape=[None, nPixels], name="Y_train" )
        #nImagesTraining = train_X.shape.as_list()[0]
        nImagesTraining = train_X.shape[0]
        # Start training
        with tf.Session() as sess:
    
            # Run the initializer
            sess.run(init)
    
            print( X.shape )
            print( train_X.shape )
    
            # Fit all training data
    
            for epoch in range(training_epochs):
                for imageIndex in range(nImagesTraining):
                    x = train_X[imageIndex].reshape(1,nPixels)
                    y = train_Y[imageIndex].reshape(1,nPixels)
                    sess.run( optimizer, feed_dict={X: x, Y: y})
    
                # Display logs per epoch step
                if (epoch+1) % display_step == 0:
                    c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                        "W=", sess.run(W), "b=", sess.run(b))
    
            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

# Interactive session vs normal?
sess = tf.InteractiveSession()
"""


#%%
""" ----------------------------------------------------------------------- """
""" Generating data """
""" ----------------------------------------------------------------------- """

# server
imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
#imageFolder = "/home/mbarbier/Documents/data/test_ref_small/DAPI/reference_images"
#roisFolder = "/home/mbarbier/Documents/data/test_ref_small/DAPI/reference_rois"
imageFormat = "png"

# laptop platsmurf
#imageFolder = "/home/mbarbier/Documents/prog/SliceMap/dataset/input/reference_images"
#roisFolder = "/home/mbarbier/Documents/prog/SliceMap/dataset/input/reference_rois"
#imageFormat = "tif"

binning = 64
regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"

# Load pre-generated data if it exists else generate and save it
contains = ""#"01"

try:
    print( "Loading pre-generated features and labels data" )
    features = np.load( os.path.join( dataFolder, "features.npy" ) ).astype(np.float32)
    labels = {}
    for region in regionList:
        labels[region] = np.load( os.path.join( dataFolder, "labels_" + region + ".npy" ) ).astype(np.float32)
except:
    print( "No pre-generated features and labels data available, generating new data" )
    features, labels = featureAndLabels( imageFolder, imageFormat, roisFolder, binning, regionList, contains )
    np.save( os.path.join( dataFolder, "features.npy" ), features )
    for region in regionList:
        np.save( os.path.join( dataFolder, "labels_" + region + ".npy" ), labels[region] )


#%%
""" ----------------------------------------------------------------------- """
""" Data to neural network """
""" ----------------------------------------------------------------------- """



training_n = 10
training_data = features[0:training_n]
training_labels = labels["cb"][0:training_n]
#a0 = training_data
#a1 = training_data.astype('uint8')
#a2 = np.asarray( a1 ).reshape(-1,84)
#i2 = Image.fromarray( a1 )
#i2.show()
#i2.save( os.path.join(dataFolder, "test.png" ) )

nPixels = features.shape[1]
nn = Network(nPixels)
#nn.show( training_data, training_labels, dataFolder )
training_epochs = 100
display_step = 1
nn.train( training_data, training_labels, training_epochs, display_step, dataFolder )



"""
with tf.Session() as sess:

    data_initializer = tf.placeholder(dtype=training_data.dtype, shape=training_data.shape)
    label_initializer = tf.placeholder(dtype=training_labels.dtype, shape=training_labels.shape)

    input_data = tf.Variable(data_initializer, trainable=False, collections=[])
    input_labels = tf.Variable(label_initializer, trainable=False, collections=[])

    sess.run(input_data.initializer,
           feed_dict={data_initializer: training_data})
    sess.run(input_labels.initializer,
           feed_dict={label_initializer: training_labels})
"""

    
#features_placeholder = tf.placeholder(features.dtype, features.shape)
#labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

#dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
### [Other transformations on `dataset`...]
### dataset = ...
##iterator = dataset.make_initializable_iterator()
#iterator = dataset.dataset.make_one_shot_iterator()

#sess.run( iterator.initializer, feed_dict={features_placeholder: features,
#    labels_placeholder: labels})
""" ----------------------------------------------------------------------- """




## Train
#for _ in range(nIteration):
#    batch_xs, batch_ys = iterator.next_batch(nBatch)
#    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    
    
    
"""
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: slices1.test.images,
                                      y_: slices1.test.labels}))
"""


#roi["xy"] = roi["xy"] * scale

#fig2 = plt.figure(2, dpi=90)
#plt.imshow(image, cmap='gray' )

#fig3 = plt.figure(3, dpi=90)
#plt.imshow(image_downscaled, cmap='gray')

#showOverlayRegions( image_downscaled, rois, 3 )


#polygon = Polygon( regionsExtended["cx"]["xy"] )
#mask = rasterizePolygon( imExtended, polygon )






#dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"
#dataName = "reference_stack_overlay_333"
#rois = loadRegionsStack( dataFolder + "/" + dataName + ".zip", "B21")
#images = loadImage( dataFolder + "/" + dataName + ".tif")
