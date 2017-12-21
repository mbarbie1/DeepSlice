# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Module
    Loading images and ROI into a  
    Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""
from __future__ import division

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
'''
def featureAndLabelsPatch( imageFolder, imageFormat, roisFolder, patchesPerRegion, patchSize, regionList, fileNameContains ):

    # Get all the images in the imageFolder
    imagePathList = findFiles( imageFolder, imageFormat, contains )
    return features, labels
'''

def featureAndLabels( imageFolder, imageFormat, roisFolder, binning, regionList, contains, reduceDimension ):
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
    if reduceDimension:
        features = np.zeros( (nImages, nPixels), float )
        labels = {}
        for region in regionList:
            labels[region] = np.zeros( (nImages, nPixels), int )
    else:
        features = np.zeros( (nImages, scaledSide, scaledSide), float )
        labels = {}
        for region in regionList:
            labels[region] = np.zeros( (nImages, scaledSide, scaledSide), int )

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
        #imgExtended = extendImagePillow( img, scaledSide )
        im = array( img )
        imExtended = extendImageNumpy( im, scaledSide )
        nPixels = imExtended.shape[0] * imExtended.shape[1]

        if reduceDimension:
            features1 = np.reshape( imExtended, (1, nPixels) )
            features[arrayIndex] = features1
        else:
            features[arrayIndex] = imExtended
            

        # Load ROIs
        rois = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
        #roisPoly = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
        # Scale the ROIs with a black border
        for key in rois.keys():
            rois[key]["xy"] = rois[key]["xy"] * scale
        # Extend the ROIs
        regionsExtended = extendRegions( rois, ( scaledSide, scaledSide ), img.size )
        # Get the mask from the extended images
        masksExtended = convertRegionsToMasks( regionsExtended, imExtended )
        for region in regionList:
            try:
                if reduceDimension:
                    labels1 = np.reshape( masksExtended[region], (1, nPixels) )
                    labels[region][arrayIndex] = labels1
                else:
                    labels[region][arrayIndex] = masksExtended[region]
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
""" Generating data """
""" ----------------------------------------------------------------------- """

def lazyGenerateSmall( imageFolder, roisFolder, dataFolder, imageFormat, contains, binning, regionList, reduceDimension ):
    """
    Load pre-generated data if it exists else generate and save it
    features = intensity of downscaled images, reshaped in 1 dimension
    labels = for each region the mask of the downscaled image, reshaped in 1 dimension
    
        imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
        roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
        imageFormat = "png"
        binning = 64
        regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
        dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"
        contains = ""#"01"
    """
    try:
        print( "Loading pre-generated features and labels data" )
        features = np.load( os.path.join( dataFolder, "features.npy" ) ).astype(np.float32)
        labels = {}
        for region in regionList:
            labels[region] = np.load( os.path.join( dataFolder, "labels_" + region + ".npy" ) ).astype(np.float32)
    except:
        print( "No pre-generated features and labels data available, generating new data" )
        features, labels = featureAndLabels( imageFolder, imageFormat, roisFolder, binning, regionList, contains, reduceDimension )
        np.save( os.path.join( dataFolder, "features.npy" ), features )
        for region in regionList:
            np.save( os.path.join( dataFolder, "labels_" + region + ".npy" ), labels[region] )
            
    return features, labels


def lazyGeneratePatch( imageFolder, roisFolder, dataFolder, imageFormat, contains, binning, regionList ):
    """
    Load pre-generated data if it exists else generate and save it
    features = intensity of downscaled images, reshaped in 1 dimension
    labels = for each region the mask of the downscaled image, reshaped in 1 dimension
    
        imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
        roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
        imageFormat = "png"
        binning = 64
        regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
        dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"
        contains = ""#"01"
    """
    try:
        print( "Loading pre-generated features and labels data" )
        features = np.load( os.path.join( dataFolder, "patch_features.npy" ) ).astype(np.float32)
        labels = {}
        for region in regionList:
            labels[region] = np.load( os.path.join( dataFolder, "patch_labels.npy" ) ).astype(np.float32)
    except:
        print( "No pre-generated features and labels data available, generating new data" )
        features, labels = featureAndLabelsPatch( imageFolder, imageFormat, roisFolder, patchesPerRegion, patchSize, regionList, fileNameContains )
        np.save( os.path.join( dataFolder, "patch_features.npy" ), features )
        for region in regionList:
            np.save( os.path.join( dataFolder, "patch_labels.npy" ), labels[region] )
