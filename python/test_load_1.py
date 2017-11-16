# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
A neural network on small images for segmentation

Created on Sun Nov 12 11:14:40 2017

@author: mbarbier
"""
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
    origin = tuple( map( lambda x: int( round( (x[0]-x[1])/2 ) ), zip( largeSize, smallSize )) )
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

#def loadImages( folderPath, imageFormat ):

imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"

imagePathList = findFiles( imageFolder, "png", "" )
sizeList = imageSizeList( imagePathList )
maxWidth, maxHeight = maxImageSize( sizeList )
side = max( maxWidth, maxHeight )

imageId =  "B31-02"
imageFormat = "png"

rois = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
roisPoly = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )

image = loadImage( imageFolder + "/" + imageId + "." + imageFormat )


#Scale the image and Rois
binning = 64
scale = 1.0 / float(binning)
image = gaussian(image, sigma=(1.0/scale)/2.0)
image_downscaled = downscale_local_mean(image, (binning, binning))
for key in rois.keys():
    rois[key]["xy"] = rois[key]["xy"] * scale
img = Image.fromarray(image_downscaled)


imgExtended = extendImagePillow( img, int( round( scale * side ) ) )
im = array( img )
imExtended = extendImageNumpy( im, int( round( scale * side ) ) )
regionsExtended = extendRegions( rois, ( scale * side, scale * side ), img.size )
#fig1 = plt.figure(1, dpi=90)
#plt.imshow( array(imExtended) )
showOverlayRegions( imExtended, regionsExtended, 5 )

masksExtended = convertRegionsToMasks( regionsExtended, imExtended )

fig2 = plt.figure(2, dpi=90)
plt.imshow( masksExtended["mb"], cmap='gray' )



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
