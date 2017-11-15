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

import tensorflow as tf
from scipy import misc
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
import rasterio
import rasterio.features
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from ijroi import read_roi_zip as read_roi_zip_ijroi
from read_roi import read_roi_zip as read_roi_zip_read_roi

FLAGS = None


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
        [(polygon, 0)],
        out_shape=image.shape,
        fill=1,
        all_touched=True,
        dtype=np.uint8)
    return mask
    
def maskImage( image, mask ):
    maskedImage = np.ma.array( data=image, mask=mask.astype(bool))
    return maskedImage

imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
imageId =  "B31-02"
imageFormat = "png"

rois = loadRegionsImage( roisFolder + "/" + imageId + ".zip", imageId )
image = loadImage( imageFolder + "/" + imageId + "." + imageFormat )

#Scale the image and Rois
binning = 64
scale = 1.0 / float(binning)
image = gaussian(image, sigma=(1.0/scale)/2.0)
image_downscaled = downscale_local_mean(image, (binning, binning))
for key in rois.keys():
    rois[key]["xy"] = rois[key]["xy"] * scale
    #roi["xy"] = roi["xy"] * scale

fig1 = plt.figure(1, dpi=90)
plt.imshow(image, cmap='gray' )

fig4 = plt.figure(2, dpi=90)
plt.imshow(image_downscaled, cmap='gray')

showOverlayRegions( image_downscaled, rois, 3 )

polygon = Polygon( rois["cx"]["xy"] )
rasterizePolygon( image_downscaled, polygon )

#dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/data"
#dataName = "reference_stack_overlay_333"
#rois = loadRegionsStack( dataFolder + "/" + dataName + ".zip", "B21")
#images = loadImage( dataFolder + "/" + dataName + ".tif")
