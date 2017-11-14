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
import numpy as np
from ijroi import read_roi_zip as read_roi_zip_ijroi
from read_roi import read_roi_zip as read_roi_zip_read_roi

FLAGS = None

def loadRegionsStack( filePath, prefix ):
    """
    Read in regions for an image
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

def loadImage( imagePath ):
    images = io.imread( imagePath )
    return images

def loadLabeledImage( imagePath, roisPath ):
    images = loadImage( imagePath )
    rois = loadRegions( roisPath )
    return images, rois

imageFolder = "/home/mbarbier/Documents/prog/SliceMap/input/reference_images"
roisFolder = "/home/mbarbier/Documents/prog/SliceMap/input/reference_rois"
dataName = "reference_stack_overlay_333"
rois = loadRegionsStack( dataFolder + "/" + dataName + ".zip", "B21")
images = loadImage( dataFolder + "/" + dataName + ".tif")
