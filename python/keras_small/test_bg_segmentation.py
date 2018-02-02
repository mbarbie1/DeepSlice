# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:41:03 2018

@author: mbarbier
"""
""" Clear all variables """
from IPython import get_ipython
get_ipython().magic('reset -sf') 

from module_utilities import writeData
from module_data import mergeLabelImage, loadImage, findFiles, convertRegionsToMasks, loadRegionsImage
import numpy as np
import os
import cv2
import argparse
from matplotlib import pyplot as plt
import scipy.misc as misc

def filter_variance_window(img, wlen):
  wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),
    borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
  return wsqrmean - wmean*wmean

def normalize_to_uint8( image ):
    image = image.astype( np.float32 )
    image /= image.max()
    image = 255 * image
    image = image.astype(np.uint8)
    return image

def threshold_manual( image, minValue ):
    
    mask = np.zeros( image.shape, np.uint8 )
    mask[image >= minValue] = 1
    return mask

def close( image, kernelSize ):
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, ( kernelSize, kernelSize ) )
    image_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image_closed

def erode( image, kernelSize, nIterations ):
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, ( kernelSize, kernelSize ) )
    image_erosion = cv2.erode( image, kernel, iterations = nIterations )
    return image_erosion

def segmentation_bg( image ):

    image = image.astype(np.float32)
    image /= image.max()
    winSize = 100
    image_var = filter_variance_window( image, winSize )
    image_var = normalize_to_uint8( image_var )
    
    mask = threshold_manual( image, 0.01 )
    #plt.figure()
    #plt.imshow( mask * 255, cmap='gray' )

    image_gauss = cv2.GaussianBlur( image, (51,51), 0 )
    mask = threshold_manual( image_gauss, 0.01 )
    mask = close( mask, 51 )
    mask = erode( mask, 6, 3 )
    #plt.figure()
    #plt.imshow( mask * 255, cmap='gray' )
    imgShow = cv2.addWeighted( normalize_to_uint8( image ), 0.9, normalize_to_uint8( mask ), 0.4, 0.0)
    #plt.figure()
    #plt.imshow( imgShow )

    return image, mask, imgShow


def segmentation_sub_bg( mask, rois, regionListIndices ):

    maskRoi_list = convertRegionsToMasks( rois, mask )
    labels = np.zeros( mask.shape )
    bg_mask = np.copy( mask )
    for key in maskRoi_list.keys():
        maskRoi = maskRoi_list[key]
        labels[ maskRoi > 0 ] = regionListIndices[key]
        bg_mask[ maskRoi > 0 ] = 0
    labels[ bg_mask > 0 ] = regionListIndices["rest"]

    return bg_mask, labels


def labelOverlay( image, labels ):

    imageColor = cv2.cvtColor( normalize_to_uint8( image ), cv2.COLOR_GRAY2BGR )
    labelsColor = cv2.applyColorMap( normalize_to_uint8( labels ), cv2.COLORMAP_JET )
    overlay_regions = cv2.addWeighted( imageColor, 0.9, labelsColor, 0.4, 0.0)

    return overlay_regions


def init():
    restRegionFolder = "/home/mbarbier/Documents/prog/DeepSlice/data/rest_region_labels_2d"
    imageFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_images"
    roisFolder = "/home/mbarbier/Documents/data/reference_libraries/B31/DAPI/reference_rois"
    regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
    regionListIndices = { "rest":7, "cb":1, "hp":2, "cx":3, "th":4, "mb":5, "bs":6 }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_rest_region_dir", help="Rest region masks path", default=restRegionFolder)
    parser.add_argument("--original_image_dir", help="training images path", default=imageFolder)
    parser.add_argument("--original_rois_dir", help="training ROIs path", default=roisFolder)
    parser.add_argument("--output_dir", help="output directory", default="./output_bg_segmentation" )
    parser.add_argument("--image_format", help="Format of input and output images", default="png" )
    parser.add_argument("--region_list", help="List of ROIs", default=regionList )
    parser.add_argument("--region_list_indices", help="List of ROIs with corresponding indices", default=regionListIndices )
    flag = parser.parse_args()
    return flag



def run( flag ):
    makeDirs( flag )
    imagePathList = findFiles( flag.original_image_dir, flag.image_format, "" )
    imageIdList = list( map( lambda x: os.path.splitext(os.path.basename(x))[0], imagePathList ) )

    #imageIdList = [imageIdList[4]]
    region = "rest"

    for imageId in imageIdList:
        imagePath = os.path.join( flag.original_image_dir, imageId + "." + flag.image_format  )
        image = loadImage( imagePath )
        rois = loadRegionsImage( flag.original_rois_dir + "/" + imageId + ".zip", imageId )
        image, mask, image_overlay = segmentation_bg( image )
        bg_mask, labels = segmentation_sub_bg( mask, rois, flag.region_list_indices )
        image_overlay_regions = labelOverlay( image, labels )
        fileName = os.path.basename( imagePath )
        savePath = os.path.join( flag.output_dir, "overlay_regions_" + fileName)
        plt.imsave( savePath, image_overlay_regions )
        savePath = os.path.join( flag.output_dir, "bg_mask_" + fileName)
        #plt.imshow( bg_mask )
        plt.imsave( savePath, bg_mask )
        savePath = os.path.join( flag.output_dir, "overlay_" + fileName)
        plt.imsave( savePath, image_overlay )
        savePath = os.path.join( flag.output_dir, "mask_" + fileName)
        misc.imsave( savePath, mask )
        savePath = os.path.join( flag.output_dir, "image_" + fileName)
        misc.imsave( savePath, image )


def makeDirs( flag ):
    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )
    if not os.path.isdir( flag.output_dir ):
        os.mkdir( flag.output_dir )

def main():
    flag = init()
    run( flag )

if __name__ == '__main__':
    main()


"""    

    image_var = filter_variance_window( image, 51 )
    image_var = normalize_to_uint8( image_var )
    mask = threshold_manual( image_var, 0.01 )
    plt.figure()
    plt.imshow( mask * 255, cmap='gray' )


    image_med = cv2.medianBlur( image, 7 )
    mask = threshold_manual( image_med, 0.01 )
    plt.figure()
    plt.imshow( mask * 255, cmap='gray' )
    
    
    plt.figure()
    plt.imshow( image_var, cmap='gray' )
    image = 255 * image
    
    image = image.astype(np.uint8)

    image_var_med = cv2.medianBlur( image_var, 5 )
    image_med = cv2.medianBlur( image, 5 )
    
    plt.figure()
    plt.imshow( image_var_med, cmap='gray' )
    plt.figure()
    plt.imshow( image_med, cmap='gray' )


    plt.figure()
    plt.imshow( image, cmap='gray' )
    plt.axis('equal')
    ret, thresh = cv2.threshold( image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )
    plt.figure()
    plt.imshow( thresh, cmap='gray' )
    plt.axis('equal')
"""