# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:36:33 2018

@author: mbarbier
"""
from module_data import loadImage
import numpy as np
import os
import cv2

dataFolder = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output_pres"
imgPath = os.path.join( dataFolder, "img.png" )
maskPath = os.path.join( dataFolder, "mask.png" )
predPath = os.path.join( dataFolder, "pred.png" )

def run( imgPath, maskPath, predPath ):
    
    print('-'*30)
    print('Load data')
    print('-'*30)
    img = loadImage( imgPath )
    mask = loadImage( maskPath )
    pred = loadImage( predPath )
    print('-'*30)
    print('Generate and save overlays')
    print('-'*30)
    makeOverlay( img, mask, "overlay_mask.png" )
    makeOverlay( img, pred, "overlay_pred.png" )

def makeOverlay( img, mask, outputName ):
    
    scaleIntensity = 500
    img = img.astype( np.float32 )
    img /= img.max()
    img = np.fmin( 255.0 * np.ones(img.shape, np.float32 ), scaleIntensity * img )
    img = img.astype( np.uint8 )
    mask = np.flip(mask, 2)    
    #imgColor = np.flip(img, 2 ) #cv2.cvtColor( img, cv2.COLOR_RGB2BGR )
    imgOverlay = cv2.addWeighted(img, 0.9, mask, 0.4, 0.0)
    output_path = os.path.join( dataFolder, outputName )
    cv2.imwrite( output_path, imgOverlay )


def main():

    print('-'*30)
    print('START presentation_overlay_masks')
    print('-'*30)
    run( imgPath, maskPath, predPath )
    print('-'*30)
    print('END presentation_overlay_masks')
    print('-'*30)

if __name__ == '__main__':
    main()
