# -*- coding: utf-8 -*-
"""
TODO : this is not at all finished


Created on Fri Jan 12 14:36:33 2018

@author: mbarbier
"""
import numpy as np
import os
import cv2


def run( plotFolder ):
    flag.run_id = 'save_images_image-size-%d' % ( flag.image_size )
    flag.output_run_dir = os.path.join( flag.output_dir, flag.run_id )
    makeDirs( flag )
    showData( flag )

def loadAsGray( imgPath ):
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calculateOverlap( maskPath, predPath ):

    mask = loadImage( maskPath )
    pred = loadImage( predPath )
   
def main():

    run( flag )
    #train_op = Unet_train.TrainModel(flag)
    #train_op.train_unet()

if __name__ == '__main__':
    main()
