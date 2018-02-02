# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:27:11 2018

@author: mbarbier
"""

import keras
import cv2
import numpy as np
import os
from glob import glob
from module_data import mergeLabelImage
from module_utilities import writeImagePathToCsv, saveStackFromSelection
from module_utilities import writeDictToCsv
from keras import backend as K

class trainCheck(keras.callbacks.Callback):

    def __init__(self, flag):
        self.flag = flag
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        print("train end")
        #par = self.params
        csvPath = self.flag.images_csv_path
        self.train_seg_save( csvPath )
        #for imageIndex        
        #for batch = 0
        #self.combineToStacks( self, batch )
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return
#        print("begin batch")
#        print(self)

    def on_batch_end(self, batch, logs={}):
        self.train_seg_mem( self.model, self.epoch, batch )
        self.train_region_dice_overlap_save( self.model, self.epoch, batch )
        return

    def train_seg_save( self, csvPath ):
        maxImageIndex = 1
        nBatches = 1
        #epochs = self.nb_epoch
        try:
            nImages = self.params['samples']
            nImages = min( nImages, maxImageIndex )
        except:
            nImages = maxImageIndex
        for batch in range( nBatches ):
            for imageIndex in range( nImages ):
                fileName = 'stack_index-%04d_batch-%04d.tif' % ( imageIndex, batch)
                outputPath = os.path.join( self.flag.output_plots_dir, fileName )
                saveStackFromSelection( csvPath, imageIndex, batch, outputPath )
        return

    def train_region_dice_overlap_save( self, model, epoch, batch ):
        maxImageIndex = 100
        print(self)
        try:
            xs = self.validation_data[0]
            ys = self.validation_data[1]
        except:
            xs = K.eval(self.model.inputs[0])
            ys = self.model.targets[0]
        ys_pred = self.model.predict( xs )
        image_size = self.flag.image_size

        # Save the predicted output prob, image, and original mask
        overlap_list = []
        output_path_csv = os.path.join( self.flag.output_plots_dir, 'overlap_batch-%04d_epoch-%04d.csv' % ( batch, epoch) )
        for imageIndex in range( min(maxImageIndex, xs.shape[0]) ):
            x = xs[imageIndex,...]
            y = ys[imageIndex,...]
            y_pred = ys_pred[imageIndex,...]
            #if multi-class y then merge them by giving different values: e.g. cx = 2, cb = 1, bg = 0
            nClasses = y.shape[2]
            overlap_regions = {}
            output_path_combined = os.path.join( self.flag.output_images_dir, 'combined_index-%04d_batch-%04d_epoch-%04d.png' % ( imageIndex, batch, epoch) )
            for regionIndex in range( nClasses ):
                regionMask = y[:,:,regionIndex].astype( np.float32 )
                regionPred = y_pred[:,:,regionIndex].astype( np.float32 )
                overlap_regions["epoch"] = str(epoch)
                overlap_regions["batch"] = str(batch)
                overlap_regions["image_index"] = str(imageIndex)
                overlap_regions["region_index"] = str(regionIndex)
                overlap_regions["region_name"] = self.flag.extended_region_list[regionIndex]
                overlap_regions["area_real"] = str( regionMask.sum() )
                overlap_regions["area_pred"] = str( regionPred.sum() )
                overlap_regions["overlap"] = str( ( ( 2.0 * ( regionMask * regionPred ).sum() ) / ( regionMask.sum() + regionPred.sum() ) ) )
                overlap_regions["reference_image"] = output_path_combined
                writeDictToCsv( output_path_csv, overlap_regions )
            overlap_list.append( overlap_regions )
        

    def train_seg_mem( self, model, epoch, batch ):
        maxImageIndex = 100
        print(self)
        try:
            xs = self.validation_data[0]
            ys = self.validation_data[1]
        except:
            xs = K.eval(self.model.inputs[0])
            ys = self.model.targets[0]
        ys_pred = self.model.predict( xs )
        image_size = self.flag.image_size
        #print("Shape xs: " + str(xs.shape) )
        #print("Shape ys: " + str(ys.shape) )
        #print("Shape pred ys: " + str(ys_pred.shape) )

        # Save the predicted output prob, image, and original mask
        for imageIndex in range( min(maxImageIndex, xs.shape[0]) ):
            x = xs[imageIndex,...]
            y = ys[imageIndex,...]
            y_pred = ys_pred[imageIndex,...]
            #if multi-class y then merge them by giving different values: e.g. cx = 2, cb = 1, bg = 0
            if y.shape[2] > 1:
                y = mergeLabelImage(y).astype( np.float32 )
                y_pred = mergeLabelImage(y_pred).astype( np.float32 )
                y /= y.max()
                y_pred /= y_pred.max()
            #print( "x min: " + str(x.min()) +", x max: " + str(x.max()) )
            #print( "y min: " + str(y.min()) +", y max: " + str(y.max()) )
            #print( "y_pred min: " + str(y_pred.min()) +", y_pred max: " + str(y_pred.max()) )
            imageName = str(imageIndex)
            imgMask = (y_pred*255).astype( np.uint8 )
            imgMaskOri = (y*255).astype( np.uint8 )
            #print( "y min: " + str(y.min()) +", y max: " + str(y.max()) )
            img = (x).astype( np.uint8 )
            imgColor = cv2.cvtColor( x, cv2.COLOR_GRAY2BGR )
            imgMaskColor = cv2.applyColorMap( imgMask, cv2.COLORMAP_JET )
            imgMaskOriColor = cv2.applyColorMap( imgMaskOri, cv2.COLORMAP_JET )
            output_path_combined = os.path.join( self.flag.output_images_dir, 'combined_index-%04d_batch-%04d_epoch-%04d.png' % ( imageIndex, batch, epoch) )
            image_combined = np.concatenate( (imgColor, imgMaskColor, imgMaskOriColor), axis=1 )
            cv2.imwrite( output_path_combined, image_combined )
            writeImagePathToCsv( self.flag.images_csv_path, imageIndex, batch, epoch, output_path_combined )
            #imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.4, 0.0)
            #output_path_mask = os.path.join( self.flag.output_images_dir, 'mask_index-%04d_batch-%04d_epoch-%04d.png' % ( imageIndex, batch, epoch)  )
            #output_path_image = os.path.join( self.flag.output_images_dir, 'image_index-%04d_batch-%04d_epoch-%04d.png' % ( imageIndex, batch, epoch) )
            #output_path_pred = os.path.join( self.flag.output_images_dir, 'pred_index-%04d_batch-%04d_epoch-%04d.png' % ( imageIndex, batch, epoch) )
            #cv2.imwrite( output_path_image, imgColor )
            #cv2.imwrite( output_path_pred, imgMaskColor )
            #cv2.imwrite( output_path_mask, imgMaskOriColor )
#        for x 
#        input_data = xs.reshape((1,image_size,image_size,self.flag.color_mode*2+1))
#        t_start = cv2.getTickCount()
#        result = model.predict(input_data, 1)
#        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
#        print("[*] Predict Time: %.3f ms"%t_total)
#        imgMask = (result[0]*255).astype(np.uint8)
#        imgShow = cv2.cvtColor(imgInput, cv2.COLOR_GRAY2BGR)
#        imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
#        imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.4, 0.0)
#        output_path = os.path.join(self.flag.output_dir, '%04d_'%epoch+os.path.basename(image_name))
#        cv2.imwrite(output_path, imgShow)


    def train_visualization_seg( self, model, epoch ):

        image_name_list = sorted(glob(os.path.join( self.flag.data_path,'train/IMAGE/*/*.png')))
        print (image_name_list)

        image_name = image_name_list[-1]
        image_size = self.flag.image_size

        imgInput = cv2.imread(image_name, self.flag.color_mode)
        output_path = self.flag.output_dir
        input_data = imgInput.reshape((1,image_size,image_size,self.flag.color_mode*2+1))

        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print("[*] Predict Time: %.3f ms"%t_total)

        imgMask = (result[0]*255).astype(np.uint8)
        imgShow = cv2.cvtColor(imgInput, cv2.COLOR_GRAY2BGR)
        imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
        imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.4, 0.0)
        output_path = os.path.join(self.flag.output_dir, '%04d_'%epoch+os.path.basename(image_name))
        cv2.imwrite(output_path, imgShow)
        # print "SAVE:[%s]"%output_path
        # cv2.imwrite(os.path.join(output_path, 'img%04d.png'%epoch), imgShow)
        # cv2.namedWindow("show", 0)
        # cv2.resizeWindow("show", 800, 800)
        # cv2.imshow("show", imgShow)
# cv2.waitKey(1)