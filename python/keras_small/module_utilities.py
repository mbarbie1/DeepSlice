# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:58:58 2017

@author: mbarbier
"""
import os#, shutil
import csv
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io
from module_data import loadImage, imageSizeList, maxImageSize
import tifffile as tiff


def writeData(filename, data):
    """
    MBARBIER: Taken/adapted from https://github.com/ChristophKirst/ClearMap/blob/master/ClearMap/IO/TIF.py
    
    Write image data to tif file
    
    Arguments:
        filename (str): file name 
        data (array): image data
    
    Returns:
        str: tif file name
    """

    d = len(data.shape);
    
    if d == 2:
        tiff.imsave(filename, data.transpose([0,1]));
    elif d == 3:   
        tiff.imsave(filename, data.transpose([2,0,1]), photometric = 'minisblack',  planarconfig = 'contig', bigtiff = True);
    elif d == 4:        
        #tiffile (z,y,x,c)
        tiff.imsave(filename, data.transpose([0,1,2,3]), photometric = 'minisblack',  planarconfig = 'contig', bigtiff = True);
    else:
        raise RuntimeError('writing multiple channel data to tif not supported!');

    return filename;

def writeImagePathToCsv( csvPath, imageId, batch, epoch, path ):

    file_exists = os.path.isfile( csvPath )
    with open( csvPath, 'a' ) as f:
        #writer = csv.writer( f )
        #writer.writerow( [ str(imageId), str(batch), str(epoch), path ] )
        writer = csv.DictWriter( f, fieldnames = [ "imageId", "batch", "epoch", "path" ])
        if not file_exists:
            writer.writeheader()
        row = { "imageId":str(imageId), "batch":str(batch), "epoch":str(epoch), "path":path }
        writer.writerow( row )

def writeDictToCsv( csvPath, dic ):

    file_exists = os.path.isfile( csvPath )
    with open( csvPath, 'a' ) as f:
        #writer = csv.writer( f )
        #writer.writerow( [ str(imageId), str(batch), str(epoch), path ] )
        writer = csv.DictWriter( f, fieldnames = dic.keys() )
        if not file_exists:
            writer.writeheader()
        row = dic
        writer.writerow( row )

def saveStackFromSelection( csvPath, imageId, batch, outputPath ):

    imagePathList = readImagePathsSelection( csvPath, imageId, batch )
    sizeList = imageSizeList( imagePathList )
    w, h = maxImageSize( sizeList )
    nImages = len( imagePathList )
    nChannels = 3
    stack = np.zeros( ( nImages, h, w, nChannels ), dtype=np.uint8 )
    for i in range( len(imagePathList) ):
        imagePath = imagePathList[i]
        img = io.imread( imagePath )
        stack[i,...] = img

    writeData(outputPath, stack)


def readImagePathsSelection( csvPath, imageId, batch ):

    # Read csv into pandas table
    df = pd.read_csv( csvPath )
    # Select rows with imageId and batch    
    sel = df[ (df['imageId'] == int(imageId)) & (df['batch'] ==  batch) ]
    # Select the column with the paths
    paths = sel.loc[:, 'path']
    # Convert to list
    imagesList = paths.tolist()
    return imagesList

def readImagePathsFromCsv( csvPath ):
    imageDescriptions = []
    with open( csvPath, 'r' ) as f:
        reader = csv.DictReader( f )
        for row in reader:
            imageDescriptions.append( row )
    return imageDescriptions

def removeFolderContents( folder ):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

"""
    TESTS testing the above module functions
"""

"""

csvPath = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output/test.csv"
imageId = "0"
batch = 0
epoch = 3
path = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/output/images/combined_index-0000_batch-0001_epoch-0003.png"
saveStackFromSelection( csvPath, imageId, batch, 'test.tif' )

#writeImagePathToCsv( csvPath, imageId, batch, epoch, path )
#
#df = pd.read_csv( csvPath )
#sel = df[ df['imageId'] < 1 ]
#
#imageList = readImagePathsFromCsv( csvPath )
#imageListpandas = readImagePathsSelection( csvPath, 1, 0 )


"""
