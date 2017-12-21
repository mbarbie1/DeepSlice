# -*- coding: utf-8 -*-
"""
Test data augmentation of the small slices data set

Created on Thu Dec 21 14:59:42 2017

@author: mbarbier
"""
from keras.preprocessing.image import ImageDataGenerator
#from keras import model
from data_small import loadData
import numpy as np
import itertools

print('-'*30)
print('Loading train data...')
print('-'*30)
imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadData( 0.8, 16 )
images = imgs_train
masks_all = imgs_mask_train_all

print('-'*30)
print('Fuse masks to single multi-label image')
print('-'*30)
regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
regionIdList = range(1, len(regionList)+1)
masks = np.zeros( masks_all[regionList[0]].shape )
for index in range(len(regionList)):
    region = regionList[index]
    value = regionIdList[index]
    masks = value * masks_all[region]

print('-'*30)
print('Fuse masks to single multi-label image')
print('-'*30)
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = itertools.izip(image_generator, mask_generator)
#train_generator = zip(image_generator, mask_generator)

#model.fit_generator(
#    train_generator,
#    steps_per_epoch=20,
#    epochs=20)
