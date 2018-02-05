# -*- coding: utf-8 -*-
"""
Test data augmentation of the small slices data set

Created on Thu Dec 21 14:59:42 2017

@author: mbarbier
"""
from keras.preprocessing.image import ImageDataGenerator
#from keras import model
from data_small import loadData
from module_utilities import removeFolderContents
import numpy as np
import itertools

print('-'*30)
print('Loading train data...')
print('-'*30)
imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadData( 0.8, 16 )
images = np.expand_dims( imgs_train, axis = 3 )
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
    masks = masks + value * masks_all[region]
masks = np.expand_dims( masks, axis = 3 )

print('-'*30)
print('Fuse masks to single multi-label image')
print('-'*30)
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

# Compute the statistics for data augmentation purposes (centering, normalization)
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

# The folders to save the extra images and corresponding masks of the data augmentation
augm_masks_folder = '/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/augm_masks'
augm_images_folder = '/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/augm_images'
fake_y = np.zeros( images.shape[0] )
image_generator = image_datagen.flow(
    x = images,
    y = fake_y,
    save_to_dir = augm_images_folder,
    save_prefix = 'augm_image',
    save_format = 'png',
    seed=seed)
mask_generator = mask_datagen.flow(
    x = masks,
    y = fake_y,
    save_to_dir = augm_masks_folder,
    save_prefix = 'augm_mask',
    save_format = 'png',
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# here's a more "manual" example
# Empty the dirs
removeFolderContents(augm_images_folder)
removeFolderContents(augm_masks_folder)
epochs = 20
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in train_generator:
        print('batch ' + str(batches) + ' : do nothing, save something?')
        #model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(images) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

##model.fit_generator(
##    train_generator,
##    steps_per_epoch=20,
##    epochs=20)

#layer_dict = dict([(layer.name, layer) for layer in model.layers])