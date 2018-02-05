# -*- coding: utf-8 -*-
"""
Test data augmentation of the small slices data set

Created on Thu Dec 21 14:59:42 2017

@author: mbarbier
"""
#from keras import model
from data_small import loadData
import numpy as np
from keras import backend as K
from module_model_unet import get_unet, preprocess
from module_callbacks import trainCheck
import matplotlib.pyplot as plt

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

#img_rows = 384
#img_cols = 384
img_rows = 96
img_cols = 96
n_epochs = 30

from scipy.misc import imsave
# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


print('-'*30)
print('Loading train data...')
print('-'*30)
imgs_train, imgs_test, imgs_mask_train_all, imgs_mask_test_all, imgs_id_test, imgs_id_train = loadData( 0.8, 16 )
images = imgs_train#np.expand_dims( imgs_train, axis = 3 )
#images = np.expand_dims( images, axis = 3 )
images = preprocess(images, img_rows, img_cols )
masks_all = imgs_mask_train_all

print('-'*30)
print('Fuse masks to single multi-label image')
print('-'*30)
regionList = [ "cb", "hp", "cx", "th", "mb", "bs" ]
regionIdList = range(1, len(regionList)+1)
masks = np.zeros( masks_all[regionList[0]].shape )
#masks = np.expand_dims( masks, axis = 3 )
masks = preprocess(masks, img_rows, img_cols )

print('-'*30)
print('Load unet model')
print('-'*30)
model = get_unet( img_rows, img_cols )
layer_dict = dict([(layer.name, layer) for layer in model.layers])

#print('-'*30)
#print('Load unet model')
#print('-'*30)
#img = images[0]
#img_float = img.astype(np.float32)
#img_float = deprocess_image(img_float)
#layer_name = 'testLayerName'
#filter_index = 0
#imsave('%s_filter_%d.png' % (layer_name, filter_index), img_float)


# Compile model
#model = get_unet( img_rows, img_cols )
#.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)


if not os.path.isdir(flag.output_dir):
    os.mkdir(flag.output_dir)

show_pred_masks = trainCheck(flag)
history = model.fit(images, masks, batch_size=32, epochs=n_epochs, verbose=1, shuffle=True,
              validation_split=0.2, callbacks=[show_pred_masks])


print(history.history.keys())              
# summarize history for accuracy
def showTrainingHistory(history):
    plt.figure()
    plt.plot(history.history['pixelwise_binary_ce'])
    plt.plot(history.history['val_pixelwise_binary_ce'])
    plt.title('model pixelwise_binary_ce')
    plt.ylabel('pixelwise_binary_ce')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['pixelwise_l2_loss'])
    plt.plot(history.history['val_pixelwise_l2_loss'])
    plt.title('model pixelwise_l2_loss')
    plt.ylabel('pixelwise_l2_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# list all data in history
print(history.history.keys())
showTrainingHistory(history)