from __future__ import print_function

# todo upgrade to keras 2.0
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Merge
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras import losses
# from keras.regularizers import ActivityRegularizer
from keras import backend as K

def segnet( nClasses=2 , optimizer=None, activationName='softmax', loss=losses.categorical_crossentropy, metrics = ['accuracy'], img_rows=96, img_cols=96 ):

    input_height = img_rows
    input_width = img_cols
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    nChannels = 1 #3 originally

    model = Sequential()
    model.add(Layer(input_shape=(input_height , input_width, nChannels )))

    # encoder
    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D( filter_size, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D( 128, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(256, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(ZeroPadding2D(padding=(pad,pad)))
    model.add(Convolution2D(512, (kernel, kernel), padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # decoder
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(512, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(256, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(128, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())

    model.add( UpSampling2D(size=(pool_size,pool_size)))
    model.add( ZeroPadding2D(padding=(pad,pad)))
    model.add( Convolution2D(filter_size, (kernel, kernel), padding='valid'))
    model.add( BatchNormalization())

    model.add(Convolution2D( nClasses , (1, 1), padding='valid',))

    model.outputHeight = model.output_shape[-3]
    model.outputWidth = model.output_shape[-2]
    print("Dimensions" + str(model.output_shape) )
    #model.add( Reshape(( model.output_shape[-3]*model.output_shape[-2], nClasses ), input_shape=( model.output_shape[-3], model.output_shape[-2], nClasses )))
    #print("Dimensions (reshape)" + str(model.output_shape) )
    #model.add(Permute((2, 1)))
    #print("Dimensions (permutation)" + str(model.output_shape) )
    model.add( Activation(activationName) )
    print("Dimensions (activation)" + str(model.output_shape) )
    #model.add( Reshape(( model.output_shape[-3]*model.output_shape[-2], nClasses ), input_shape=( model.output_shape[-2], model.output_shape[-1], nClasses )))

    if optimizer is None:
        model.compile(loss=loss, optimizer=Adam(lr=1e-5) , metrics=metrics )
    else:
        model.compile(loss=loss, optimizer=optimizer , metrics=metrics )

    return model
