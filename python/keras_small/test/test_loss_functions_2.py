# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:39:33 2018

@author: mbarbier
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Input
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
 
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def freq_mse( y_true, y_pred ):
    weights = K.variable( value = np.array([[ 2,   0.5 ]]) )
    print(K.int_shape(weights))
    weights = K.transpose( weights )
    print(K.int_shape(weights))
    mse_mean_1 = K.mean(K.square(y_pred - y_true), axis=1)
    print(K.int_shape(mse_mean_1))
    #print(K.eval(mse_mean_1))
    mse_w = K.dot( mse_mean_1, weights )
    return mse_w#K.mean( mse_w )

def freq1_mse( y_true, y_pred ):
    weights = K.variable( value = np.array([[ 1,   0 ]]) )
    print(K.int_shape(weights))
    weights = K.transpose( weights )
    print(K.int_shape(weights))
    mse_mean_1 = K.mean(K.square(y_pred - y_true), axis=1)
    print(K.int_shape(mse_mean_1))
    #print(K.eval(mse_mean_1))
    mse_w = K.dot( mse_mean_1, weights )
    return mse_w#K.mean( mse_w )

def freq2_mse( y_true, y_pred ):
    weights = K.variable( value = np.array([[ 0,   1 ]]) )
    print(K.int_shape(weights))
    weights = K.transpose( weights )
    print(K.int_shape(weights))
    mse_mean_1 = K.mean(K.square(y_pred - y_true), axis=1)
    print(K.int_shape(mse_mean_1))
    #print(K.eval(mse_mean_1))
    mse_w = K.dot( mse_mean_1, weights )
    return mse_w#K.mean( mse_w )
 
#def catacc(y_true, y_pred):
#	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
 
# prepare sequence
X = np.transpose( np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]) )
X = np.expand_dims(X, axis=0)
Y = np.transpose( np.array([
                [ 1,   1,   1,   1,   1,   1,   1,   0,   0,   0],
                [ 0,   0,   0,   0,   0,   0,   0,   1,   1,   1]
              ]) )
# create model
Y = np.expand_dims(Y, axis=0)
nClasses = 2
nCols = 10
inputs = Input(( nCols, 1))
conv1 = Conv1D(32, (1), activation='relu', padding='same')(inputs)
conv10 = Conv1D(nClasses, (1), activation='sigmoid')(conv1)
model = Model(inputs=[inputs], outputs=[conv10])

#model = Sequential()
#model.add( Dense( 100, input_shape = (1, 10), activation='relu') )
#model.add( Conv1D( 10, (1), activation='sigmoid') )#, input_shape = (1, 10)

model.compile(loss='mse', optimizer='adam', metrics = [ rmse, freq_mse, freq1_mse, freq2_mse ])

# train model
history = model.fit(X, Y, epochs=500, batch_size=len(X), verbose=2)

y_pred = model.predict( X, batch_size=None, verbose=0)
y_pred_label = np.argmax(y_pred, axis = 2)

plt.scatter(X, y_pred_label, c="g", alpha=0.5, label="pred 1")
plt.xlabel("Leprechauns")
plt.ylabel("Gold")
plt.legend(loc=2)
plt.show()

# plot metrics
plt.figure()
plt.plot(history.history['rmse'])
plt.show()

# plot metrics
plt.figure()
plt.plot(history.history['freq_mse'])
plt.show()

# plot metrics
plt.figure()
plt.plot(history.history['freq1_mse'])
plt.show()

# plot metrics
plt.figure()
plt.plot(history.history['freq2_mse'])
plt.show()
