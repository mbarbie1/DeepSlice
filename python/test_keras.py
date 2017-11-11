# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras import backend as K

inputs = K.placeholder(shape=(2, 4, 5))

import numpy as np
val = np.random.random((3, 4, 5))
#var = K.variable(value=val)

# all-zeros variable:
#var = K.zeros(shape=(3, 4, 5))
# all-ones:
#var = K.ones(shape=(3, 4, 5))

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

print(model)