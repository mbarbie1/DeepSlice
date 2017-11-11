# -*- coding: utf-8 -*-
"""

"""
# Python
import tensorflow as tf
import sys

def is_virtual():
    """ Return if we run in a virtual environtment. """
    # Check supports venv && virtualenv
    return (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix or hasattr(sys, 'real_prefix'))

print( 'Virtual: ' + str(is_virtual()) )
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print( sess.run(hello) )

