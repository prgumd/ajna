#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
import Misc.TFUtils as tu
from Misc.Decorators import *
import Misc.warpICSTN2 as warp2

# TODO: Add training flag
    
class BaseLayers(object):
    def __init__(self):
        self.CurrBlock = 0
    # Decorator to count number of functions have been called
    # Ideas from
    # https://stackoverflow.com/questions/13852138/how-can-i-define-decorator-method-inside-class
    # https://stackoverflow.com/questions/41678265/how-to-increase-a-number-every-time-a-function-is-run

    @CountAndScope
    @add_arg_scope
    def ConvBNReLUBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv =  self.Conv(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    @add_arg_scope
    def ConvTransposeBNReLUBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv =  self.ConvTranspose(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    @add_arg_scope
    def Conv(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, activation=None, name=None):
        Output = tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size, \
                                  strides = strides, padding = padding, activation=activation, name=name) # kernel_constraint=tf.keras.constraints.UnitNorm(axis=0), #  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        return Output

    @CountAndScope
    @add_arg_scope
    def ConvTranspose(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, activation=None, name=None):
        Output = tf.layers.conv2d_transpose(inputs = inputs, filters = filters, kernel_size = kernel_size, \
                                  strides = strides, padding = padding, activation=activation, name=name) # kernel_constraint=tf.keras.constraints.UnitNorm(axis=0), #  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        return Output

    @CountAndScope
    @add_arg_scope
    def BN(self, inputs = None):
        Output = tf.layers.batch_normalization(inputs = inputs) 
        return Output
    
    @CountAndScope
    @add_arg_scope
    def ReLU(self, inputs = None):
        Output = tf.nn.relu(inputs)
        return Output

    @CountAndScope
    @add_arg_scope
    def Concat(self, inputs = None, axis=0):
        Output = tf.concat(values = inputs, axis = axis)
        return Output

    @CountAndScope
    @add_arg_scope
    def Flatten(self, inputs = None):
        # https://stackoverflow.com/questions/37868935/tensorflow-reshape-tensor
        Shape = inputs.get_shape().as_list()       
        Dim = np.prod(Shape[1:])        
        Output = tf.reshape(inputs, [-1, Dim])         
        return Output

    @CountAndScope
    @add_arg_scope
    def Dropout(self, inputs = None, rate = None):
        if(rate is None):
            rate = 0.5
        Output = tf.layers.dropout(inputs, rate=rate)
        return Output

    @CountAndScope
    @add_arg_scope
    def Dense(self, inputs = None, filters = None, activation=None, name=None):
        Output = tf.layers.dense(inputs, units = filters, activation=activation, name=name) # kernel_constraint=tf.keras.constraints.UnitNorm(axis=0),
        return Output
